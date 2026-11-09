"""总结模块：调用 DeepSeek（OpenAI SDK 兼容模式）生成结构化中文 Markdown 总结。

两阶段策略（Map-Reduce）：
    Map   : 把带时间戳的 transcript 按字数切成若干 block，逐块请求局部摘要
            （JSON 结构化输出：主题 / 摘要 / 带时间戳要点 / 术语 / 待解决问题）；
    Reduce: 把局部摘要拼成「片段索引」，一次请求全局总结；
            若索引仍超过上下文预算，则先分组二次压缩（可多层），再出全局总结。

时间戳原则：
    局部摘要的时间戳来自我们自己切 block 时记录的起止时间，不依赖模型生成；
    模型只在「要点」里挑选真实出现过的时间戳，未命中的会被 `_sanitize_points` 过滤，
    因此最终 Markdown 里的时间戳不会凭空出现。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.config import resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "4096"))
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "8192"))
# 单个 block 的字符上限：8000 字符 ≈ 2~3k tokens，DeepSeek 64k 上下文下很安全
SUMMARY_BLOCK_CHARS = int(os.getenv("SUMMARY_BLOCK_CHARS", "8000"))
# 全局聚合阶段允许的最大输入字符数，超出则先分组压缩
AGG_MAX_CHARS = int(os.getenv("AGG_MAX_CHARS", "24000"))
MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "180"))

# 相对路径按项目根目录解析（Windows 下 CWD 可能不是项目根），不写死任何绝对路径
OUTPUT_DIR = resolve_path(os.getenv("DATA_ROOT") or "data") / "outputs"


class SummarizationError(RuntimeError):
    """总结失败（上层翻译成 HTTP 500）。"""


# ---------------------------------------------------------------------------
# Prompt（用 %XXX% 占位符而不是 str.format，避免转录文本里的花括号炸掉模板）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一位资深的多语言内容分析师，擅长把口语化的音视频转录整理成可直接阅读的中文结构化总结。

硬性规则（必须全部遵守）：
1. 全部使用简体中文输出。人名、公司名、产品名、技术术语保留原文（英文/韩文/日文），首次出现时用括号给出中文解释，例如：Retrieval-Augmented Generation（检索增强生成）。
2. 时间戳必须保留，格式为 HH:MM:SS，只在「章节脉络」和「关键观点」中使用，且只能使用转录中真实出现过的时间点，严禁编造或估算。
3. 必须区分说话人。归纳观点时标明是谁说的（如「Speaker 2 认为…」）；不要把你自己的推断安到某个说话人头上。
4. 严格区分三类信息：转录中明确陈述的事实、说话人的主观观点、以及尚待确认的疑问。事实要写清楚数据与条件（数字、时间、版本、比例等）。
5. 若转录中出现韩语、日语或混合语言片段，先给出中文意译，再在括号内保留关键词原文。
6. 禁止编造转录中不存在的内容；信息不足时写「转录中未提及」，不要用常识补全。
7. 输出纯 Markdown，不要用代码块包裹整篇总结，不要输出「以下是总结」这类寒暄。

固定输出结构：
# {一句话标题：说清主题与结论倾向}
> 时长 HH:MM:SS ｜ 说话人 N 位 ｜ 检测语言 xx

## 一、内容概览
3~6 句话说明：这是什么内容、讨论了什么、结论是什么。让没看过原文的人 1 分钟能抓住重点。

## 二、章节脉络
按时间顺序分 3~8 个章节，每章一个小标题，格式：
### [HH:MM:SS - HH:MM:SS] 章节标题
- 该章节讲了什么（2~4 条，带说话人）

## 三、关键观点与事实
按主题（不是按时间）归纳，每条格式：
- **观点/结论**：具体内容 ｜ 提出者：Speaker X ｜ 依据：数据/例子（如有）
把有分歧的地方单独列出：`分歧点：A 认为…，B 认为…`。

## 四、说话人观察
每位说话人一段：主要立场、表达特征、关注点。

## 五、待办与未解决问题
会议/对话中明确提到的行动项（谁、做什么、什么时候）与悬而未决的问题；没有就写「无」。

## 六、术语与专有名词
表格形式：| 原文 | 中文解释 | 出现时间 |"""

SINGLE_SPEAKER_SYSTEM_PROMPT = """你是一位资深的多语言内容分析师，擅长把口语化的音视频转录整理成可直接阅读的中文结构化总结。

注意：本次转录**没有做说话人分离**，所有内容都属于同一个讲述者，请按「单人口播 / 单人讲解」处理。

硬性规则（必须全部遵守）：
1. 全部使用简体中文输出。人名、公司名、产品名、技术术语保留原文（英文/韩文/日文），首次出现时用括号给出中文解释，例如：Retrieval-Augmented Generation（检索增强生成）。
2. 时间戳必须保留，格式为 HH:MM:SS，只在「章节脉络」和「关键观点」中使用，且只能使用转录中真实出现过的时间点，严禁编造或估算。
3. 不要出现 Speaker 1 / 说话人 1 这类标签，也不要猜测讲述者身份；需要指代时用「讲述者」。
4. 严格区分三类信息：转录中明确陈述的事实、讲述者的主观观点、以及尚待确认的疑问。事实要写清楚数据与条件（数字、时间、版本、比例等）。
5. 若转录中出现韩语、日语或混合语言片段，先给出中文意译，再在括号内保留关键词原文。
6. 禁止编造转录中不存在的内容；信息不足时写「转录中未提及」，不要用常识补全。
7. 输出纯 Markdown，不要用代码块包裹整篇总结，不要输出「以下是总结」这类寒暄。

固定输出结构：
# {一句话标题：说清主题与结论倾向}
> 时长 HH:MM:SS ｜ 未做说话人分离 ｜ 检测语言 xx

## 一、内容概览
3~6 句话说明：这是什么内容、讨论了什么、结论是什么。让没看过原文的人 1 分钟能抓住重点。

## 二、章节脉络
按时间顺序分 3~8 个章节，每章一个小标题，格式：
### [HH:MM:SS - HH:MM:SS] 章节标题
- 该章节讲了什么（2~4 条）

## 三、关键观点与事实
按主题（不是按时间）归纳，每条格式：
- **观点/结论**：具体内容 ｜ 依据：数据/例子（如有）
把前后观点变化或存在分歧的地方单独列出。

## 四、待办与未解决问题
明确提到的行动项（做什么、什么时候）与悬而未决的问题；没有就写「无」。

## 五、术语与专有名词
表格形式：| 原文 | 中文解释 | 出现时间 |"""

MULTI_UNLABELED_SYSTEM_PROMPT = """你是一位资深的多语言内容分析师，擅长把口语化的音视频转录整理成可直接阅读的中文结构化总结。

注意：这是**多人对话**（访谈 / 新闻 / 节目），但本次转录**没有做说话人分离**，文本里没有可靠的说话人标签。请按「多人在交谈」处理：不要假设只有一个讲述者，也不要编造 Speaker 1 / 说话人 1 这类标签；需要指代时用角色泛称（主持人、嘉宾、旁白、介绍人等），无法判断时用「有人提到」。

硬性规则（必须全部遵守）：
1. 全部使用简体中文输出。人名、公司名、产品名、技术术语保留原文（英文/韩文/日文），首次出现时用括号给出中文解释，例如：Retrieval-Augmented Generation（检索增强生成）。
2. 时间戳必须保留，格式为 HH:MM:SS，只在「章节脉络」和「关键观点」中使用，且只能使用转录中真实出现过的时间点，严禁编造或估算。
3. 不要输出「说话人观察」章节：没有分离结果，任何关于说话人身份的归纳都只能是猜测。
4. 严格区分三类信息：转录中明确陈述的事实、说话人的主观观点、以及尚待确认的疑问。事实要写清楚数据与条件（数字、时间、版本、比例等）。
5. 若转录中出现韩语、日语或混合语言片段，先给出中文意译，再在括号内保留关键词原文。
6. 禁止编造转录中不存在的内容；信息不足时写「转录中未提及」，不要用常识补全。
7. 输出纯 Markdown，不要用代码块包裹整篇总结，不要输出「以下是总结」这类寒暄。

固定输出结构：
# {一句话标题：说清主题与结论倾向}
> 时长 HH:MM:SS ｜ 多人对话（未分离说话人） ｜ 检测语言 xx

## 一、内容概览
3~6 句话说明：这是什么内容、讨论了什么、结论是什么。让没看过原文的人 1 分钟能抓住重点。

## 二、章节脉络
按时间顺序分 3~8 个章节，每章一个小标题，格式：
### [HH:MM:SS - HH:MM:SS] 章节标题
- 该章节讲了什么（2~4 条，能判断角色时用角色泛称）

## 三、关键观点与事实
按主题（不是按时间）归纳，每条格式：
- **观点/结论**：具体内容 ｜ 来源：主持人/嘉宾（可判断时）｜ 依据：数据/例子（如有）
把有分歧的地方单独列出：`分歧点：一方认为…，另一方认为…`。

## 四、待办与未解决问题
明确提到的行动项（做什么、什么时候）与悬而未决的问题；没有就写「无」。

## 五、术语与专有名词
表格形式：| 原文 | 中文解释 | 出现时间 |"""

LOCAL_SYSTEM_PROMPT = """你是音视频转录的片段压缩器。你的任务是把一小段带时间戳的转录压缩为结构化 JSON，供后续全局汇总使用。

要求：
1. 只压缩、不扩写，不引入转录中不存在的信息。
2. 时间戳只能从给定文本里挑选真实出现过的时间点，格式 HH:MM:SS，不得编造。
3. 摘要与要点使用简体中文；专有名词保留原文，可加简短中文注释。
4. 严格输出 JSON 对象，不要输出 Markdown、不要输出解释性文字。

JSON schema：
{
  "title": "本片段主题，不超过 20 字",
  "abstract": "本片段 80~150 字的中文概括",
  "points": [
    {"timestamp": "HH:MM:SS", "speaker": "Speaker X", "point": "一条具体观点或事实，不超过 60 字"}
  ],
  "terms": [{"term": "原文术语", "translation": "中文解释"}],
  "open_questions": ["本片段中未解决的问题，没有则空数组"]
}"""

LOCAL_USER_TEMPLATE = """片段信息：第 %INDEX% 段，共 %TOTAL% 段，时间范围 %START% ~ %END%。
总时长 %DURATION%，检测语言 %LANGUAGE%，说话人：%SPEAKERS%。

转录文本（每行格式为 [HH:MM:SS] 说话人: 内容）：
%TRANSCRIPT%

请按 schema 输出 JSON。"""

REDUCE_SYSTEM_PROMPT = """你是中文内容编辑。用户会给你一批「片段摘要」，请把它们压缩成更紧凑的分析笔记。

要求：
1. 合并重复信息，保留所有时间戳与说话人归属，时间戳格式 HH:MM:SS。
2. 输出 Markdown 无序列表（每条以 `- [HH:MM:SS] ` 开头，若该条涉及多位说话人则用 `Speaker X/Y`）。
3. 不新增任何原文没有的信息，不做最终总结，只做压缩。
4. 长度控制在输入的三分之一以内。"""

REDUCE_USER_TEMPLATE = """以下是需要压缩的片段摘要：
%SUMMARIES%

请输出压缩后的 Markdown 列表。"""

FINAL_USER_TEMPLATE = """请基于下面的「片段摘要索引」写出最终总结。

基础信息：
- 内容总时长：%DURATION%
- 说话人：%SPEAKERS%
- 检测语言：%LANGUAGE%
- 片段数：%BLOCKS%

片段摘要索引（按时间顺序，时间戳均为原文真实时间）：
%DIGEST%

请严格按固定输出结构输出中文 Markdown。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class TranscriptBlock:
    """一个待局部摘要的转录块。起始/结束时间由我们自己记录，保证时间戳真实。"""

    index: int
    start: float
    end: float
    lines: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def chars(self) -> int:
        return sum(len(line) for line in self.lines)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "start_ts": format_timestamp(self.start),
            "end_ts": format_timestamp(self.end),
            "chars": self.chars,
        }


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
_client: Any = None


def get_client() -> Any:
    """初始化 DeepSeek 客户端（OpenAI SDK 兼容模式）。

    注意：base_url 用 https://api.deepseek.com（不带 /v1 也可以，
    SDK 会自动拼 /chat/completions；官方同时也支持 https://api.deepseek.com/v1）。
    """
    global _client
    if _client is not None:
        return _client

    from openai import OpenAI  # noqa: PLC0415 延迟导入，模块可在无 openai 环境下被导入/单测

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SummarizationError("缺少 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek")

    _client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=1,  # 重试统一由 _chat() 控制
    )
    logger.info("DeepSeek 客户端就绪（base_url=%s, model=%s）", DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    return _client


def _is_reasoner(model: str) -> bool:
    return "reasoner" in model or model.endswith("-r1")


def _build_params(messages: list[dict], *, model: str, max_tokens: int, json_mode: bool) -> dict:
    """按模型裁剪参数：deepseek-reasoner 不支持 temperature / response_format。"""
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if not _is_reasoner(model):
        params["temperature"] = SUMMARY_TEMPERATURE
        if json_mode:
            params["response_format"] = {"type": "json_object"}
    return params


def _explain_api_error(exc: Exception) -> str:
    """把 OpenAI/DeepSeek 的 HTTP 错误翻译成可操作的中文提示。"""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        if status == 401:
            return "API Key 无效或已失效，请检查 .env 里的 DEEPSEEK_API_KEY"
        if status == 402:
            return (
                "DeepSeek 账户余额不足，请先充值 https://platform.deepseek.com/top_up 后重试"
                "（转录结果已保存，无需重新转录）"
            )
        if status == 429:
            return "请求过于频繁（429），请稍后重试或调小并发"
        if status >= 500:
            return f"DeepSeek 服务端错误（{status}），请稍后重试"
    return str(exc)


def _retryable(exc: Exception) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError  # noqa: PLC0415

        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    text = str(exc).lower()
    return any(key in text for key in ("timeout", "timed out", "rate limit", "502", "503", "504"))


def _chat(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = SUMMARY_MAX_TOKENS,
    json_mode: bool = False,
    purpose: str = "",
) -> tuple[str, dict]:
    """带重试的单轮对话，返回 (content, usage)。"""
    client = get_client()
    model = model or DEEPSEEK_MODEL
    params = _build_params(messages, model=model, max_tokens=max_tokens, json_mode=json_mode)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        try:
            response = client.chat.completions.create(**params)
            content = (response.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }
            finish = getattr(response.choices[0], "finish_reason", None)
            logger.info(
                "[%s] 调用成功 %.1fs（prompt=%d, completion=%d, finish=%s）",
                purpose or "chat", time.time() - started,
                usage["prompt_tokens"], usage["completion_tokens"], finish,
            )
            if finish == "length":
                logger.warning("[%s] 输出被 max_tokens 截断，可调大 SUMMARY_MAX_TOKENS", purpose)
            if not content:
                raise SummarizationError("DeepSeek 返回空内容")
            return content, usage
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= MAX_RETRIES or not _retryable(exc):
                break
            backoff = min(2 ** attempt, 10)
            logger.warning("[%s] 第 %d 次调用失败：%s，%ds 后重试", purpose or "chat", attempt, exc, backoff)
            time.sleep(backoff)

    raise SummarizationError(f"DeepSeek 调用失败（{purpose}）：{_explain_api_error(last_exc)}") from last_exc


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown|md)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(content: str) -> dict | None:
    candidate = _strip_code_fence(content)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """12.4 -> 00:00:12；3725 -> 01:02:05。"""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_segment(seg: dict) -> str:
    return f"[{format_timestamp(seg.get('start', 0))}] {seg.get('speaker') or 'Speaker ?'}: {str(seg.get('text', '')).strip()}"


def format_transcript(segments: Sequence[dict]) -> str:
    """整段转录 -> 带时间戳与说话人的纯文本（调试/存档用）。"""
    return "\n".join(format_segment(seg) for seg in segments if str(seg.get("text", "")).strip())


def _collect_speakers(segments: Sequence[dict]) -> list[str]:
    seen: list[str] = []
    for seg in segments:
        name = str(seg.get("speaker") or "UNKNOWN")
        if name not in seen:
            seen.append(name)
    return seen


def build_blocks(segments: Sequence[dict], max_chars: int = SUMMARY_BLOCK_CHARS) -> list[TranscriptBlock]:
    """按字数把 segments 切成若干个 block；切点只落在 segment 边界，不会切断句子。"""
    max_chars = max(500, max_chars)
    blocks: list[TranscriptBlock] = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        line = format_segment(seg)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        current = blocks[-1] if blocks else None
        if current is None or (current.chars + len(line) > max_chars and current.lines):
            blocks.append(
                TranscriptBlock(index=len(blocks), start=start, end=end, lines=[line], segments=[seg])
            )
        else:
            current.lines.append(line)
            current.segments.append(seg)
            current.end = max(current.end, end)

    # 尾块太短则并入前一块，避免产生「只有一句话」的碎片摘要
    if len(blocks) >= 2 and blocks[-1].chars < max_chars * 0.1:
        tail = blocks.pop()
        blocks[-1].lines.extend(tail.lines)
        blocks[-1].segments.extend(tail.segments)
        blocks[-1].end = max(blocks[-1].end, tail.end)

    for i, block in enumerate(blocks):
        block.index = i
    logger.info("转录切分为 %d 个摘要块（每块上限 %d 字符）", len(blocks), max_chars)
    return blocks


# ---------------------------------------------------------------------------
# 第一阶段：局部摘要
# ---------------------------------------------------------------------------
def _sanitize_points(points: Any, valid_ts: set[str]) -> list[dict]:
    """丢弃时间戳不在本片段范围内的要点，防止模型编时间。"""
    cleaned: list[dict] = []
    if not isinstance(points, list):
        return cleaned
    for item in points:
        if not isinstance(item, dict):
            continue
        point = str(item.get("point", "")).strip()
        if not point:
            continue
        ts = str(item.get("timestamp", "")).strip()
        if ts and ts not in valid_ts:
            logger.debug("丢弃非本片段时间戳的要点：%s %s", ts, point[:30])
            ts = ""
        cleaned.append(
            {
                "timestamp": ts,
                "speaker": str(item.get("speaker", "")).strip() or "Speaker ?",
                "point": point,
            }
        )
    return cleaned


def _valid_timestamps(block: TranscriptBlock) -> set[str]:
    """本片段内真实出现过的时间戳集合。"""
    stamps = {format_timestamp(seg.get("start", 0)) for seg in block.segments}
    stamps.add(format_timestamp(block.start))
    stamps.add(format_timestamp(block.end))
    return stamps


def _speaker_label(speakers: Sequence[str], speaker_mode: str) -> str:
    if speaker_mode == "labeled":
        return "、".join(speakers) or "未知"
    if speaker_mode == "multi_unlabeled":
        return "多人对话（未做说话人分离，无说话人标签）"
    return "未做说话人分离（单人口播）"


def _system_prompt(speaker_mode: str) -> str:
    if speaker_mode == "labeled":
        return SYSTEM_PROMPT
    if speaker_mode == "multi_unlabeled":
        return MULTI_UNLABELED_SYSTEM_PROMPT
    return SINGLE_SPEAKER_SYSTEM_PROMPT


def _speaker_breakdown(speaker_stats: Sequence[dict]) -> str:
    """把 speaker_stats 压成一行占比，例如 `Speaker 1 62% / Speaker 2 33% / Speaker 3 5%`。

    只占 5% 的说话人不该在「说话人观察」里占一整段，读者需要看到权重。
    """
    total = sum(float(item.get("seconds") or 0) for item in speaker_stats)
    if total <= 0:
        return ""
    return " / ".join(
        f"{item.get('speaker')} {float(item.get('seconds') or 0) / total * 100:.0f}%"
        for item in speaker_stats
    )


def summarize_block(
    block: TranscriptBlock,
    total: int,
    duration: float,
    language: str,
    speakers: Sequence[str],
    speaker_mode: str = "labeled",
) -> dict:
    """对一个 block 做局部摘要，失败时返回兜底结构（不阻断整体流程）。"""
    user_prompt = (
        LOCAL_USER_TEMPLATE.replace("%INDEX%", str(block.index + 1))
        .replace("%TOTAL%", str(total))
        .replace("%START%", format_timestamp(block.start))
        .replace("%END%", format_timestamp(block.end))
        .replace("%DURATION%", format_timestamp(duration))
        .replace("%LANGUAGE%", language)
        .replace("%SPEAKERS%", _speaker_label(speakers, speaker_mode))
        .replace("%TRANSCRIPT%", block.text)
    )
    messages = [
        {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, usage = _chat(messages, json_mode=True, purpose=f"block-{block.index + 1}/{total}")
    parsed = _parse_json(content)
    if parsed is None:
        logger.warning("block %d 返回非 JSON，改用纯文本兜底", block.index + 1)
        parsed = {"title": f"片段 {block.index + 1}", "abstract": _strip_code_fence(content), "points": []}

    summary = {
        "index": block.index,
        "start": round(block.start, 3),
        "end": round(block.end, 3),
        "start_ts": format_timestamp(block.start),
        "end_ts": format_timestamp(block.end),
        "title": str(parsed.get("title", f"片段 {block.index + 1}")).strip(),
        "abstract": str(parsed.get("abstract", "")).strip(),
        "points": _sanitize_points(parsed.get("points"), _valid_timestamps(block)),
        "terms": [
            {"term": str(t.get("term", "")).strip(), "translation": str(t.get("translation", "")).strip()}
            for t in (parsed.get("terms") or [])
            if isinstance(t, dict) and str(t.get("term", "")).strip()
        ],
        "open_questions": [str(q).strip() for q in (parsed.get("open_questions") or []) if str(q).strip()],
        "chars": block.chars,
        "usage": usage,
    }
    logger.info("片段 %d 摘要完成：%s（%d 条要点）", block.index + 1, summary["title"], len(summary["points"]))
    return summary


# ---------------------------------------------------------------------------
# 第二阶段：聚合
# ---------------------------------------------------------------------------
def build_digest(summaries: Sequence[dict]) -> str:
    """把局部摘要拼成给聚合阶段用的索引文本。"""
    chunks: list[str] = []
    for item in summaries:
        lines = [f"### 片段 {item['index'] + 1} ｜ {item['start_ts']} - {item['end_ts']} ｜ {item['title']}"]
        if item["abstract"]:
            lines.append(f"摘要：{item['abstract']}")
        if item["points"]:
            lines.append("要点：")
            for point in item["points"]:
                stamp = f"[{point['timestamp']}] " if point["timestamp"] else ""
                lines.append(f"- {stamp}{point['speaker']}：{point['point']}")
        if item["terms"]:
            lines.append("术语：" + "；".join(f"{t['term']}（{t['translation']}）" for t in item["terms"] if t["translation"]))
        if item["open_questions"]:
            lines.append("待解决：" + "；".join(item["open_questions"]))
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _split_text(text: str, max_chars: int) -> list[str]:
    """按行切分超长文本，尽量在段落边界断开。"""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and buffer:
            parts.append("".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += len(line)
    if buffer:
        parts.append("".join(buffer))
    return parts


def _condense(text: str, max_chars: int = AGG_MAX_CHARS, depth: int = 0) -> str:
    """输入超过上下文预算时，分组压缩后再合并（最多 3 层，防止无限递归）。"""
    if len(text) <= max_chars or depth >= 3:
        return text

    parts = _split_text(text, max_chars)
    logger.info("聚合输入 %d 字符，分 %d 组压缩（depth=%d）", len(text), len(parts), depth)
    compressed: list[str] = []
    for i, part in enumerate(parts):
        user_prompt = REDUCE_USER_TEMPLATE.replace("%SUMMARIES%", part)
        content, _usage = _chat(
            [{"role": "system", "content": REDUCE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=SUMMARY_MAX_TOKENS,
            purpose=f"reduce-{i + 1}/{len(parts)}",
        )
        compressed.append(_strip_code_fence(content))

    merged = "\n\n".join(compressed)
    if len(merged) > max_chars:
        return _condense(merged, max_chars=max_chars, depth=depth + 1)
    return merged


def global_summary(
    summaries: Sequence[dict],
    *,
    duration: float,
    speakers: Sequence[str],
    language: str,
    speaker_mode: str = "labeled",
    speaker_stats: Sequence[dict] = (),
) -> tuple[str, dict]:
    """第二阶段：局部摘要 -> 全局中文 Markdown 总结。"""
    digest = build_digest(summaries)
    digest = _condense(digest)

    user_prompt = (
        FINAL_USER_TEMPLATE.replace("%DURATION%", format_timestamp(duration))
        .replace("%SPEAKERS%", _speaker_label(speakers, speaker_mode))
        .replace("%LANGUAGE%", language)
        .replace("%BLOCKS%", str(len(summaries)))
        .replace("%DIGEST%", digest)
    )
    messages = [
        {"role": "system", "content": _system_prompt(speaker_mode)},
        {"role": "user", "content": user_prompt},
    ]
    content, usage = _chat(
        messages,
        max_tokens=FINAL_MAX_TOKENS,
        purpose="final-summary",
    )
    markdown = _strip_code_fence(content)
    markdown = _postprocess_markdown(
        markdown,
        duration=duration,
        speakers=speakers,
        language=language,
        speaker_mode=speaker_mode,
        speaker_stats=speaker_stats,
    )
    return markdown, usage


def _postprocess_markdown(
    markdown: str,
    *,
    duration: float,
    speakers: Sequence[str],
    language: str,
    speaker_mode: str = "labeled",
    speaker_stats: Sequence[dict] = (),
) -> str:
    """补齐/修正头部元信息，保证输出一定有「一级标题 + 元信息行」。"""
    if speaker_mode == "labeled":
        breakdown = _speaker_breakdown(speaker_stats)
        speaker_part = (
            f"说话人 {len(speakers)} 位（{breakdown}）" if breakdown else f"说话人 {len(speakers)} 位"
        )
        meta = f"> 时长 {format_timestamp(duration)} ｜ {speaker_part} ｜ 检测语言 {language}"
    elif speaker_mode == "multi_unlabeled":
        meta = f"> 时长 {format_timestamp(duration)} ｜ 多人对话（未分离说话人） ｜ 检测语言 {language}"
    else:
        meta = f"> 时长 {format_timestamp(duration)} ｜ 未做说话人分离 ｜ 检测语言 {language}"
    lines = markdown.strip().splitlines()

    # 只认一级标题；模型直接以 "## 一、..." 开头时补一个 H1，避免元信息插进章节中间
    if not lines or not re.match(r"^#\s+\S", lines[0].lstrip()):
        lines = ["# 内容总结", "", *lines]

    # 模型自己写的元信息行信息量不足（只有「说话人 4 位」），用我们算好的版本替换
    for i, line in enumerate(lines[:6]):
        if line.strip().startswith("> 时长"):
            lines[i] = meta
            return "\n".join(lines).strip() + "\n"

    lines.insert(1, "")
    lines.insert(2, meta)
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# 对外主入口
# ---------------------------------------------------------------------------
def normalize_payload(payload: Any) -> dict:
    """校验并规范化转录 JSON（接口导入 / CLI / 内部调用共用同一套规则）。

    容忍两种输入：完整 dict（含 segments/speakers/duration_sec），或直接一个 segment 列表。
    :raises SummarizationError: 结构不合法或没有可用文本
    """
    if isinstance(payload, list):
        payload = {"segments": payload}
    if not isinstance(payload, dict):
        raise SummarizationError(f"不支持的转录输入类型: {type(payload)!r}")

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise SummarizationError("转录 JSON 里没有 segments 数组，或 segments 为空")

    cleaned: list[dict] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        cleaned.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": str(item.get("speaker") or "UNKNOWN"),
                "text": text,
            }
        )

    if not cleaned:
        raise SummarizationError("segments 里没有任何可用的文本（text 全为空或时间戳非法）")
    cleaned.sort(key=lambda s: s["start"])

    normalized = dict(payload)
    normalized["segments"] = cleaned
    normalized["num_segments"] = len(cleaned)
    if not isinstance(normalized.get("speakers"), list) or not normalized["speakers"]:
        normalized["speakers"] = _collect_speakers(cleaned)
    if not normalized.get("duration_sec"):
        normalized["duration_sec"] = cleaned[-1]["end"]
    return normalized


def _extract_segments(payload: Any) -> tuple[list[dict], dict]:
    """规范化输入并拆出 (segments, meta)。"""
    payload = normalize_payload(payload)
    segments = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": seg["speaker"],
            "text": seg["text"],
        }
        for seg in payload["segments"]
    ]
    speakers = payload.get("speakers") or []
    # 只有「确实分离出了具名说话人」才按多人对话总结；
    # 否则模型会被要求区分根本不存在的 Speaker，只能编造。
    # 导入的 JSON 可能没有 diarized 字段，此时以 speakers 是否有具名标签为准。
    named_speakers = any(str(name).strip().upper() not in {"", "UNKNOWN"} for name in speakers)
    diarized = named_speakers and payload.get("diarized") is not False

    # 三态：有标签 / 多人但没分离出来 / 单人口播。
    # 分离失败时如果一律按「单人口播」总结，会把双人访谈写成「讲述者认为…」。
    diarization = payload.get("diarization")
    diar_requested = bool(diarization.get("requested")) if isinstance(diarization, dict) else False
    if diarized:
        speaker_mode = "labeled"
    elif diar_requested:
        speaker_mode = "multi_unlabeled"
    else:
        speaker_mode = "single"

    raw_stats = payload.get("speaker_stats")
    speaker_stats = [
        item for item in raw_stats
        if isinstance(item, dict) and item.get("speaker")
    ] if isinstance(raw_stats, list) else []

    meta = {
        "duration": float(payload.get("duration_sec") or 0.0),
        "language": str(payload.get("language") or "auto"),
        "speakers": speakers,
        "speaker_stats": speaker_stats,
        "diarized": diarized,
        "speaker_mode": speaker_mode,
        "diarization": diarization if isinstance(diarization, dict) else None,
    }
    if meta["duration"] <= 0 and segments:
        meta["duration"] = segments[-1]["end"]
    return segments, meta


def summarize(
    payload: Any,
    *,
    block_chars: int | None = None,
    save_to: str | Path | None = None,
) -> dict:
    """转录结果 -> 结构化中文 Markdown 总结。

    :param payload: 第三步返回的 dict（含 segments/speakers/duration_sec），或纯 segment 列表
    :param block_chars: 覆盖 SUMMARY_BLOCK_CHARS
    :param save_to: 若提供路径，则把 Markdown 落盘
    :return: {"markdown", "blocks", "usage", "meta"}
    """
    started = time.time()
    segments, meta = _extract_segments(payload)
    if not segments:
        raise SummarizationError("转录结果为空，无法生成总结")

    blocks = build_blocks(segments, max_chars=block_chars or SUMMARY_BLOCK_CHARS)
    mode_label = {
        "labeled": f"说话人 {meta['speakers']}",
        "multi_unlabeled": "多人对话但未做说话人分离（多人口播 prompt）",
        "single": "未做说话人分离（单人口播 prompt）",
    }[meta["speaker_mode"]]
    logger.info(
        "开始总结：%d 段 / %d 块 / 时长 %s / %s",
        len(segments), len(blocks), format_timestamp(meta["duration"]), mode_label,
    )

    summaries: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for block in blocks:
        summary = summarize_block(
            block,
            total=len(blocks),
            duration=meta["duration"],
            language=meta["language"],
            speakers=meta["speakers"],
            speaker_mode=meta["speaker_mode"],
        )
        summaries.append(summary)
        for key in usage_total:
            usage_total[key] += summary["usage"].get(key, 0)

    markdown, final_usage = global_summary(
        summaries,
        duration=meta["duration"],
        speakers=meta["speakers"],
        language=meta["language"],
        speaker_mode=meta["speaker_mode"],
        speaker_stats=meta["speaker_stats"],
    )
    for key in usage_total:
        usage_total[key] += final_usage.get(key, 0)

    result = {
        "markdown": markdown,
        "blocks": list(summaries),
        "usage": usage_total,
        "meta": {
            "model": DEEPSEEK_MODEL,
            "num_segments": len(segments),
            "num_blocks": len(blocks),
            "duration_sec": round(meta["duration"], 3),
            "duration_ts": format_timestamp(meta["duration"]),
            "language": meta["language"],
            "speakers": meta["speakers"],
            "speaker_stats": meta["speaker_stats"],
            "diarized": meta["diarized"],
            "speaker_mode": meta["speaker_mode"],
            "diarization": meta["diarization"],
            "input_chars": sum(b.chars for b in blocks),
            "elapsed_sec": round(time.time() - started, 1),
        },
    }
    for item in result["blocks"]:
        item.pop("usage", None)

    if save_to:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        logger.info("总结已写入 %s", path)

    logger.info(
        "总结完成：%d 块，tokens=%d，耗时 %.1fs",
        len(blocks), usage_total["total_tokens"], result["meta"]["elapsed_sec"],
    )
    return result


# ---------------------------------------------------------------------------
# CLI：python -m app.services.summarizer transcript.json --out summary.md
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek 两阶段总结")
    parser.add_argument("transcript", help="第三步输出的 JSON（或 segments 数组）")
    parser.add_argument("--out", default=None, help="Markdown 输出路径")
    parser.add_argument("--block-chars", type=int, default=None, help="单块字符上限")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    payload = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    result = summarize(payload, block_chars=args.block_chars, save_to=args.out)

    print(result["markdown"])
    print("\n---")
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    _main()
