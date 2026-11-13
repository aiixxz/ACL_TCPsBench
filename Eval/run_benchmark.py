#!/usr/bin/env python3
import argparse
import asyncio
import base64
from io import BytesIO
import json
import mimetypes
import os
import random
import re
import sys
import warnings
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp
from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_SCRIPTS = PROJECT_ROOT / "paper" / "scripts"
if str(PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS))

# Reuse the exact seven-dimensional rubric, evidence schema, parser, and hashes
# used by the standalone open-ended evaluator.
import judge_openended_7d as seven_dim_judge

try:
    from PIL import Image
except ImportError:
    Image = None


SINGLE_PROMPT = """你是一位中国绘画与艺术史专家，擅长解答中国画相关的单选题。
请根据提供的图片回答以下单选题，仅需输出选项字母（如A、B、C、D），不要输出其他内容。

【严格格式要求】
- 只输出一个大写英文字母
- 不得包含任何分析、推理过程或解释
- 不得输出句子、标点或任何其他字符

以下是几个示例，展示正确的输出格式：

【示例1】
问题：此画的作者最可能是哪位画家？
A. 齐白石  B. 张大千  C. 徐悲鸿  D. 吴昌硕
输出：B

【示例2】
问题：该作品采用的主要皴法是？
A. 披麻皴  B. 斧劈皴  C. 雨点皴  D. 折带皴
输出：A

【示例3】
问题：这幅画属于哪种绘画类型？
A. 山水画  B. 花鸟画  C. 人物画  D. 界画
输出：C

【问题】
{question}

【选项】
{options_text}"""

FILL_PROMPT_T1 = """TASK 1:
你是一个中国绘画图像理解模型，专注于识别图像。请你根据图片内容，输出最简洁的答案，注意不要带书名号"""

FILL_PROMPT_T2 = """TASK 2:
你是一个中国绘画图像理解模型，只输出简短物象名称,尽可能与国画中的术语对齐"""

FILL_PROMPT_T3 = """TASK 3:
你必须只从下列中国画构图方式中选择一个作为最终回答，不得输出列表外内容：
- 两线构图
- 三线构图
- 三角形构图
- 之字形构图
- 三分法构图
- 折枝式构图
- 满构图
- 甲字形构图
- 由字形构图
- 则字形构图
- 须字形构图
- 边角构图
- 水平线构图
- 垂直线构图
- 中心构图
- 三叠两段式构图

注意要求：
1. 你只输出一个构图名称，不附带解释。
2. 不允许输出列表之外的构图。
3. 不允许描述图片标题或路径。

【输出规则】
- 从下方列表中选择恰好一个构图名称输出
- 直接输出名称，不加任何前缀、后缀或解释
- 不允许输出列表外的构图名称

【输出示例】
正确：折枝式构图
正确：三分法构图
错误（禁止）：根据图中元素的排列，我认为应该是折枝式构图
错误（禁止）：这幅画采用了折枝式构图的方式来组织画面
"""

FILL_PROMPT_T4 = """TASK 4:
你是一名中国传统绘画（国画）领域的专家，精通山水、人物、花鸟的笔法、皴法、描法与设色技法。
请你严格使用中国画专业术语回答问题，优先从下列术语中选择最贴切的名称作答。
如需输出多个术语，请以顿号分隔，避免解释性语句，不要输出无关描述，仅给出术语本身。

可用术语包括：
披麻皴、解索皴、荷叶皴、牛毛皴、斧劈皴、雨点皴、拖泥带水皴、乱柴皴、弹涡皴、卷云皴、米点皴、折带皴；
高古游丝描、琴弦描、铁线描、混描、曹衣描、钉头鼠尾描、镢头钉描、蚂蝗描、柳叶描、橄榄描、枣核描、折芦描、竹叶描、战笔水纹描、减笔描、枯柴描、蚯蚓描、行云流水描、泼墨；
双勾法（写意）、勾染法（写意）、勾填法（写意）、没骨法（写意）、破墨破色法（写意）、兼工带写法、泼墨泼彩法（写意）；
白描法（工笔）、工笔重彩法、工笔淡彩法、浓淡相间法（工笔）、没骨法（工笔）、工笔写意兼工带写。

【输出规则】
- 从术语列表中选择最贴切的术语，多个术语以顿号分隔
- 只输出术语本身，不得包含解释性语句或完整句子

【输出示例】
正确：披麻皴、工笔淡彩法
正确：铁线描、没骨法（工笔）
错误（禁止）：这幅山水画采用了披麻皴，同时结合工笔淡彩法上色
错误（禁止）：笔法上使用了铁线描来勾勒轮廓

如果图像内容无法明确对应具体术语，请给出最接近的术语，不要回答“无法判断”。"""

FILL_PROMPT_T5 = """TASK 5:
你是一名中国传统绘画（国画）领域的专家，熟悉中国画的审美范畴与意境评判体系。
请你依据题目与图像内容，从指定的意境词汇中选择最贴切的意境进行回答。

你的回答必须严格来自以下意境词汇集合，不得引入新的描述词，不得进行解释或评论。
如意境较为复合，可选择多个最核心的意境词，用顿号分隔即可。

可用意境词包括：
气韵神妙、高古、苍润、沉雄、冲和、淡逸、朴拙、超脱、奇辟、纵横、淋漓、荒寒、清旷、性灵、圆浑、幽邃、明净、健拔、简洁、精谨、隽爽、空灵、韶秀。

【输出规则】
- 从意境词汇列表中选择1-2个最贴切的词，以顿号分隔
- 不得输出列表外的词汇，不得附加任何解释

【输出示例】
正确：荒寒、空灵
正确：淡逸
错误（禁止）：这幅画给人一种荒寒和空灵的感觉，整体意境深远
错误（禁止）：意境：荒寒

若图像意境不够明确，请选择最接近整体风格的意境词，不要回答“无法判断”或类似表述。"""

JUDGE_PROMPT = """你是一位中国绘画与艺术史专家，擅长解答中国画的判断题。
图中内容描述：{question}。请回答：是 或 否。禁止生成除此之外的任何回答。

【输出示例】
正确示例：是
正确示例：否
错误示例（禁止）：是的，图中确实描绘了……
错误示例（禁止）：<thinking>从图中来看……</thinking>否
错误示例（禁止）：答案是否，因为……

"""

MULTI_PROMPT = """你是一位精通国画知识的答题者，请识别图片中的中国画画作信息并回答下面这道多选题。
多选题的答案可能是2、3、4个选项，不可能只有一个选项。你必须严格保证你输出的答案是2或3或4个
输出格式：请严格保证你输出的答案仅包含按顺序的英文大写字母，不包含解释、标点等其他任何内容（输出示例：BCD）

【严格格式要求】
- 只输出连续的大写英文字母，不含任何标点、空格或其他字符
- 答案数量必须在2到4个之间，不能只选1个
- 字母按字母表顺序排列
- 禁止输出分析过程、解释或其他任何内容

【输出示例】
正确示例1：BC
正确示例2：ABD
正确示例3：ABCD
错误示例（禁止）：A、B、D
错误示例（禁止）：答案是ABD，因为……
错误示例（禁止）：(ABD)
错误示例（禁止）：B（只有一个选项，不符合多选要求）


{question_text}"""

SHORT_ANSWER_SYSTEM_PROMPT = """你是一位中国绘画与艺术史专家，擅长解答中国画相关的简答题。请根据图中画作信息，用中文回答对应问题。
【核心约束】
1. 你的思考过程必须完全保留在内部，绝对不能出现在输出中
2. 不得输出<thinking>、<analysis>等任何标签或其内容
3. 不得以"我认为"、"根据图像"、"答案是"等前缀开头
4. 直接输出题目要求的答案格式，第一个字符就是答案本身"""

OBJECTIVE_SYSTEM_PROMPT = (
    "你是一位中国绘画与艺术史专家。"
    "你必须严格遵守用户给出的输出格式要求，只输出最终答案本身。"
    "不要解释，不要复述题目，不要展示推理过程，不要输出分析步骤。"
)

# Historical two-dimensional prompt retained only so old reports remain
# interpretable. The active score phase uses --judge-prompt and the exact
# seven-dimensional implementation imported from judge_openended_7d.py.
LEGACY_SHORT_ANSWER_EVAL_PROMPT = """你是一位中国画与艺术史领域的评审专家，需严格按照以下评分规则对模型回答进行评估，最终总分是两个维度得分之和（满分10分，均为整数）。

## 评分规则
### 维度 1：术语准确性（5 分）
5 分：所有国画术语（如 “折枝构图”“高古游丝描”）完全准确，与标准答案/国画专业定义无偏差；
4 分：仅 1 个次要术语表述不精准（如将 “工笔重彩” 简化为 “重彩”），核心术语准确；
3 分：1 个核心术语表述错误（如混淆 “写意” 与 “兼工带写”），或 2 个次要术语不准确；
2 分：2 个及以上核心术语错误，仅少数术语符合专业规范；
1 分：绝大多数术语错误，与国画专业表述无关。

### 维度 2：语言表达连贯程度（5 分）
5 分：语句逻辑清晰、流畅，符合简答题 “简洁说明” 的场景，无冗余/病句；
4 分：语句基本流畅，仅 1 处表述略显生硬（如术语衔接不自然），不影响理解；
3 分：存在 2 处语句不通顺或逻辑跳跃，需结合图像才能理解核心意思；
2 分：语句混乱，逻辑断裂，需反复梳理才能部分理解；
1 分：表述模糊、碎片化，无法清晰传达核心信息。

## 评分示例（Few-shot）
### 示例 1
【问题】简述这幅作品的构图技法
【参考答案】该作品采用折枝构图，以高古游丝描勾勒物象轮廓，属于典型的工笔重彩技法。
【模型回答】这幅作品用了折枝的构图方式，以高古游丝描画轮廓，是工笔重彩的风格。
【评分结果】
{{
  "score": 10,
  "analysis": {{
    "术语准确性得分": 5,
    "术语准确性分析": "所有国画术语（折枝构图、高古游丝描、工笔重彩）均准确无误，符合专业定义。",
    "语言表达连贯程度得分": 5,
    "语言表达连贯程度分析": "语句逻辑清晰、流畅，简洁说明构图技法，无冗余和病句。",
    "overall_comment": "模型回答完全符合要求，术语准确且表达流畅。"
  }}
}}

### 示例 2
【问题】说明该作品的绘画技法类型
【参考答案】此作品为写意花鸟画，运用泼墨技法表现物象的气韵。
【模型回答】这幅画是兼工带写的花鸟，用泼墨的方法画的。
【评分结果】
{{
  "score": 3,
  "analysis": {{
    "术语准确性得分": 1,
    "术语准确性分析": "核心术语“写意”被错误表述为“兼工带写”，且无其他准确术语，符合1分标准。",
    "语言表达连贯程度得分": 2,
    "语言表达连贯程度分析": "语句逻辑断裂，“兼工带写的花鸟”表述不完整，需反复梳理才能理解核心意思。",
    "overall_comment": "核心术语错误，语言表达混乱，整体回答质量差。"
  }}
}}

### 示例 3
【问题】分析这幅作品的线条技法
【参考答案】作品运用铁线描勾勒轮廓，结合兰叶描表现细节，属于工笔白描范畴。
【模型回答】这幅作品用铁线描画轮廓，细节用兰叶描表现，是白描的手法。
【评分结果】
{{
  "score": 9,
  "analysis": {{
    "术语准确性得分": 4,
    "术语准确性分析": "仅将“工笔白描”简化为“白描”，属于1个次要术语表述不精准，核心术语（铁线描、兰叶描）准确，符合4分标准。",
    "语言表达连贯程度得分": 5,
    "语言表达连贯程度分析": "语句逻辑清晰、流畅，简洁说明线条技法，无冗余和病句。",
    "overall_comment": "仅次要术语简化，核心术语准确，表达流畅，整体回答质量高。"
  }}
}}

## 正式评估
【问题】
{question}

【参考答案】
{reference}

【模型回答】
{candidate}

【输出要求】
1. 严格按照JSON格式输出，不得添加任何额外内容；
2. score为术语准确性得分+语言表达连贯程度得分（整数，0-10分）；
3. analysis需包含各维度的得分和详细分析，以及整体评价；
4. 分析内容需紧扣评分规则，结合具体术语和语句进行说明。

【输出格式】
{{
  "score": number,
  "analysis": {{
    "术语准确性得分": number,
    "术语准确性分析": string,
    "语言表达连贯程度得分": number,
    "语言表达连贯程度分析": string,
    "overall_comment": string
  }}
}}"""


@dataclass
class Sample:
    sample_id: str
    sample_type: str
    source_task: str
    question: str
    reference_answer: str
    image_path: str
    # 本地映射的文件路径（在 image_path 被转换为 data: URL 之后仍保留原始文件路径）
    image_file: str = ""
    image_paths: List[str] = field(default_factory=list)
    image_files: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str
    api_style: str


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["__line__"] = line_no
            obj["__file__"] = file_path.name
            rows.append(obj)
    return rows


def extract_qa_from_conversations(record: Dict[str, Any]) -> Tuple[str, str]:
    question = ""
    answer = ""
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("from", "")).strip().lower()
            value = item.get("value")
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value:
                continue
            if speaker.startswith("human") and not question:
                question = re.sub(r"^<image>\s*", "", value, flags=re.IGNORECASE).strip()
            elif speaker.startswith("gpt") and not answer:
                answer = value
    if not question:
        for key in ["question", "prompt", "input", "query", "instruction"]:
            val = record.get(key)
            if isinstance(val, str) and val.strip():
                question = val.strip()
                break
    if not answer:
        for key in ["answer", "label", "target", "output", "completion", "reference"]:
            val = record.get(key)
            if isinstance(val, str) and val.strip():
                answer = val.strip()
                break
    return question, answer


def normalize_sample_type(task_type: str) -> str:
    if not task_type:
        return "Fill"
    t = task_type.strip().lower()
    if t in {"single", "singlechoice", "choice"}:
        return "Single"
    if t in {"fill", "blank", "fillblank"}:
        return "Fill"
    if t in {"judge", "boolean", "truefalse"}:
        return "Judge"
    if t in {"multi", "multiple", "multiplechoice"}:
        return "Multi"
    if t in {"shortanswer", "sa", "short_answer"} or t.startswith("shortanswer"):
        return "ShortAnswer"
    return task_type


def infer_type_from_source(source_name: str, current_type: str) -> str:
    if current_type and current_type not in {"Fill", "", None}:
        return current_type
    if source_name.startswith("T7") or source_name.startswith("T8"):
        return "ShortAnswer"
    return current_type or "Fill"


def parse_samples(data_dir: Path, input_files: List[str]) -> List[Sample]:
    samples: List[Sample] = []
    for name in input_files:
        file_path = data_dir / name
        if not file_path.exists():
            log(f"跳过不存在文件: {file_path}")
            continue

        records = load_jsonl(file_path)
        for record in records:
            question, reference_answer = extract_qa_from_conversations(record)
            if not question or not reference_answer:
                continue

            image_path = ""
            image_paths: List[str] = []
            image = record.get("image")
            if isinstance(image, str):
                image_path = image.strip()
                if image_path:
                    image_paths = [image_path]
            images = record.get("images")
            if isinstance(images, list):
                image_paths = [str(x).strip() for x in images if isinstance(x, str) and str(x).strip()]
                if image_paths:
                    image_path = image_paths[0]

            sample_type = normalize_sample_type(str(record.get("task_type", "")).strip())
            sample_type = infer_type_from_source(file_path.name, sample_type)
            sample_id = f"{file_path.stem}_{record.get('__line__', 0)}"

            samples.append(
                Sample(
                    sample_id=sample_id,
                    sample_type=sample_type,
                    source_task=file_path.stem,
                    question=question,
                    reference_answer=reference_answer,
                    image_path=image_path,
                    image_file="",
                    image_paths=image_paths,
                    image_files=[],
                )
            )
    return samples


def extract_options_text(question_text: str) -> Tuple[str, str]:
    # Split into stem and options if options are embedded as A. / B. style.
    parts = re.split(r"\s(?=[A-DＡ-Ｄ][\.、:：])", question_text)
    if len(parts) <= 1:
        return question_text, ""
    stem = parts[0].strip()
    options = "\n".join(p.strip() for p in parts[1:] if p.strip())
    return stem, options


def pick_fill_prompt(source_task: str, question: str) -> str:
    if source_task.startswith("T1"):
        return FILL_PROMPT_T1
    if source_task.startswith("T2"):
        return FILL_PROMPT_T2
    if source_task.startswith("T3"):
        return FILL_PROMPT_T3
    if source_task.startswith("T4"):
        return FILL_PROMPT_T4
    if source_task.startswith("T5"):
        return FILL_PROMPT_T5

    q = question
    if "构图" in q:
        return FILL_PROMPT_T3
    if "皴" in q or "描" in q or "技法" in q:
        return FILL_PROMPT_T4
    if "意境" in q:
        return FILL_PROMPT_T5
    return FILL_PROMPT_T2


def normalize_path_string(path_str: str) -> str:
    return path_str.replace("\\", "/").strip()


def normalize_api_base(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return "https://api.zjuqx.cn/v1"
    if re.search(r"/v\d+$", url):
        return url
    return f"{url}/v1"


_IMAGE_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _image_root_cache_key(image_root: str) -> str:
    root = (image_root or "").strip()
    if not root:
        return ""
    try:
        return os.path.normcase(str(Path(root).resolve()))
    except Exception:
        return os.path.normcase(root)


def _build_image_index(image_root: str) -> Dict[str, Any]:
    cache_key = _image_root_cache_key(image_root)
    if not cache_key:
        return {"files": [], "by_name": {}, "by_norm": {}}

    cached = _IMAGE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    root_path = Path(image_root)
    log(f"开始建立图片索引: {root_path}")
    files: List[str] = []
    by_name: Dict[str, List[str]] = {}
    by_dir: Dict[str, List[str]] = {}
    by_uuid: Dict[str, List[str]] = {}
    by_norm: Dict[str, str] = {}
    scanned_count = 0

    if root_path.exists():
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            scanned_count += 1
            normalized = normalize_path_string(str(file_path))
            files.append(normalized)
            by_norm[os.path.normcase(normalized)] = normalized
            name_key = os.path.normcase(file_path.name)
            by_name.setdefault(name_key, []).append(normalized)
            dir_key = os.path.normcase(file_path.parent.name)
            by_dir.setdefault(dir_key, []).append(normalized)

            uuid_match = re.match(r"^([0-9a-fA-F-]{36})", file_path.name)
            if uuid_match:
                uuid_key = uuid_match.group(1).lower()
                by_uuid.setdefault(uuid_key, []).append(normalized)

            for part in file_path.parts:
                part_uuid = re.search(r"([0-9a-fA-F-]{36})", part)
                if part_uuid:
                    by_uuid.setdefault(part_uuid.group(1).lower(), []).append(normalized)

            if scanned_count % 5000 == 0:
                log(f"图片索引扫描进度: {scanned_count} 个文件")

    index = {"files": files, "by_name": by_name, "by_dir": by_dir, "by_uuid": by_uuid, "by_norm": by_norm}
    _IMAGE_INDEX_CACHE[cache_key] = index
    log(f"图片索引建立完成: {scanned_count} 个文件")
    return index


def _resolve_existing_image_path(path_str: str, source_image_prefix: str, image_root: str) -> str:
    if not path_str:
        return ""

    p = normalize_path_string(path_str)
    if re.match(r"^https?://", p, flags=re.IGNORECASE):
        return p

    candidates: List[str] = []

    def add_candidate(candidate: str) -> None:
        candidate = normalize_path_string(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if source_image_prefix:
        src = normalize_path_string(source_image_prefix)
        if p.startswith(src):
            suffix = p[len(src):].lstrip("/")
            add_candidate(str(Path(image_root) / suffix))

    downloaded_prefix = "downloaded_images/"
    if downloaded_prefix in p:
        suffix = p.split(downloaded_prefix, 1)[1].lstrip("/")
        add_candidate(str(Path(image_root) / suffix))

    if p.startswith("../downloaded_images/") or p.startswith("..\\downloaded_images\\"):
        suffix = re.split(r"\.\./downloaded_images[\\/]+", p, maxsplit=1, flags=re.IGNORECASE)
        if len(suffix) == 2 and suffix[1]:
            add_candidate(str(Path(image_root) / suffix[1].lstrip("/")))

    if os.path.isabs(p):
        add_candidate(p)

    if image_root:
        add_candidate(str(Path(image_root) / p.lstrip("./")))
        add_candidate(str(Path(image_root) / Path(p).name))

    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return normalize_path_string(candidate)

    index = _build_image_index(image_root)
    norm_candidates = [os.path.normcase(c) for c in candidates]
    for candidate in candidates:
        candidate_key = os.path.normcase(candidate)
        mapped = index["by_norm"].get(candidate_key)
        if mapped:
            return mapped

    for candidate in candidates:
        name_key = os.path.normcase(Path(candidate).name)
        matches = index["by_name"].get(name_key, [])
        if matches:
            return matches[0]

    uuid_patterns = [
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        r"([0-9a-fA-F]{32})",
    ]

    def _extract_uuid(text: str) -> str:
        for pattern in uuid_patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1).lower()
        return ""

    for candidate in candidates:
        uuid_key = _extract_uuid(candidate)
        if not uuid_key:
            continue
        matches = index["by_uuid"].get(uuid_key, [])
        if matches:
            return matches[0]

    # 如果文件名不一致，但图片仍然放在同名 UUID 子目录里，则退回到目录匹配
    for candidate in candidates:
        parent_key = os.path.normcase(Path(candidate).parent.name)
        if not parent_key:
            continue
        matches = index["by_dir"].get(parent_key, [])
        if matches:
            # 优先选同目录下与目标文件名最接近的图片；否则取第一张图
            target_stem = Path(candidate).stem.lower()
            for match in matches:
                if Path(match).stem.lower() == target_stem:
                    return match
            for match in matches:
                if Path(match).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
                    return match
            return matches[0]

    wanted = os.path.normcase(p).replace("\\", "/").lstrip("./")
    if wanted:
        scored: List[Tuple[int, int, str]] = []
        for existing in index["files"]:
            existing_norm = os.path.normcase(existing).replace("\\", "/")
            if existing_norm.endswith(wanted) or wanted.endswith(existing_norm):
                scored.append((0, len(existing_norm), existing))
                continue
            if Path(existing).name and Path(existing).name.lower() in wanted.lower():
                scored.append((1, len(existing_norm), existing))

        if scored:
            scored.sort(key=lambda item: (item[0], item[1]))
            return scored[0][2]

    return ""


def remap_image_path(path_str: str, source_image_prefix: str, image_root: str) -> str:
    return _resolve_existing_image_path(path_str, source_image_prefix, image_root)


def image_to_data_url(image_path_or_url: str) -> str:
    if re.match(r"^https?://", image_path_or_url, flags=re.IGNORECASE):
        return image_path_or_url

    path = Path(image_path_or_url)
    if not path.exists() or not path.is_file():
        if image_path_or_url:
            log(f"[warn] 图片文件不存在，已跳过: {image_path_or_url}")
        return ""

    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"

    raw = path.read_bytes()
    original_size = len(raw)

    # Large inline images can blow up the request body after base64 expansion.
    # Re-encode oversized images into a smaller JPEG payload before sending.
    should_compress = (
        Image is not None
        and (original_size > 512 * 1024 or mime not in {"image/jpeg", "image/jpg"})
    )
    if should_compress:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as img:
                    img.load()

                    for max_side, quality in ((1280, 80), (1024, 75), (768, 70)):
                        work_img = img.copy()

                        if max(work_img.size) > max_side:
                            work_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                        if work_img.mode not in {"RGB", "L"}:
                            if "A" in work_img.getbands():
                                background = Image.new("RGB", work_img.size, (255, 255, 255))
                                alpha = work_img.getchannel("A")
                                base = work_img.convert("RGB")
                                background.paste(base, mask=alpha)
                                work_img = background
                            else:
                                work_img = work_img.convert("RGB")

                        buffer = BytesIO()
                        work_img.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
                        compressed = buffer.getvalue()

                        if len(compressed) < original_size or len(compressed) <= 512 * 1024:
                            raw = compressed
                            mime = "image/jpeg"
                            break
        except Exception:
            pass

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(sample: Sample, extra_user_prefix: str = "") -> List[Dict[str, Any]]:
    sample_type = sample.sample_type
    has_image = sample_has_valid_image(sample)
    no_image_suffix = ""
    if sample_type == "Single":
        q_stem, options_text = extract_options_text(sample.question)
        prompt = SINGLE_PROMPT.format(question=q_stem, options_text=options_text)
        text = prompt
    elif sample_type == "Fill":
        fill_prompt = pick_fill_prompt(sample.source_task, sample.question)
        text = f"{fill_prompt}\n\n【问题】\n{sample.question}"
    elif sample_type == "Judge":
        text = JUDGE_PROMPT.format(question=sample.question)
    elif sample_type == "Multi":
        text = MULTI_PROMPT.format(question_text=sample.question)
    elif sample_type == "ShortAnswer":
        text = sample.question
    else:
        text = sample.question

    if not has_image:
        no_image_suffix = " 当前未找到对应图片时，只根据题目文字作答，不要提及图片缺失。"

    if extra_user_prefix:
        text = extra_user_prefix.rstrip() + "\n\n" + text
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    image_urls = resolve_sample_image_urls(sample)
    for index, image_url in enumerate(image_urls):
        if len(image_urls) > 1:
            content.append({"type": "text", "text": f"画{chr(65 + index)}:"})
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    if sample_type == "ShortAnswer":
        return [
            {"role": "system", "content": SHORT_ANSWER_SYSTEM_PROMPT + no_image_suffix},
            {"role": "user", "content": content},
        ]

    extra_rule = ""
    if sample_type == "Single":
        extra_rule = "这是单选题。你只能输出一个大写字母 A、B、C 或 D。"
    elif sample_type == "Multi":
        extra_rule = "这是多选题。你只能输出 2 到 4 个按顺序排列的大写字母，例如 ABC。"
    elif sample_type == "Judge":
        extra_rule = "这是判断题。你只能输出“是”或“否”。"
    elif sample_type == "Fill":
        extra_rule = "这是填空题。你只能输出简短最终答案，不要输出分析过程。"

    system_prompt = OBJECTIVE_SYSTEM_PROMPT + extra_rule + no_image_suffix
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def clean_text_core(text: str) -> str:
    x = text.strip()
    x = x.replace("\u3000", " ")
    x = re.sub(r"\s+", "", x)
    x = x.replace("。", "").replace("，", "").replace("、", "")
    x = x.replace("；", "").replace("：", "").replace(":", "")
    return x


def normalize_choice_text(text: str) -> str:
    return str(text or "").translate(str.maketrans("ＡＢＣＤａｂｃｄ", "ABCDabcd"))


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_choice_letters(raw: str, allow_multiple: bool) -> str:
    text = normalize_choice_text(raw)
    upper = text.upper()
    cue_patterns = [
        r"(?:最终答案|答案|answer|final answer|the answer is|therefore|thus|so|故选|应选|选择|选项)\s*[:：]?\s*([A-D](?:\s*[,，、/\s]\s*[A-D]){0,3}|[A-D]{1,4})",
        r"(?:选择|选|故|因此|所以)\s*([A-D](?:\s*[,，、/\s]\s*[A-D]){0,3}|[A-D]{1,4})",
    ]
    for pattern in cue_patterns:
        m = re.search(pattern, upper, flags=re.IGNORECASE)
        if not m:
            continue
        letters = re.findall(r"[A-D]", m.group(1))
        if not letters:
            continue
        unique_letters = dedupe_preserve_order(letters)
        if allow_multiple:
            if 2 <= len(unique_letters) <= 4:
                return "".join(unique_letters)
        else:
            if len(unique_letters) == 1:
                return unique_letters[0]

    line_groups = re.findall(r"(?m)^\s*([A-D]{1,4})\s*$", upper)
    if line_groups:
        candidate = dedupe_preserve_order(list(line_groups[-1]))
        if allow_multiple:
            if 2 <= len(candidate) <= 4:
                return "".join(candidate)
        else:
            if len(candidate) == 1:
                return candidate[0]

    standalone = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", upper)
    standalone = dedupe_preserve_order(standalone)
    if allow_multiple:
        if 2 <= len(standalone) <= 4:
            return "".join(standalone)
    else:
        if len(standalone) == 1:
            return standalone[0]

    return ""


def clean_single_answer(raw: str) -> str:
    return extract_choice_letters(raw, allow_multiple=False)


def clean_multi_answer(raw: str) -> str:
    return extract_choice_letters(raw, allow_multiple=True)


def clean_judge_answer(raw: str) -> str:
    concise = extract_concise_answer(raw, "Judge")
    text = clean_text_core(concise or raw)
    if "是" in text and "否" not in text:
        return "是"
    if "否" in text and "是" not in text:
        return "否"
    lower = text.lower()
    if "true" in lower or lower in {"yes", "y"}:
        return "是"
    if "false" in lower or lower in {"no", "n"}:
        return "否"
    return text[:1] if text else ""


def normalize_fill_token(text: str) -> List[str]:
    t = text.strip()
    t = re.sub(r"[\n\r\t ]+", "", t)
    parts = re.split(r"[、,，;；/]", t)
    parts = [re.sub(r"[。.!！？?]", "", p).strip() for p in parts if p.strip()]
    return sorted(parts)


def looks_like_reasoning_text(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    markers = (
        "let's",
        "wait",
        "double check",
        "analy",
        "request",
        "prompt",
        "user wants",
        "first,",
        "首先",
        "先看",
        "用户",
        "分析",
        "根据图",
        "先回忆",
        "仔细看",
        "不对",
        "题目",
    )
    return any(marker in lowered for marker in markers)


def extract_concise_answer(raw: str, sample_type: str) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return ""
    if sample_type == "Fill":
        for pattern in (
            r"(?:最终答案|答案|answer|final answer)[:：]\s*([^\n\r]{1,80})",
            r"(?:作品名|画名|名称|作者|朝代)[:：]\s*([^\n\r]{1,80})",
        ):
            m = re.search(pattern, txt, flags=re.IGNORECASE)
            if m:
                candidate = re.sub(r"\s+", " ", m.group(1)).strip(" \"'`[](){}")
                if candidate:
                    return candidate[:80]

        # 优先匹配典型的中文画名结尾（如“图”）
        m = re.search(r"([\u4e00-\u9fff]{2,40}图)", txt)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"作品名[:：]\s*([\u4e00-\u9fff]{2,40})", txt)
        if m2:
            return m2.group(1).strip()

        for l in lines:
            if len(l) <= 40 and not looks_like_reasoning_text(l):
                candidate = re.sub(r"\s+", " ", l).strip(" \"'`[](){}")
                if candidate:
                    return candidate

        # 仅在短句且不像推理过程时，才尝试关键词提取，避免把提示词片段截成“答案”
        for l in lines:
            if any(k in l for k in ("图", "画", "卷", "作", "名")) and len(l) <= 60 and not looks_like_reasoning_text(l):
                return re.sub(r"[^\u4e00-\u9fff，,、；;。.!！？?]", "", l).strip()

        # 对明显是推理过程的输出，不再激进裁剪成半句，保留首行以便排查
        return re.sub(r"\s+", " ", lines[0])[:120]

    # 其他类型默认返回原始文本的首行，避免把异常输出二次裁剪得更难排查
    return lines[0][:120]


def clean_objective_answer(sample_type: str, raw: str) -> str:
    if sample_type == "Single":
        return clean_single_answer(raw)
    if sample_type == "Multi":
        return clean_multi_answer(raw)
    if sample_type == "Judge":
        return clean_judge_answer(raw)
    if sample_type == "Fill":
        concise = extract_concise_answer(raw, "Fill")
        if concise and not looks_like_reasoning_text(concise):
            tokens = normalize_fill_token(concise)
            return "、".join(tokens) if tokens else clean_text_core(concise)
        tokens = normalize_fill_token(raw)
        return "、".join(tokens) if tokens and not looks_like_reasoning_text(raw) else ""
    return raw.strip()


def extract_model_answer(sample_type: str, raw: str) -> str:
    if sample_type == "Single":
        return clean_single_answer(raw)
    if sample_type == "Multi":
        return clean_multi_answer(raw)
    if sample_type == "Judge":
        return clean_judge_answer(raw)
    return extract_concise_answer(raw, sample_type)


def compare_objective(sample_type: str, cleaned_pred: str, reference: str) -> bool:
    cleaned_ref = clean_objective_answer(sample_type, reference)
    if sample_type == "Multi":
        return "".join(sorted(cleaned_pred)) == "".join(sorted(cleaned_ref))
    if sample_type == "Fill":
        pred_tokens = normalize_fill_token(cleaned_pred)
        ref_tokens = normalize_fill_token(cleaned_ref)
        return pred_tokens == ref_tokens
    return cleaned_pred == cleaned_ref


def extract_json_candidate(text: str) -> Optional[str]:
    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1)

    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        return brace.group(1)

    return None


def parse_eval_json(raw: str) -> Optional[Dict[str, Any]]:
    candidate = extract_json_candidate(raw)
    if not candidate:
        return None

    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate2)
        except Exception:
            return None


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_value = getattr(item, "text", "")
                if isinstance(text_value, str):
                    parts.append(text_value)
                continue
            text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                parts.append(text_value)
                continue
            if isinstance(item, dict):
                parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if str(part).strip()).strip()

    text_value = getattr(content, "text", None)
    if isinstance(text_value, str):
        return text_value.strip()

    try:
        return str(content).strip()
    except Exception:
        return ""


class FatalAPIError(RuntimeError):
    """Non-recoverable API failure that must stop the current benchmark run."""


class InvalidTokenError(FatalAPIError):
    """Fatal authentication failure that must stop the current benchmark run."""


class InsufficientCreditsError(FatalAPIError):
    """Fatal account-credit failure that must stop the current benchmark run."""


def is_invalid_token_401(error: Any) -> bool:
    message = str(error or "").lower()
    return "401" in message and "invalid token" in message


def is_insufficient_credits_402(error: Any) -> bool:
    message = str(error or "").lower()
    return "402" in message and (
        "insufficient credits" in message or "openrouter_credits" in message
    )


def fatal_api_exception(error: Any) -> Optional[FatalAPIError]:
    message = str(error or "")
    if is_invalid_token_401(message):
        return InvalidTokenError(
            "HTTP 401 Invalid token. Benchmark stopped; verify the API base and token before resuming. "
            + message[:500]
        )
    if is_insufficient_credits_402(message):
        return InsufficientCreditsError(
            "HTTP 402 Insufficient credits. Benchmark stopped; add credits or switch API credentials before resuming. "
            + message[:500]
        )
    return None


def raise_for_fatal_api_record(record: Dict[str, Any]) -> None:
    """Second-line circuit breaker for errors already wrapped into result rows."""
    for field in ("error", "evaluator_error"):
        fatal = fatal_api_exception(record.get(field))
        if fatal is not None:
            raise fatal


class AsyncVisionClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 120,
        max_retries: int = 6,
        backoff_base: float = 1.2,
        allow_text_fallback: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.allow_text_fallback = allow_text_fallback
        self.anthropic_version = "2023-06-01"
        self.openai_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.fatal_error_event = asyncio.Event()
        self.fatal_error: Optional[FatalAPIError] = None

    def _raise_if_fatal_api_error(self, error: Any = None) -> None:
        if error is not None:
            fatal = fatal_api_exception(error)
            if fatal is not None:
                self.fatal_error = fatal
                self.fatal_error_event.set()
        if self.fatal_error_event.is_set() and self.fatal_error is not None:
            raise self.fatal_error

    async def _openai_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Tuple[str, str]:
        last_error = ""

        def _stringify_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """仅在 fallback 时使用：把多模态内容退化为纯文本（图片描述替换）"""
            out: List[Dict[str, Any]] = []
            for m in messages:
                role = str(m.get("role", "user"))
                content = m.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            parts.append(str(item))
                            continue
                        itype = item.get("type")
                        if itype == "text":
                            parts.append(str(item.get("text", "")))
                        elif itype == "image_url":
                            image_url = item.get("image_url") or {}
                            url = ""
                            if isinstance(image_url, dict):
                                url = str(image_url.get("url", ""))
                            if url.startswith("data:"):
                                try:
                                    after = url.split("base64,", 1)[1]
                                    b64_len = len(after)
                                    est_size = int(b64_len * 3 / 4)
                                    parts.append(f"Image embedded (omitted, approx {est_size} bytes)")
                                except Exception:
                                    parts.append("Image embedded (omitted)")
                            else:
                                parts.append(f"Image URL: {url}")
                        else:
                            parts.append(str(item))
                    text = "\n".join(p for p in parts if p)
                else:
                    text = str(content)
                out.append({"role": role, "content": text})
            return out

        def _extract_response_content(response) -> Tuple[str, str]:
            """从 openai SDK response 中提取文本和 finish_reason"""
            if not getattr(response, "choices", None):
                resp_str = str(response)
                candidate = extract_json_candidate(resp_str)
                if candidate:
                    try:
                        jd = json.loads(candidate)
                        for k in ("content", "output", "outputs", "text", "completion"):
                            if k in jd:
                                parsed_alt = extract_text_from_content(jd.get(k))
                                if parsed_alt:
                                    finish_reason = str(
                                        jd.get("finish_reason", "") or jd.get("stop_reason", "") or ""
                                    )
                                    return parsed_alt, finish_reason
                    except Exception:
                        pass
                return "", ""

            choice = response.choices[0]
            message = choice.message
            content = extract_text_from_content(getattr(message, "content", None))

            if not content:
                text_alt = getattr(choice, "text", None)
                if isinstance(text_alt, str) and text_alt.strip():
                    content = text_alt.strip()
                else:
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        content = extract_text_from_content(getattr(delta, "content", delta))

            if not content and os.environ.get("BENCH_DEBUG") == "1":
                try:
                    log(f"openai-style empty content; raw response preview: {str(response)[:2000]}")
                except Exception:
                    pass

            if not content:
                refusal = getattr(message, "refusal", None)
                if isinstance(refusal, str) and refusal.strip():
                    content = refusal.strip()
            if not content:
                reasoning = getattr(message, "reasoning", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    content = reasoning.strip()
            if not content:
                reasoning_details = getattr(message, "reasoning_details", None)
                if isinstance(reasoning_details, list):
                    parts: List[str] = []
                    for it in reasoning_details:
                        if isinstance(it, dict) and isinstance(it.get("text"), str):
                            parts.append(it.get("text"))
                    if parts:
                        content = "\n".join(parts).strip()
            if not content:
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    content = extract_text_from_content(tool_calls)

            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            return content, finish_reason

        # ── 多模态错误关键词，触发时 fallback 到纯文本 ──────────────────────────
        _VISION_ERROR_TOKENS = (
            "image", "content", "multimodal", "vision", "invalid type", "unsupported"
        )

        for attempt in range(self.max_retries):
            self._raise_if_fatal_api_error()
            try:
                # ★ 核心修复：直接传原始 messages（含 image_url），不再 stringify
                req_started = time.perf_counter()
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self.timeout,
                    ),
                    timeout=self.timeout + 5,
                )
                if os.environ.get("BENCH_DEBUG") == "1":
                    log(
                        f"[info] openai-style request finished: model={model}, "
                        f"elapsed={time.perf_counter() - req_started:.2f}s"
                    )
                return _extract_response_content(response)

            except asyncio.TimeoutError:
                last_error = f"request timeout after {self.timeout + 5}s"
                if os.environ.get("BENCH_DEBUG") == "1":
                    log(f"[warn] openai-style request timeout: model={model}, max_tokens={max_tokens}, attempt={attempt}")

            except Exception as e:
                self._raise_if_fatal_api_error(e)
                last_error = str(e)
                lowered = last_error.lower()

                # 仅当明确是多模态/图片不支持时，才 fallback 到纯文本重试一次
                if self.allow_text_fallback and any(token in lowered for token in _VISION_ERROR_TOKENS):
                    log(f"[warn] 模型 {model} 触发多模态 fallback，原始错误: {last_error[:200]}")
                    try:
                        text_only_messages = _stringify_openai_messages(messages)
                        req_started = time.perf_counter()
                        response = await asyncio.wait_for(
                            self.openai_client.chat.completions.create(
                                model=model,
                                messages=text_only_messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                timeout=self.timeout,
                            ),
                            timeout=self.timeout + 5,
                        )
                        if os.environ.get("BENCH_DEBUG") == "1":
                            log(
                                f"[info] openai fallback finished: model={model}, "
                                f"elapsed={time.perf_counter() - req_started:.2f}s"
                            )
                        return _extract_response_content(response)
                    except asyncio.TimeoutError:
                        last_error = f"fallback request timeout after {self.timeout + 5}s"
                    except Exception as fallback_exc:
                        self._raise_if_fatal_api_error(fallback_exc)
                        last_error = f"{last_error}; fallback: {fallback_exc}"

            sleep_s = self.backoff_base * (2 ** attempt) + random.random()
            await asyncio.sleep(min(sleep_s, 30.0))

        raise RuntimeError(f"openai-style request failed after retries: {last_error}")

    @staticmethod
    def _parse_data_url_image(data_url: str) -> Optional[Tuple[str, str]]:
        m = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", data_url, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1), m.group(2)

    def _to_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_text: Optional[str] = None
        out_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_text = content
                elif isinstance(content, list):
                    texts: List[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(str(item.get("text", "")))
                    system_text = "\n".join(texts).strip()
                continue

            anthropic_content: List[Dict[str, Any]] = []
            if isinstance(content, str):
                anthropic_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        anthropic_content.append({"type": "text", "text": str(item.get("text", ""))})
                    elif item_type == "image_url":
                        image_url = item.get("image_url", {})
                        url = ""
                        if isinstance(image_url, dict):
                            url = str(image_url.get("url", ""))
                        parsed = self._parse_data_url_image(url)
                        if parsed is not None:
                            media_type, b64_data = parsed
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                }
                            )
                        else:
                            anthropic_content.append(
                                {
                                    "type": "text",
                                    "text": f"Image URL: {url}",
                                }
                            )

            out_messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": anthropic_content,
                }
            )

        return system_text, out_messages

    async def _anthropic_messages(
        self,
        session: aiohttp.ClientSession,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Tuple[str, str]:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

        system, anthropic_messages = self._to_anthropic_messages(messages)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        last_error = ""
        for attempt in range(self.max_retries):
            self._raise_if_fatal_api_error()
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        data = json.loads(text)
                        # anthropic 风格接口可能在不同字段返回文本，尝试多种键名
                        content = data.get("content", [])
                        finish_reason = str(data.get("stop_reason", "") or data.get("finish_reason", "") or "")
                        parsed = extract_text_from_content(content)
                        if not parsed:
                            for alt in ("outputs", "output", "text", "completion"):
                                if alt in data:
                                    parsed = extract_text_from_content(data.get(alt))
                                    if parsed:
                                        break
                            # 再次尝试从原始响应字符串中提取 JSON 内的文本
                            if not parsed:
                                candidate = extract_json_candidate(text)
                                if candidate:
                                    try:
                                        jd = json.loads(candidate)
                                        for k in ("content", "output", "outputs", "text", "completion"):
                                            if k in jd:
                                                parsed = extract_text_from_content(jd.get(k))
                                                if parsed:
                                                    finish_reason = str(jd.get("stop_reason", "") or jd.get("finish_reason", "") or finish_reason)
                                                    break
                                    except Exception:
                                        pass
                        if not parsed and os.environ.get("BENCH_DEBUG") == "1":
                            try:
                                log(f"anthropic empty content; response snippet: {text[:1000]}")
                                log(f"anthropic sent payload preview: model={model}, messages_len={len(anthropic_messages)}")
                            except Exception:
                                pass
                        return parsed, finish_reason

                    if resp.status in {408, 409, 425, 429, 500, 502, 503, 504}:
                        last_error = f"HTTP {resp.status}: {text[:500]}"
                    else:
                        error = RuntimeError(f"HTTP {resp.status}: {text[:1000]}")
                        self._raise_if_fatal_api_error(error)
                        raise error
            except asyncio.TimeoutError:
                last_error = "request timeout"
            except aiohttp.ClientError as e:
                last_error = f"network error: {e}"

            sleep_s = self.backoff_base * (2 ** attempt) + random.random()
            await asyncio.sleep(min(sleep_s, 30.0))

        raise RuntimeError(f"anthropic-style request failed after retries: {last_error}")

    async def generate(
        self,
        session: aiohttp.ClientSession,
        model: str,
        api_style: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Tuple[str, str]:
        self._raise_if_fatal_api_error()
        style = api_style.strip().lower()
        if style == "openai":
            res = await self._openai_chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return res
        if style == "anthropic":
            res = await self._anthropic_messages(
                session=session,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return res
        raise ValueError(f"unsupported api_style: {api_style}")

    async def preflight_openai_models(self, session: aiohttp.ClientSession) -> List[str]:
        models_response = await self.openai_client.models.list(timeout=min(self.timeout, 30))
        model_ids: List[str] = []
        for item in models_response.data:
            if isinstance(item.id, str):
                model_ids.append(item.id)
        return model_ids


def infer_api_style_from_model(model: str) -> str:
    m = model.strip().lower()
    if m.startswith("anthropic/") or "claude" in m:
        return "anthropic"
    return "openai"


def parse_model_style_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            continue
        name, style = raw.split("=", 1)
        name = name.strip()
        style = style.strip().lower()
        if not name or style not in {"openai", "anthropic"}:
            continue
        out[name] = style
    return out


def build_model_configs(models: List[str], default_api_style: str, overrides: Dict[str, str]) -> List[ModelConfig]:
    configs: List[ModelConfig] = []
    default_style = default_api_style.strip().lower()
    for model in models:
        style = overrides.get(model)
        if not style:
            if default_style == "auto":
                style = infer_api_style_from_model(model)
            else:
                style = default_style
        configs.append(ModelConfig(name=model, api_style=style))
    return configs


async def evaluate_one_sample(
    sample: Sample,
    model_config: ModelConfig,
    client: AsyncVisionClient,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    objective_max_tokens: int,
    short_answer_max_tokens: int,
    short_answer_variants: List[str],
) -> Dict[str, Any]:
    if os.environ.get("BENCH_DEBUG") == "1":
        log(f"[{model_config.name}] 开始处理样本: {sample.sample_id} (类型: {sample.sample_type})")
    
    async def _generate_with_length_retry(
        *,
        model: str,
        api_style: str,
        messages: List[Dict[str, Any]],
        initial_max_tokens: int,
        hard_cap: int,
        temperature: float = 0.0,
    ) -> Tuple[str, str, int, int]:
        current_max_tokens = max(64, initial_max_tokens)
        attempts = 0
        last_text = ""
        last_reason = ""
        truncated_reasons = {"length", "max_tokens", "max_tokens_exceeded"}
        max_length_retries = 8

        while True:
            if os.environ.get("BENCH_DEBUG") == "1":
                log(
                    f"[{model_config.name}] 样本 {sample.sample_id} 发起请求: "
                    f"api_style={api_style}, max_tokens={current_max_tokens}, attempt={attempts}"
                )
            last_text, last_reason = await client.generate(
                session=session,
                model=model,
                api_style=api_style,
                messages=messages,
                temperature=temperature,
                max_tokens=current_max_tokens,
            )
            if os.environ.get("BENCH_DEBUG") == "1":
                log(
                    f"[{model_config.name}] 样本 {sample.sample_id} 请求返回: "
                    f"finish_reason={last_reason}, text_len={len(last_text.strip())}"
                )
            is_truncated = str(last_reason).lower() in truncated_reasons
            if last_text.strip() and not is_truncated:
                return last_text, last_reason, current_max_tokens, attempts

            # 调试信息：当返回为空时记录更多上下文，便于定位服务端为何未返回文本
            if os.environ.get("BENCH_DEBUG") == "1" and (not last_text.strip() or is_truncated):
                try:
                    preview_msgs = None
                    try:
                        preview_msgs = [extract_text_from_content(m.get("content", "")) for m in messages]
                    except Exception:
                        preview_msgs = [str(m)[:500] for m in messages]
                    log(
                        f"generate returned incomplete text; model={model}, api_style={api_style}, "
                        f"finish_reason={last_reason}, attempts={attempts}, max_tokens={current_max_tokens}, "
                        f"text_len={len(last_text.strip())}"
                    )
                    log(f"messages preview: {preview_msgs[:3]}")
                except Exception:
                    pass

            if not is_truncated:
                return last_text, last_reason, current_max_tokens, attempts

            if current_max_tokens >= hard_cap or attempts >= max_length_retries:
                return last_text, last_reason, current_max_tokens, attempts

            current_max_tokens = min(current_max_tokens * 2, hard_cap)
            attempts += 1

    if os.environ.get("BENCH_DEBUG") == "1":
        log(f"[{model_config.name}] 样本 {sample.sample_id} 等待信号量...")
    async with sem:
        if os.environ.get("BENCH_DEBUG") == "1":
            log(f"[{model_config.name}] 样本 {sample.sample_id} 已获取信号量")
        started = time.perf_counter()
        base_record: Dict[str, Any] = {
            "base_id": sample.sample_id,
            "type": sample.sample_type,
            "task": sample.source_task,
            "question": sample.question,
            "image_path": sample.image_path,
            "image_file": sample.image_file,
            "image_paths": sample.image_paths,
            "image_files": sample.image_files,
            "reference_answer": sample.reference_answer,
        }

        variants = short_answer_variants if sample.sample_type == "ShortAnswer" else ["noprompt"]
        if not sample_has_all_images(sample):
            failures = []
            for variant in variants:
                row = dict(base_record)
                row.update({
                    "id": sample.sample_id + (f"__{variant}" if sample.sample_type == "ShortAnswer" else ""),
                    "prompt_variant": variant,
                    "raw_model_output": "",
                    "error": "one or more required images are missing; text-only fallback is disabled",
                    "response_time_seconds": round(time.perf_counter() - started, 3),
                })
                failures.append(row)
            return failures if sample.sample_type == "ShortAnswer" else failures[0]

        if sample.sample_type in {"Single", "Judge"}:
            initial_max_tokens, hard_cap = 64, 256
        elif sample.sample_type in {"Fill", "ShortAnswer"}:
            initial_max_tokens, hard_cap = short_answer_max_tokens, 8192
        else:
            initial_max_tokens, hard_cap = objective_max_tokens, 8192

        rows_out: List[Dict[str, Any]] = []
        for variant in variants:
            row = dict(base_record)
            row.update({
                "id": sample.sample_id + (f"__{variant}" if sample.sample_type == "ShortAnswer" else ""),
                "prompt_variant": variant,
            })
            try:
                prefix = build_local_info_text(sample) if variant == "localinfo" else ""
                messages = build_messages(sample, extra_user_prefix=prefix)
                raw_output, finish_reason, used_max_tokens, retry_count = await _generate_with_length_retry(
                    model=model_config.name,
                    api_style=model_config.api_style,
                    messages=messages,
                    initial_max_tokens=initial_max_tokens,
                    hard_cap=hard_cap,
                )
                row.update({
                    "raw_model_output": raw_output,
                    "model_answer": extract_model_answer(sample.sample_type, raw_output),
                    "finish_reason": finish_reason,
                    "was_truncated": str(finish_reason).lower() in {"length", "max_tokens", "max_tokens_exceeded"},
                    "generation_max_tokens": used_max_tokens,
                    "generation_retry_count": retry_count,
                    "response_time_seconds": round(time.perf_counter() - started, 3),
                })
            except FatalAPIError:
                raise
            except Exception as exc:
                row.update({
                    "raw_model_output": "",
                    "error": str(exc)[:1000],
                    "response_time_seconds": round(time.perf_counter() - started, 3),
                })
            rows_out.append(row)

        return rows_out if sample.sample_type == "ShortAnswer" else rows_out[0]


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def format_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0/0"
    filled = int(round(width * current / total))
    filled = max(0, min(width, filled))
    empty = width - filled
    pct = (current / total) * 100.0
    return f"[{('=' * filled)}{('-' * empty)}] {current}/{total} ({pct:.1f}%)"


def summarize_results(model: str, rows: List[Dict[str, Any]], short_bad_threshold: int) -> str:
    def _task_group(task_name: str) -> str:
        if task_name.startswith("T"):
            return task_name.split("_", 1)[0]
        return task_name

    def _type_name(sample_type: str) -> str:
        return str(sample_type or "Unknown").strip().lower()

    def _fill_char_tokens(s: str) -> List[str]:
        stripped = str(s).strip()
        return list(stripped) if stripped else []

    def _get_fill_text(row: Dict[str, Any]) -> str:
        for key in ["cleaned_model_output", "cleaned_answer", "model_answer", "raw_model_output"]:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _compute_fill_metrics(items: List[Dict[str, Any]]) -> Tuple[int, float, float, float]:
        exact = 0
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for r in items:
            pred_text = _get_fill_text(r)
            ref_text = str(r.get("reference_answer", ""))

            if "is_correct" in r:
                is_exact = r["is_correct"] is True
            else:
                is_exact = pred_text.strip() == ref_text.strip()
            if is_exact:
                exact += 1

            pred_counter = Counter(_fill_char_tokens(pred_text))
            ref_counter = Counter(_fill_char_tokens(ref_text))

            pred_len = sum(pred_counter.values())
            ref_len = sum(ref_counter.values())

            if pred_len == 0 and ref_len == 0:
                precision = recall = f1 = 1.0
            else:
                overlap_count = sum((pred_counter & ref_counter).values())
                precision = (overlap_count / pred_len) if pred_len > 0 else 0.0
                recall = (overlap_count / ref_len) if ref_len > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        macro_precision = (sum(precisions) / len(precisions) * 100.0) if precisions else 0.0
        macro_recall = (sum(recalls) / len(recalls) * 100.0) if recalls else 0.0
        macro_f1 = (sum(f1s) / len(f1s) * 100.0) if f1s else 0.0
        return exact, macro_precision, macro_recall, macro_f1

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        task = _task_group(str(row.get("task", "Unknown")))
        sample_type = _type_name(str(row.get("type", "Unknown")))
        grouped.setdefault(task, {}).setdefault(sample_type, []).append(row)

    task_order = [f"T{i}" for i in range(1, 9)]
    # 包含默认顺序外的其他任务（例如 T0），保持稳定排序
    extra_tasks = sorted([k for k in grouped.keys() if k not in task_order])
    if extra_tasks:
        task_order.extend(extra_tasks)
    type_order = ["single", "judge", "multi", "fill", "shortanswer"]

    lines: List[str] = [f"# Benchmark Summary - {model}", ""]

    for task in task_order:
        if task not in grouped:
            continue
        lines.append(f"## {task}")
        lines.append("")

        for sample_type in type_order:
            items = grouped[task].get(sample_type, [])
            if not items:
                continue

            total = len(items)
            lines.append(f"### {sample_type}")
            lines.append("")

            if sample_type in {"single", "judge", "multi"}:
                correct = sum(1 for r in items if r.get("is_correct") is True)
                acc = (correct / total * 100.0) if total else 0.0
                lines.append(f"- 总样本量: {total}")
                lines.append(f"- 答对样本量: {correct}")
                lines.append(f"- 准确率: {acc:.2f}%")
                lines.append("")
                continue

            if sample_type == "fill":
                exact, macro_precision, macro_recall, macro_f1 = _compute_fill_metrics(items)
                macro_exact = (exact / total * 100.0) if total else 0.0
                lines.append(f"- 总样本量: {total}")
                lines.append(f"- 答对样本量: {exact}")
                lines.append(f"- macro_exact_match: {macro_exact:.2f}%")
                lines.append(f"- macro_precision: {macro_precision:.2f}%")
                lines.append(f"- macro_recall: {macro_recall:.2f}%")
                lines.append(f"- macro_f1: {macro_f1:.2f}%")
                lines.append("")
                continue

            if sample_type == "shortanswer":
                variant_map = {"noprompt": "无提示", "localinfo": "局部标注"}
                variants: Dict[str, List[Dict[str, Any]]] = {}
                for row in items:
                    variant = str(row.get("prompt_variant") or "noprompt").strip().lower()
                    variants.setdefault(variant, []).append(row)

                for variant in ("noprompt", "localinfo"):
                    group = variants.get(variant, [])
                    if not group:
                        continue
                    valid = [row for row in group if row.get("evaluator_status") == "ok"]
                    scores = [float(row["evaluator_score_0_100"]) for row in valid]
                    lines.append(f"#### {sample_type} — {variant_map.get(variant, variant)}")
                    lines.append(f"- 生成样本数: {len(group)}")
                    lines.append(f"- 七维有效评分数: {len(valid)}")
                    lines.append(f"- 七维平均分 (0-100): {(sum(scores) / len(scores)) if scores else 0.0:.2f}")
                    for dimension in seven_dim_judge.DIMENSIONS:
                        values = [
                            float(row["evaluator_dimensions"][dimension]["score"])
                            for row in valid
                            if isinstance(row.get("evaluator_dimensions"), dict)
                            and isinstance(row["evaluator_dimensions"].get(dimension), dict)
                        ]
                        lines.append(
                            f"- {dimension}: {(sum(values) / len(values)) if values else 0.0:.2f}"
                        )
                    lines.append("")
                continue

    return "\n".join(lines)


def build_simple_report_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    simple_rows: List[Dict[str, Any]] = []
    for row in rows:
        raw_model_output = row.get("raw_model_output", "")
        simple_row: Dict[str, Any] = {
            "id": row.get("id"),
            "task": row.get("task"),
            "type": row.get("type"),
            "question": row.get("question", ""),
            "reference_answer": row.get("reference_answer", ""),
            # Prefer explicit `model_answer` (concise) if the evaluator code populated it,
            # otherwise fall back to the raw model output.
            "model_answer": row.get("model_answer") or raw_model_output,
        }
        for key in (
            "raw_model_output",
            "cleaned_model_output",
            "is_correct",
            "scoring_status",
            "prompt_variant",
            "finish_reason",
            "was_truncated",
            "generation_max_tokens",
            "generation_retry_count",
            "response_time_seconds",
            "error",
        ):
            if key in row:
                simple_row[key] = row.get(key)
        for key in ("image_path", "image_file", "image_paths", "image_files"):
            if key in row:
                simple_row[key] = row.get(key)
        if is_open_ended_type(str(row.get("type") or "")):
            for key in (
                "evaluator_status",
                "evaluator_error",
                "evaluator_model",
                "evaluator_rubric_id",
                "evaluator_prompt_hash",
                "evaluator_request_hash",
                "evaluator_candidate_hash",
                "evaluator_dimensions",
                "evaluator_score",
                "evaluator_score_0_100",
                "evaluator_overall_comment",
                "evaluator_raw_json",
                "evaluator_raw_text",
                "evaluator_finish_reason",
                "evaluator_generation_max_tokens",
            ):
                if key in row:
                    simple_row[key] = row.get(key)
        simple_rows.append(simple_row)
    return simple_rows


def build_markdown_report(model: str, rows: List[Dict[str, Any]]) -> str:
    # 补全从 report 文件中可能缺失的派生字段（例如中断后简化 report 里可能没有 is_correct）
    for row in rows:
        try:
            st = str(row.get("type", "")).strip()
            st_l = st.lower()
            if st_l in {"single", "judge", "multi"}:
                sample_type = st_l.title()
                # 优先使用 model_answer，否则尝试 raw_model_output
                model_answer = str(row.get("model_answer") or row.get("raw_model_output") or "")
                if "cleaned_model_output" not in row:
                    row["cleaned_model_output"] = clean_objective_answer(sample_type, model_answer)
                if "is_correct" not in row:
                    row["is_correct"] = compare_objective(sample_type, row.get("cleaned_model_output", ""), row.get("reference_answer", ""))
        except Exception:
            # 不要因为单条样本出错而阻塞总体统计
            continue

    # 如果行里缺少 prompt_variant，则尝试从 id 后缀推断（例如 T0_5__noprompt）
    for row in rows:
        try:
            if "prompt_variant" not in row:
                rid = str(row.get("id") or "")
                if "__" in rid:
                    _base, suf = rid.split("__", 1)
                    if suf in {"noprompt", "localinfo"}:
                        row["prompt_variant"] = suf
        except Exception:
            continue

    # Prefer the richer summarization which computes task/type metrics
    short_bad_threshold = 6
    try:
        return summarize_results(model, rows, short_bad_threshold)
    except Exception:
        # Fallback to a simple per-sample listing if summarizer fails
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            task = str(row.get("task", "Unknown")).strip() or "Unknown"
            grouped.setdefault(task, []).append(row)

        lines: List[str] = [f"# Benchmark Report - {model}", ""]
        for task in sorted(grouped.keys()):
            lines.append(f"## {task}")
            lines.append("")
            for row in grouped[task]:
                row_id = row.get("id", "")
                sample_type = row.get("type", "")
                question = str(row.get("question", "")).strip()
                model_answer = str(row.get("model_answer", "")).strip()
                ref_answer = str(row.get("reference_answer", "")).strip()
                lines.append(f"- **{row_id}** [{sample_type}]")
                lines.append(f"  - question: {question}")
                lines.append(f"  - reference_answer: {ref_answer}")
                lines.append(f"  - model_answer: {model_answer}")
                if row.get("type") == "ShortAnswer":
                    lines.append(f"  - evaluator_score: {row.get('evaluator_score', '')}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def load_report_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log(f"已有 report 读取失败，将从头开始: {path} ({type(exc).__name__}: {exc})")
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    log(f"已有 report 格式不是列表，将从头开始: {path}")
    return []


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


async def evaluate_model(
    model_config: ModelConfig,
    samples: List[Sample],
    client: AsyncVisionClient,
    concurrency: int,
    objective_max_tokens: int,
    short_answer_max_tokens: int,
    short_answer_variants: List[str],
    on_result: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 20), ssl=False)
    timeout = aiohttp.ClientTimeout(total=None)

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        log(f"[{model_config.name}] 开始创建评测任务...")
        tasks = [
            asyncio.create_task(
                evaluate_one_sample(
                    sample=s,
                    model_config=model_config,
                    client=client,
                    session=session,
                    sem=sem,
                    objective_max_tokens=objective_max_tokens,
                    short_answer_max_tokens=short_answer_max_tokens,
                    short_answer_variants=short_answer_variants,
                )
            )
            for s in samples
        ]

        results: List[Dict[str, Any]] = []
        total = len(tasks)
        finished = 0
        last_progress_len = 0
        log(f"[{model_config.name}] 已启动 {total} 个样本评测任务，开始并发执行...")
        for future in asyncio.as_completed(tasks):
            try:
                row = await future
                # evaluate_one_sample 可能返回单个 dict 或者 shortanswer 的 list
                if isinstance(row, list):
                    for sub in row:
                        results.append(sub)
                        if on_result is not None:
                            on_result(sub, finished + 1, total)
                        raise_for_fatal_api_record(sub)
                else:
                    results.append(row)
                    if on_result is not None:
                        on_result(row, finished + 1, total)
                    raise_for_fatal_api_record(row)
            except FatalAPIError as exc:
                log(f"[{model_config.name}] FATAL: {exc}")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            except Exception as e:
                log(f"[{model_config.name}] 样本处理异常: {type(e).__name__}: {str(e)[:500]}")
                raise
            finished += 1
            bar = format_progress_bar(finished, total)
            line = f"[{model_config.name}] 评测进度 {bar}"
            padding = max(0, last_progress_len - len(line))
            sys.stdout.write("\r" + line + (" " * padding))
            sys.stdout.flush()
            last_progress_len = len(line)

        sys.stdout.write("\n")
        sys.stdout.flush()
        log(f"[{model_config.name}] 所有评测任务已完成")

        return results


def is_open_ended_type(sample_type: str) -> bool:
    return str(sample_type or "").strip().lower() == "shortanswer"


def _row_image_files(row: Dict[str, Any]) -> List[str]:
    values = row.get("image_files")
    if isinstance(values, list):
        return [str(x) for x in values if str(x).strip()]
    value = str(row.get("image_file") or "").strip()
    return [value] if value else []


def build_seven_dim_item(row: Dict[str, Any]) -> Dict[str, Any]:
    paintings = []
    for index, image_file in enumerate(_row_image_files(row)):
        path = Path(image_file)
        tag_path = path.parent / "tag.json"
        if not tag_path.exists():
            raise FileNotFoundError(f"tag.json missing for image: {image_file}")
        match = seven_dim_judge.oracle.UUID.search(str(path))
        painting_id = match.group(0).lower() if match else path.stem
        paintings.append(seven_dim_judge.painting_evidence(
            str(tag_path), chr(65 + index), painting_id
        ))
    if not paintings:
        raise ValueError("no painting evidence available for seven-dimensional judge")
    return {
        "question": str(row.get("question") or ""),
        "reference": str(row.get("reference_answer") or ""),
        "candidate": str(row.get("raw_model_output") or "").strip(),
        "painting_evidence": paintings,
    }


def score_objective_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in rows:
        if is_open_ended_type(str(row.get("type") or "")):
            continue
        raw = str(row.get("raw_model_output") or "")
        sample_type = str(row.get("type") or "")
        model_answer = extract_model_answer(sample_type, raw)
        cleaned = clean_objective_answer(sample_type, model_answer or raw)
        row["model_answer"] = model_answer
        row["cleaned_model_output"] = cleaned
        row["is_correct"] = compare_objective(
            sample_type, cleaned, str(row.get("reference_answer") or "")
        )
        row["scoring_status"] = "ok"
    return rows


async def score_report_rows(
    tested_model: str,
    rows: List[Dict[str, Any]],
    evaluator_config: ModelConfig,
    client: AsyncVisionClient,
    concurrency: int,
    evaluator_max_tokens: int,
    prompt_path: Path,
) -> List[Dict[str, Any]]:
    if os.environ.get("BENCH_DEBUG") == "1":
        log(f"[{tested_model}] score_report_rows start rows={len(rows)}")
    if re.sub(r"[^a-z0-9]", "", tested_model.lower()) == re.sub(
        r"[^a-z0-9]", "", evaluator_config.name.lower()
    ):
        raise ValueError("Judge model must differ from the tested model")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    prompt_hash = seven_dim_judge.sha256_text(
        system_prompt + "\n" + seven_dim_judge.INPUT_SCHEMA
    )
    if os.environ.get("BENCH_DEBUG") == "1":
        log(f"[{tested_model}] seven-dimensional prompt loaded")

    # Objective questions are scored deterministically in this separate scoring phase.
    score_objective_rows(rows)

    open_indices = [
        index for index, row in enumerate(rows)
        if is_open_ended_type(str(row.get("type") or ""))
    ]
    if os.environ.get("BENCH_DEBUG") == "1":
        log(f"[{tested_model}] objective scoring done; open_rows={len(open_indices)}")
    if not open_indices:
        return rows

    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 20), ssl=False)
    timeout = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker(index: int) -> Tuple[int, Dict[str, Any]]:
            row = rows[index]
            candidate = str(row.get("raw_model_output") or "").strip()
            finish_reason = str(row.get("finish_reason") or "").lower()
            if not candidate:
                return index, {"evaluator_status": "error", "evaluator_error": "empty candidate"}
            if row.get("was_truncated") is True or finish_reason in {"length", "max_tokens", "max_tokens_exceeded"}:
                return index, {"evaluator_status": "error", "evaluator_error": "truncated candidate"}

            try:
                item = build_seven_dim_item(row)
                messages = seven_dim_judge.judge_messages(item, system_prompt)
                request_hash = seven_dim_judge.sha256_text(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True)
                )
                candidate_hash = seven_dim_judge.sha256_text(candidate)
                if (
                    row.get("evaluator_status") == "ok"
                    and row.get("evaluator_model") == evaluator_config.name
                    and row.get("evaluator_prompt_hash") == prompt_hash
                    and row.get("evaluator_request_hash") == request_hash
                    and row.get("evaluator_candidate_hash") == candidate_hash
                ):
                    return index, {}

                max_tokens = evaluator_max_tokens
                while True:
                    async with sem:
                        raw, reason = await client.generate(
                            session=session,
                            model=evaluator_config.name,
                            api_style=evaluator_config.api_style,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=max_tokens,
                        )
                    if str(reason).lower() not in {"length", "max_tokens", "max_tokens_exceeded"}:
                        break
                    if max_tokens >= 8192:
                        raise ValueError("Judge output truncated at max token cap")
                    max_tokens = min(8192, max_tokens * 2)

                dimensions, score_0_10, score_0_100, comment = seven_dim_judge.parse_judge(raw)
                return index, {
                    "evaluator_status": "ok",
                    "evaluator_model": evaluator_config.name,
                    "evaluator_rubric_id": seven_dim_judge.RUBRIC_ID,
                    "evaluator_prompt_hash": prompt_hash,
                    "evaluator_request_hash": request_hash,
                    "evaluator_candidate_hash": candidate_hash,
                    "evaluator_dimensions": dimensions,
                    "evaluator_score": score_0_10,
                    "evaluator_score_0_100": score_0_100,
                    "evaluator_overall_comment": comment,
                    "evaluator_raw_json": {
                        "dimensions": dimensions,
                        "overall_comment": comment,
                    },
                    "evaluator_raw_text": raw,
                    "evaluator_finish_reason": reason,
                    "evaluator_generation_max_tokens": max_tokens,
                }
            except FatalAPIError:
                raise
            except Exception as exc:
                return index, {
                    "evaluator_status": "error",
                    "evaluator_model": evaluator_config.name,
                    "evaluator_prompt_hash": prompt_hash,
                    "evaluator_error": str(exc)[:1000],
                }

        jobs = [asyncio.create_task(worker(index)) for index in open_indices]
        completed = 0
        try:
            for future in asyncio.as_completed(jobs):
                index, update = await future
                if update:
                    rows[index].update(update)
                    raise_for_fatal_api_record(update)
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    log(f"[{tested_model}] seven-dimensional scoring {completed}/{len(jobs)}")
        except FatalAPIError as exc:
            log(f"[{tested_model}] FATAL: {exc}")
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise

    return rows


def default_input_files() -> List[str]:
    return [
        # "T0.jsonl",
        "T1_Basic_Attribute.jsonl",
        "T2_Object_Recognition.jsonl",
        "T3_Composition_Recognition.jsonl",
        "T4_Technique_Recognition.jsonl",
        "T5_Artistic_Conception.jsonl",
        "T6_Cultural_Symbolism.jsonl",
        "T7_Single_Painting_Analysis.jsonl",
        "T8_Cross_Painting_Analysis.jsonl",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="异步多模型多模态 Benchmark 评测脚本")
    parser.add_argument(
        "action", nargs="?", choices=("generate", "score", "summarize", "all"), default="all",
        help="generate仅生成；score统一规则/七维评分；summarize离线汇总；all依次执行三阶段",
    )
    parser.add_argument("--data-dir", default=".", help="Benchmark 数据目录")
    parser.add_argument("--input-files", nargs="*", default=default_input_files(), help="待评测 JSONL 文件列表")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="结果输出目录")

    parser.add_argument("--api-base", default="https://api.zjuqx.cn/", help="OpenAI 兼容 API Base")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API Key；默认读取 OPENAI_API_KEY")
    parser.add_argument(
        "--judge-api-base", default=os.environ.get("ORACLE_JUDGE_API_BASE", ""),
        help="Judge API Base；为空时复用 --api-base",
    )
    parser.add_argument(
        "--judge-api-key", default=os.environ.get("ORACLE_JUDGE_API_KEY", ""),
        help="Judge API Key；为空时复用 --api-key",
    )
    parser.add_argument(
        "--api-style",
        default="openai",
        choices=["auto", "openai", "anthropic"],
        help="默认请求风格；对 zjuqx 建议使用 openai",
    )
    parser.add_argument(
        "--model-api-styles",
        nargs="*",
        default=[],
        help="按模型覆盖请求风格，格式 model=style，例如 anthropic/claude-opus-4.7=anthropic",
    )
    

    parser.add_argument(
        "--models",
        nargs="+",
        required=False,
        default=[
            # "xiaomi/mimo-v2.5",
            "google/gemini-3.5-flash",
            # "openai/gpt-5.5",
        ],
        help="待测模型列表",
    )
    parser.add_argument("--evaluator-model", default="", help="七维开放题评审模型；score/all 时必填")
    parser.add_argument(
        "--judge-prompt", type=Path, default=seven_dim_judge.DEFAULT_PROMPT,
        help="七维 Judge prompt；默认与 paper/scripts/judge_openended_7d.py 完全一致",
    )

    parser.add_argument("--concurrency", type=int, default=5, help="单模型并发请求数")
    parser.add_argument("--model-concurrency", type=int, default=2, help="同时评测的模型数量")
    parser.add_argument("--request-timeout", type=int, default=60, help="单次请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="最大重试次数")
    parser.add_argument("--objective-max-tokens", type=int, default=1024, help="客观题生成最大 token 数")
    parser.add_argument("--short-answer-max-tokens", type=int, default=2048, help="简答题生成最大 token 数")
    parser.add_argument("--evaluator-max-tokens", type=int, default=2048, help="简答题评审最大 token 数")
    parser.add_argument(
        "--short-answer-variants", nargs="+", choices=("noprompt", "localinfo"), default=["noprompt"],
        help="开放简答题生成条件；默认仅 noprompt，可显式加入 localinfo",
    )
    parser.add_argument(
        "--allow-text-fallback", action="store_true",
        help="显式允许视觉请求失败后退化为纯文本；正式 benchmark 不应开启",
    )

    parser.add_argument(
        "--source-image-prefix",
        default="/root/autodl-tmp/data/finalData/merge_data/",
        help="数据集中旧图片路径前缀",
    )
    parser.add_argument("--image-root", default="./merge_data/merge_data", help="本地图片根目录，用于路径映射")

    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条，0 表示全量")
    parser.add_argument("--short-bad-threshold", type=int, default=6, help="简答题低分阈值")
    return parser.parse_args()


def remap_sample_image(sample: Sample, source_image_prefix: str, image_root: str) -> None:
    source_paths = sample.image_paths or ([sample.image_path] if sample.image_path else [])
    sample.image_files = [
        remap_image_path(path, source_image_prefix, image_root) for path in source_paths
    ]
    sample.image_file = sample.image_files[0] if sample.image_files else ""


def remap_samples_images(samples: List[Sample], source_image_prefix: str, image_root: str) -> int:
    missing_count = 0
    for s in samples:
        remap_sample_image(s, source_image_prefix, image_root)
        source_paths = s.image_paths or ([s.image_path] if s.image_path else [])
        missing_count += sum(1 for index, _ in enumerate(source_paths)
                             if index >= len(s.image_files) or not s.image_files[index])
    return missing_count


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def sample_has_valid_image(sample: Sample) -> bool:
    return bool(resolve_sample_image_urls(sample))


def sample_has_all_images(sample: Sample) -> bool:
    expected = len(sample.image_paths or ([sample.image_path] if sample.image_path else []))
    return expected > 0 and len(resolve_sample_image_urls(sample)) == expected


def _valid_image_url(image_source: str) -> str:
    if not image_source:
        return ""
    if re.match(r"^https?://", image_source, flags=re.IGNORECASE):
        return image_source
    path = Path(image_source)
    if path.exists() and path.is_file():
        return image_to_data_url(str(path))
    return ""


def resolve_sample_image_urls(sample: Sample) -> List[str]:
    source_paths = sample.image_paths or ([sample.image_path] if sample.image_path else [])
    mapped_paths = sample.image_files or ([sample.image_file] if sample.image_file else [])
    urls: List[str] = []
    for index, source_path in enumerate(source_paths):
        mapped = mapped_paths[index] if index < len(mapped_paths) else ""
        url = _valid_image_url(mapped) or _valid_image_url(source_path)
        if url:
            urls.append(url)
    return urls


def resolve_sample_image_url(sample: Sample) -> str:
    urls = resolve_sample_image_urls(sample)
    return urls[0] if urls else ""


def sample_tag_paths(sample: Sample) -> List[Path]:
    paths: List[Path] = []
    for image_file in sample.image_files or ([sample.image_file] if sample.image_file else []):
        if image_file:
            paths.append(Path(image_file).parent / "tag.json")
    return paths


def build_local_info_text(sample: Sample) -> str:
    blocks: List[str] = []
    for index, tag_path in enumerate(sample_tag_paths(sample)):
        if not tag_path.exists():
            continue
        try:
            tag = json.loads(tag_path.read_text(encoding="utf-8"))
            local_info = tag.get("localInfo") or []
        except (OSError, ValueError, TypeError):
            continue
        if local_info:
            blocks.append(
                f"画{chr(65 + index)}的局部标注：" +
                json.dumps(local_info, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n\n".join(blocks)


def _legacy_sample_has_valid_image(sample: Sample) -> bool:
    """Retained only for compatibility with old callers; new code requires all images."""
    for image_source in (sample.image_file, sample.image_path):
        if not image_source:
            continue
        if re.match(r"^https?://", image_source, flags=re.IGNORECASE):
            return True
        path = Path(image_source)
        if path.exists() and path.is_file():
            return True
    return False


async def async_main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    log("Fatal API circuit breakers enabled: 401 invalid token; 402 insufficient credits")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_style_overrides = parse_model_style_overrides(args.model_api_styles)
    model_configs = build_model_configs(args.models, args.api_style, model_style_overrides)
    if args.action == "summarize":
        for model_cfg in model_configs:
            model_safe = safe_model_name(model_cfg.name)
            report_path = out_dir / f"report_{model_safe}.json"
            rows = load_report_rows(report_path)
            if not rows:
                log(f"跳过无报告模型: {model_cfg.name}")
                continue
            save_text(out_dir / f"summary_{model_safe}.md", build_markdown_report(model_cfg.name, rows))
        return

    log("加载数据集...")
    samples = parse_samples(data_dir, args.input_files)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("未解析到任何有效样本，请检查输入文件与字段格式")
    missing_image_count = remap_samples_images(samples, args.source_image_prefix, args.image_root)
    log(f"样本总数: {len(samples)}；缺失图片数: {missing_image_count}")

    normalized_api_base = normalize_api_base(args.api_base)
    client: Optional[AsyncVisionClient] = None
    if args.action in {"generate", "all"}:
        if not args.api_key:
            raise RuntimeError("生成阶段需要 --api-key 或 OPENAI_API_KEY")
        client = AsyncVisionClient(
            base_url=normalized_api_base,
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            allow_text_fallback=args.allow_text_fallback,
        )

        async def generate_one_model(model_cfg: ModelConfig) -> None:
            model_safe = safe_model_name(model_cfg.name)
            report_path = out_dir / f"report_{model_safe}.json"
            existing_rows = load_report_rows(report_path)
            rows_by_id = {str(row.get("id")): row for row in existing_rows if row.get("id")}

            def completed(row_id: str) -> bool:
                row = rows_by_id.get(row_id, {})
                return bool(str(row.get("raw_model_output") or "").strip()) and not row.get("was_truncated")

            pending: List[Sample] = []
            for sample in samples:
                expected = (
                    [sample.sample_id + f"__{variant}" for variant in args.short_answer_variants]
                    if sample.sample_type == "ShortAnswer" else [sample.sample_id]
                )
                if not all(completed(row_id) for row_id in expected):
                    pending.append(sample)
            log(f"[{model_cfg.name}] generation pending {len(pending)}/{len(samples)}")

            def write_progress(row: Dict[str, Any], finished: int, total: int) -> None:
                simple = build_simple_report_rows([row])[0]
                rows_by_id[str(simple["id"])] = simple
                if finished == total or finished % 10 == 0:
                    ordered = [rows_by_id[key] for key in sorted(rows_by_id)]
                    save_json_atomic(report_path, ordered)

            if pending:
                try:
                    generated = await evaluate_model(
                        model_config=model_cfg,
                        samples=pending,
                        client=client,
                        concurrency=args.concurrency,
                        objective_max_tokens=args.objective_max_tokens,
                        short_answer_max_tokens=args.short_answer_max_tokens,
                        short_answer_variants=args.short_answer_variants,
                        on_result=write_progress,
                    )
                except FatalAPIError:
                    ordered = [rows_by_id[key] for key in sorted(rows_by_id)]
                    save_json_atomic(report_path, ordered)
                    raise
                for row in build_simple_report_rows(generated):
                    rows_by_id[str(row["id"])] = row
                save_json_atomic(report_path, [rows_by_id[key] for key in sorted(rows_by_id)])
            log(f"[{model_cfg.name}] generation saved: {report_path}")

        model_sem = asyncio.Semaphore(max(1, args.model_concurrency))
        async def limited_generate(model_cfg: ModelConfig) -> None:
            async with model_sem:
                await generate_one_model(model_cfg)
        model_tasks = [asyncio.create_task(limited_generate(config)) for config in model_configs]
        try:
            await asyncio.gather(*model_tasks)
        except FatalAPIError:
            for task in model_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*model_tasks, return_exceptions=True)
            raise

    if args.action in {"score", "all"}:
        if not args.evaluator_model:
            raise RuntimeError("score/all 阶段必须指定 --evaluator-model")
        judge_key = args.judge_api_key or args.api_key
        judge_base = normalize_api_base(args.judge_api_base or args.api_base)
        if not judge_key:
            raise RuntimeError("评分阶段需要 --judge-api-key、ORACLE_JUDGE_API_KEY 或 --api-key")
        evaluator_style = model_style_overrides.get(args.evaluator_model)
        if not evaluator_style:
            evaluator_style = infer_api_style_from_model(args.evaluator_model) if args.api_style == "auto" else args.api_style
        evaluator_config = ModelConfig(args.evaluator_model, evaluator_style)
        judge_client = AsyncVisionClient(
            base_url=judge_base,
            api_key=judge_key,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            allow_text_fallback=False,
        )
        sample_by_id = {sample.sample_id: sample for sample in samples}
        for model_cfg in model_configs:
            model_safe = safe_model_name(model_cfg.name)
            report_path = out_dir / f"report_{model_safe}.json"
            rows = load_report_rows(report_path)
            if not rows:
                raise FileNotFoundError(f"先运行 generate；缺少 {report_path}")
            # Enrich older reports with complete image provenance before scoring.
            for row in rows:
                base_id = str(row.get("base_id") or str(row.get("id") or "").split("__", 1)[0])
                sample = sample_by_id.get(base_id)
                if sample:
                    row.update({
                        "base_id": base_id,
                        "image_path": sample.image_path,
                        "image_file": sample.image_file,
                        "image_paths": sample.image_paths,
                        "image_files": sample.image_files,
                    })
            try:
                rows = await score_report_rows(
                    tested_model=model_cfg.name,
                    rows=rows,
                    evaluator_config=evaluator_config,
                    client=judge_client,
                    concurrency=args.concurrency,
                    evaluator_max_tokens=args.evaluator_max_tokens,
                    prompt_path=args.judge_prompt,
                )
            except FatalAPIError:
                save_json_atomic(report_path, build_simple_report_rows(rows))
                raise
            save_json_atomic(report_path, build_simple_report_rows(rows))
            log(f"[{model_cfg.name}] scoring saved: {report_path}")

    if args.action in {"score", "all"}:
        for model_cfg in model_configs:
            model_safe = safe_model_name(model_cfg.name)
            report_path = out_dir / f"report_{model_safe}.json"
            rows = load_report_rows(report_path)
            save_text(out_dir / f"summary_{model_safe}.md", build_markdown_report(model_cfg.name, rows))


def main() -> None:
    args = parse_args()
    if args.action == "summarize":
        out_dir = Path(args.output_dir)
        model_style_overrides = parse_model_style_overrides(args.model_api_styles)
        model_configs = build_model_configs(args.models, args.api_style, model_style_overrides)
        for model_cfg in model_configs:
            model_safe = safe_model_name(model_cfg.name)
            report_path = out_dir / f"report_{model_safe}.json"
            rows = load_report_rows(report_path)
            if not rows:
                log(f"跳过无报告模型: {model_cfg.name}")
                continue
            summary_path = out_dir / f"summary_{model_safe}.md"
            save_text(summary_path, build_markdown_report(model_cfg.name, rows))
            log(f"已输出: {summary_path}")
        return
    try:
        asyncio.run(async_main(args))
    except FatalAPIError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
