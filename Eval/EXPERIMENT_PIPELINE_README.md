# TCPsBenchmark 实验运行手册

本文档统一说明以下程序的用途、调用顺序、环境变量、输出文件和断点续跑方式：

- `run_benchmark.py`：T1–T8 全量回答生成、统一评分与汇总；
- `judge_openended_7d.py`：七维开放题 Judge；

## 1. 程序关系

```text
全量 benchmark
run_benchmark.py generate
        ↓
report_<model>.json（仅回答）
        ↓
run_benchmark.py score
        ├─ Single/Judge/Multi：确定性准确率
        ├─ Fill：原始 exact match、precision、recall、F1
        └─ ShortAnswer：统一七维 Judge
        ↓
run_benchmark.py summarize
```

## 2. 题型与评分方式

| 类型 | 评分方式 | 主要指标 |
| --- | --- | --- |
| Single | 与参考答案确定性比较 | Accuracy |
| Judge | 与参考答案确定性比较 | Accuracy |
| Multi | 选项归一化后比较 | Accuracy |
| Fill | 原始字符/答案匹配方法 | Exact match、macro precision、macro recall、macro F1 |
| ShortAnswer | 七维 LLM Judge | 七维均分与 0–100 总分 |

`ShortAnswer_artistic_methods` 会归一化为 `ShortAnswer`。Fill 不调用 LLM Judge。

七维 Judge 包含：

1. `semantic_similarity`；
2. `factual_consistency`；
3. `question_alignment`；
4. `technique_accuracy`；
5. `structure_clarity`；
6. `coverage_depth`；
7. `hallucination_control`。

每维为 0–10 整数。程序计算七维均值，并映射到 0–100。

## 3. API 环境变量

### 3.1 全量 benchmark 生成

`run_benchmark.py generate` 默认读取 `OPENAI_API_KEY`：

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new(
    '',
    (Read-Host 'Generation API key' -AsSecureString)
).Password.Trim()
```

### 3.2 Judge 评分

`run_benchmark.py score`、七维 Judge 和旧两维重评分优先读取：

```powershell
$env:ORACLE_JUDGE_API_BASE = 'https://YOUR_JUDGE_GATEWAY/v1'
$env:ORACLE_JUDGE_API_KEY = [System.Net.NetworkCredential]::new(
    '',
    (Read-Host 'Judge API key' -AsSecureString)
).Password.Trim()
```

Judge 变量为空时，部分脚本回退到 `ORACLE_API_BASE` 和 `ORACLE_API_KEY`。

密钥只保存在当前 PowerShell 进程。不要把真实密钥写入脚本、README、日志或命令历史。

## 4. 全量 T1–T8 benchmark

程序：`benchmark/run_benchmark.py`

推荐使用独立的新输出目录：

```text
benchmark/benchmark_outputs_v2
```

### 4.1 第一步：只生成回答

```powershell
py -3.13 benchmark/run_benchmark.py generate `
  --data-dir benchmark `
  --image-root benchmark/merge_data/merge_data `
  --output-dir benchmark/benchmark_outputs_v2 `
  --api-base https://YOUR_GENERATION_GATEWAY/v1 `
  --models MODEL_A MODEL_B MODEL_C `
  --short-answer-variants noprompt `
  --concurrency 4 `
  --model-concurrency 2
```

这一阶段只调用被测模型，不调用 Judge。

T7/T8 默认只运行 `noprompt`。显式加入局部标注条件时使用：

```powershell
--short-answer-variants noprompt localinfo
```

正式主实验保持：

```powershell
--short-answer-variants noprompt
```

T8 会按照题目中的 A/B/C 顺序发送全部 2–3 幅图片。缺少任一图片时记录错误。正式实验不启用 `--allow-text-fallback`。

### 4.2 第二步：统一评分

```powershell
py -3.13 benchmark/run_benchmark.py score `
  --data-dir benchmark `
  --image-root benchmark/merge_data/merge_data `
  --output-dir benchmark/benchmark_outputs_v2 `
  --models MODEL_A MODEL_B MODEL_C `
  --evaluator-model YOUR_VALIDATED_JUDGE_MODEL `
  --short-answer-variants noprompt `
  --concurrency 4
```

该阶段执行：

- Single/Judge/Multi 的准确率计算；
- Fill 的原始匹配指标；
- ShortAnswer 的七维 Judge。

七维评分直接复用：

```text
paper/scripts/openended_7d_judge_prompt.txt
paper/scripts/judge_openended_7d.py
```

Judge 接收问题、参考答案、候选回答以及所有画作的 metadata 和结构化标注。Judge 不接收被测模型名称。

### 4.3 第三步：离线汇总

```powershell
py -3.13 benchmark/run_benchmark.py summarize `
  --output-dir benchmark/benchmark_outputs_v2 `
  --models MODEL_A MODEL_B MODEL_C
```

输出：

```text
benchmark/benchmark_outputs_v2/
├── report_MODEL_A.json
├── summary_MODEL_A.md
├── report_MODEL_B.json
└── summary_MODEL_B.md
```

### 4.4 小规模试跑

```powershell
py -3.13 benchmark/run_benchmark.py generate `
  --data-dir benchmark `
  --image-root benchmark/merge_data/merge_data `
  --output-dir benchmark/benchmark_outputs_pilot `
  --api-base https://YOUR_GENERATION_GATEWAY/v1 `
  --models MODEL_A `
  --short-answer-variants noprompt `
  --limit 10 `
  --concurrency 1
```

`--limit 10` 表示合并数据后的前10条，不是每个任务各10条。

### 4.5 断点续跑

`generate` 根据记录 ID 和完整回答判断是否完成。成功回答会被跳过，缺失、空回答和截断回答会重试。

`score` 独立于生成阶段。Judge 失败后重新执行 `score` 即可，不需要重新生成模型回答。

