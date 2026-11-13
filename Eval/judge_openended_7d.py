"""Rejudge existing T7/T8 noprompt answers with seven dimensions and painting evidence.

prepare and summarize are offline; score makes paid API calls only with --execute.
No candidate answer is regenerated, and no image is sent to the judge.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rescore_main_t7t8 import PROJECT, choose_candidate, normalized, oracle, report_item_id, source_rows

DIMENSIONS = (
    "semantic_similarity", "factual_consistency", "question_alignment",
    "technique_accuracy", "structure_clarity", "coverage_depth", "hallucination_control",
)
RUBRIC_ID = "metadata_annotation_seven_dimension_v2"
INPUT_SCHEMA = "question_reference_metadata_annotations_candidate_v2"
DEFAULT_PROMPT = Path(__file__).with_name("openended_7d_judge_prompt.txt")
DEFAULT_OUTPUT = PROJECT / "paper" / "outputs" / "openended_7d_annotations_v2"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def painting_evidence(tag_path: str, label: str, painting_id: str) -> dict:
    """Preserve labeled global/local tags and local XY boxes; omit narrative description."""
    tag = json.loads(Path(tag_path).read_text(encoding="utf-8"))
    show_key = str(tag.get("showKey") or "").split("/", 3)
    fields = ("museum", "dynasty", "artist", "title")
    metadata = {name: show_key[i].strip() if i < len(show_key) else "" for i, name in enumerate(fields)}
    global_tags = defaultdict(list)
    for entry in tag.get("globalTags") or []:
        if isinstance(entry, dict):
            global_tags[str(entry.get("typeName") or "")].extend(oracle.text_tags(entry.get("tags")))
    for key, tag_name in (("theme", "主题"), ("style", "类型"), ("genre", "题材"),
                          ("material", "材质"), ("format", "形制")):
        metadata[key] = list(dict.fromkeys(global_tags[tag_name]))
    global_annotations = [{"type_name": kind, "tags": list(dict.fromkeys(values))}
                          for kind, values in global_tags.items() if values]
    local_annotations = []
    for region in tag.get("localInfo") or []:
        if not isinstance(region, dict):
            continue
        entries = []
        for entry in region.get("tags") or []:
            if isinstance(entry, dict):
                values = oracle.text_tags(entry.get("tags"))
                kind = str(entry.get("typeName") or "未分类")
            elif isinstance(entry, str):
                values = oracle.text_tags(entry)
                kind = "未分类"
            else:
                continue
            if values:
                entries.append({"type_name": kind, "tags": values})
        if entries:
            local_annotations.append({"area_id": region.get("areaId"), "bbox_xy": oracle.bbox(region),
                                      "tags": entries})
    return {"label": label, "painting_id": painting_id, "metadata": metadata,
            "global_annotations": global_annotations, "local_annotations": local_annotations}


def oracle_generation_candidates(args, source: dict) -> tuple[list[dict], Counter]:
    generations_path = args.answers_file.resolve()
    manifest_path = (args.oracle_manifest or generations_path.with_name("manifest.jsonl")).resolve()
    experiment_items = {x["id"]: x for x in oracle.read_jsonl(manifest_path)}
    if not experiment_items:
        raise ValueError(f"Oracle experiment manifest missing or empty: {manifest_path}")
    latest = {}
    audit = Counter()
    for entry in oracle.read_jsonl(generations_path):
        item_id = str(entry.get("item_id") or "")
        condition = str(entry.get("condition") or "")
        if not item_id.startswith(("T7_", "T8_")) or not condition:
            continue
        model = str(entry.get("model") or "")
        task = int(item_id[1])
        group = (model, task)
        audit[(*group, "generation_record")] += 1
        if entry.get("status") != "ok" or entry.get("finish_reason") in {"length", "max_tokens"}:
            audit[(*group, "failed_or_truncated")] += 1
            continue
        if not isinstance(entry.get("text"), str) or len(entry["text"].strip()) <= args.min_answer_chars:
            audit[(*group, "short_answer")] += 1
            continue
        latest[(model, item_id, condition)] = entry
    selected = []
    for (model, item_id, condition), entry in sorted(latest.items()):
        task = int(item_id[1])
        group = (model, task)
        row, problem = source.get(item_id, (None, "source_id_missing"))
        if problem:
            audit[(*group, problem)] += 1
            continue
        experiment_row = experiment_items.get(item_id)
        if not experiment_row:
            audit[(*group, "oracle_manifest_id_missing")] += 1
            continue
        if (normalized(experiment_row.get("question")) != normalized(row["question"]) or
                normalized(experiment_row.get("reference")) != normalized(row["reference"]) or
                experiment_row.get("painting_ids") != row["painting_ids"]):
            audit[(*group, "oracle_manifest_mismatch")] += 1
            continue
        try:
            paintings = [painting_evidence(path, chr(65 + i), painting_id)
                         for i, (path, painting_id) in enumerate(zip(row["tag_paths"], row["painting_ids"]))]
        except (OSError, ValueError, TypeError):
            audit[(*group, "invalid_metadata")] += 1
            continue
        answer = entry["text"].strip()
        selected.append({"model": model, "task": task, "item_id": item_id,
                         "condition": condition, "source_kind": "oracle_generation",
                         "report_file": str(generations_path), "report_id": f"{model}/{item_id}/{condition}",
                         "answer_source": "oracle_generation_text", "candidate": answer,
                         "candidate_chars": len(answer), "question": row["question"],
                         "reference": row["reference"], "painting_evidence": paintings,
                         "old_score_0_10": None})
        audit[(*group, "eligible")] += 1
    return selected, audit


def prepare(args) -> None:
    if (args.output / "judgments.jsonl").exists():
        raise ValueError("Judgments already exist; use a fresh --output for a new manifest")
    source = source_rows(args.data_dir)
    if args.answers_file and args.reports:
        raise ValueError("Use either --answers-file or --reports, not both")
    if args.answers_file and args.answers_file.suffix.lower() == ".jsonl":
        selected, audit = oracle_generation_candidates(args, source)
        finish_prepare(args, selected, audit)
        return
    if args.answers_file and args.answers_file.suffix.lower() != ".json":
        raise ValueError("--answers-file must be a report .json or oracle generations .jsonl")
    reports = args.reports or ([args.answers_file] if args.answers_file else sorted(args.reports_dir.glob("report_*.json")))
    if not reports:
        raise ValueError("No report_*.json files found")
    selected, audit = [], Counter()
    for report_path in reports:
        report_path = report_path.resolve()
        model = report_path.stem.removeprefix("report_")
        for entry in json.loads(report_path.read_text(encoding="utf-8-sig")):
            item_id = report_item_id(entry)
            if not item_id or not item_id.startswith(("T7_", "T8_")):
                continue
            task = int(item_id[1])
            group = (model, task)
            audit[(*group, "report_noprompt")] += 1
            row, problem = source.get(item_id, (None, "source_id_missing"))
            if problem:
                audit[(*group, problem)] += 1
                continue
            if normalized(entry.get("question")) != normalized(row["question"]):
                audit[(*group, "question_mismatch")] += 1
                continue
            if normalized(entry.get("reference_answer")) != normalized(row["reference"]):
                audit[(*group, "reference_mismatch")] += 1
                continue
            answer, answer_source = choose_candidate(entry, args.allow_extracted, args.min_answer_chars)
            if answer is None:
                audit[(*group, answer_source)] += 1
                continue
            try:
                paintings = [painting_evidence(path, chr(65 + i), painting_id)
                             for i, (path, painting_id) in enumerate(zip(row["tag_paths"], row["painting_ids"]))]
            except (OSError, ValueError, TypeError):
                audit[(*group, "invalid_metadata")] += 1
                continue
            old_score = entry.get("evaluator_score")
            selected.append({"model": model, "task": task, "item_id": item_id,
                             "condition": "noprompt", "source_kind": "benchmark_report",
                             "report_file": str(report_path), "report_id": entry["id"],
                             "answer_source": answer_source, "candidate": answer,
                             "candidate_chars": len(answer), "question": row["question"],
                             "reference": row["reference"], "painting_evidence": paintings,
                             "old_score_0_10": old_score if isinstance(old_score, (int, float)) and not isinstance(old_score, bool) else None})
            audit[(*group, "eligible")] += 1
    finish_prepare(args, selected, audit)


def finish_prepare(args, selected: list[dict], audit: Counter) -> None:
    if not selected:
        raise ValueError("No eligible T7/T8 answers; inspect report provenance and --allow-extracted")
    if args.sample_per_task:
        models = {item["model"] for item in selected}
        rng = random.Random(args.seed)
        chosen = {}
        for task in (7, 8):
            by_model = [{item["item_id"] for item in selected if item["model"] == model and item["task"] == task}
                        for model in models]
            common = sorted(set.intersection(*by_model)) if by_model else []
            if not common:
                raise ValueError(f"T{task}: no common eligible question across models")
            chosen[task] = set(rng.sample(common, min(args.sample_per_task, len(common))))
        selected = [item for item in selected if item["item_id"] in chosen[item["task"]]]
    for item in selected:
        audit[(item["model"], item["task"], "prepared")] += 1
    selected.sort(key=lambda x: (x["model"], x["task"], x["item_id"], x["condition"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    with (args.output / "coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "task", "status", "count"))
        writer.writerows((*key, count) for key, count in sorted(audit.items()))
    print("prepared", len(selected), "manifest", args.output / "manifest.jsonl", flush=True)
    for key, count in sorted(audit.items()):
        if key[2] in {"report_noprompt", "prepared", "raw_output_missing", "short_answer", "missing_tag"}:
            print(*key, count, flush=True)


def judge_messages(item: dict, system_prompt: str) -> list[dict]:
    payload = {"input_schema": INPUT_SCHEMA, "question": item["question"],
               "reference_answer": item["reference"], "painting_evidence": item["painting_evidence"],
               "candidate_answer": item["candidate"]}
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def parse_judge(raw: str) -> tuple[dict, float, float, str]:
    try:
        obj = json.loads(raw)
    except ValueError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("Judge did not return JSON") from None
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or not isinstance(obj.get("dimensions"), dict):
        raise ValueError("Missing dimensions object")
    dimensions = obj["dimensions"]
    if set(dimensions) != set(DIMENSIONS):
        raise ValueError("Dimension keys differ from the seven-dimension rubric")
    for key in DIMENSIONS:
        value = dimensions[key]
        if not isinstance(value, dict) or isinstance(value.get("score"), bool) or not isinstance(value.get("score"), int):
            raise ValueError(f"Invalid integer score: {key}")
        if not 0 <= value["score"] <= 10:
            raise ValueError(f"Score out of range: {key}")
        if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
            raise ValueError(f"Missing rationale: {key}")
    comment = obj.get("overall_comment")
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Missing overall_comment")
    average = sum(dimensions[key]["score"] for key in DIMENSIONS) / len(DIMENSIONS)
    return dimensions, round(average, 4), round(10 * average, 2), comment


def score(args) -> None:
    if not args.execute:
        raise ValueError("Paid judge calls require --execute")
    api_key = os.environ.get("ORACLE_JUDGE_API_KEY") or os.environ.get("ORACLE_API_KEY")
    api_base = os.environ.get("ORACLE_JUDGE_API_BASE") or os.environ.get("ORACLE_API_BASE")
    if not api_key or not api_base:
        raise ValueError("Set ORACLE_JUDGE_API_BASE and ORACLE_JUDGE_API_KEY")
    items = oracle.read_jsonl(args.output / "manifest.jsonl")
    if not items:
        raise ValueError("Run prepare first; manifest is empty or missing")
    normalize_model = lambda name: re.sub(r"[^a-z0-9]", "", name.lower())
    if normalize_model(args.judge_model) in {normalize_model(x["model"]) for x in items}:
        raise ValueError("Judge model must differ from every tested model")
    system_prompt = args.prompt.read_text(encoding="utf-8")
    prompt_hash = sha256_text(system_prompt + "\n" + INPUT_SCHEMA)
    output = args.output / "judgments.jsonl"
    done = {(x.get("model"), x.get("item_id"), x.get("judge_model"),
             x.get("condition"), x.get("prompt_hash"), x.get("request_hash"))
            for x in oracle.read_jsonl(output) if x.get("status") == "ok"}

    def jobs():
        for item in items:
            messages = judge_messages(item, system_prompt)
            request_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            if (item["model"], item["item_id"], args.judge_model,
                    item["condition"], prompt_hash, request_hash) not in done:
                yield item, messages, request_hash

    def worker(job):
        item, messages, request_hash = job
        record = {"model": item["model"], "task": item["task"], "item_id": item["item_id"],
                  "condition": item["condition"], "source_kind": item["source_kind"],
                  "judge_model": args.judge_model, "rubric_id": RUBRIC_ID,
                  "prompt_hash": prompt_hash, "request_hash": request_hash,
                  "answer_source": item["answer_source"]}
        try:
            response = oracle.api_call(api_base, api_key, args.judge_model, messages,
                                       args.judge_max_tokens, args.timeout, args.retries, "openai")
            if response.get("finish_reason") in {"length", "max_tokens"}:
                raise ValueError("Judge output truncated")
            dimensions, score_0_10, score_0_100, comment = parse_judge(response["text"])
            record.update({"status": "ok", "dimensions": dimensions, "score_0_10": score_0_10,
                           "score_0_100": score_0_100, "overall_comment": comment,
                           "raw_judge_text": response["text"], "usage": response.get("usage", {})})
        except Exception as exc:
            record.update({"status": "error", "error": str(exc)[:500]})
        return record

    oracle.dispatch_jobs(jobs(), worker, output, args.concurrency, args.max_requests,
                         lambda x: f"judge {x['model']} {x['item_id']} {x['condition']}")


def summarize(args) -> None:
    items = {(x["model"], x["item_id"], x["condition"]): x
             for x in oracle.read_jsonl(args.output / "manifest.jsonl")}
    if not items:
        raise ValueError("Manifest missing; run prepare first")
    system_prompt = args.prompt.read_text(encoding="utf-8")
    prompt_hash = sha256_text(system_prompt + "\n" + INPUT_SCHEMA)
    judgments = {}
    for record in oracle.read_jsonl(args.output / "judgments.jsonl"):
        key = (record.get("model"), record.get("item_id"), record.get("condition"))
        if record.get("status") != "ok" or key not in items or record.get("rubric_id") != RUBRIC_ID:
            continue
        if record.get("judge_model") != args.judge_model or record.get("prompt_hash") != prompt_hash:
            continue
        messages = judge_messages(items[key], system_prompt)
        if record.get("request_hash") != sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)):
            continue
        judgments[key] = record
    groups = defaultdict(list)
    for key, judgment in judgments.items():
        item = items[key]
        groups[(item["model"], item["task"], item["condition"])].append((judgment, item))
    summary = []
    for (model, task, condition), rows in sorted(groups.items()):
        scores = [judgment["score_0_100"] for judgment, _ in rows]
        n = len(scores)
        rng = random.Random(args.seed)
        boot = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(args.bootstrap_reps))
        sources = Counter(item["answer_source"] for _, item in rows)
        result = {"model": model, "task": task, "condition": condition, "n": n,
                  "mean_0_100": round(sum(scores) / n, 2),
                  "ci95_low": round(boot[int(.025 * (len(boot) - 1))], 2),
                  "ci95_high": round(boot[int(.975 * (len(boot) - 1))], 2),
                  "raw_output_n": sources["raw_model_output"],
                  "extracted_unverified_n": sources["model_answer_unverified"]}
        for dimension in DIMENSIONS:
            result[dimension] = round(sum(judgment["dimensions"][dimension]["score"] for judgment, _ in rows) / n, 2)
        summary.append(result)
        print(result)
    fields = ["model", "task", "condition", "n", "mean_0_100", "ci95_low", "ci95_high",
              "raw_output_n", "extracted_unverified_n", *DIMENSIONS]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(summary)
    comparisons = (("B", "M"), ("M", "V"), ("V", "F"), ("F", "S"), ("B", "D"))
    paired_groups = defaultdict(list)
    for (model, item_id, condition), record in judgments.items():
        for before, after in comparisons:
            if condition != after:
                continue
            baseline = judgments.get((model, item_id, before))
            if baseline is not None:
                task = items[(model, item_id, after)]["task"]
                paired_groups[(model, task, f"{before}->{after}")].append(
                    record["score_0_100"] - baseline["score_0_100"])
    paired_summary = []
    for (model, task, comparison), deltas in sorted(paired_groups.items()):
        n = len(deltas)
        rng = random.Random(args.seed)
        boot = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(args.bootstrap_reps))
        paired_summary.append({"model": model, "task": task, "comparison": comparison, "n": n,
                               "mean_delta_0_100": round(sum(deltas) / n, 2),
                               "ci95_low": round(boot[int(.025 * (len(boot) - 1))], 2),
                               "ci95_high": round(boot[int(.975 * (len(boot) - 1))], 2)})
    with (args.output / "paired_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, ["model", "task", "comparison", "n", "mean_delta_0_100",
                                         "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(paired_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score", "summarize"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "benchmark")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT / "benchmark" / "benchmark_outputs")
    parser.add_argument("--reports", type=Path, nargs="*", help="Explicit report_*.json files; default all reports")
    parser.add_argument("--answers-file", type=Path,
                        help="A report .json or an oracle experiment generations.jsonl")
    parser.add_argument("--oracle-manifest", type=Path,
                        help="Question manifest for oracle generations; default manifest.jsonl beside answers file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--allow-extracted", action="store_true",
                        help="Opt in to report model_answer when raw_model_output is absent; provenance is unverified")
    parser.add_argument("--min-answer-chars", type=int, default=20)
    parser.add_argument("--sample-per-task", type=int, default=0,
                        help="Randomly choose this many shared eligible question IDs per task; 0 keeps all")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 100 or args.max_requests < 0 or args.min_answer_chars < 0 or args.sample_per_task < 0 or args.bootstrap_reps < 100:
        parser.error("Use concurrency 1-6, nonnegative request/sample/character limits, bootstrap reps >=100")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "score":
        score(args)
    else:
        summarize(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
