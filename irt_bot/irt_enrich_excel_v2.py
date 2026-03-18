"""
STEP 1 (v2): Enrich Excel with Solution + Final Status (+ References) (faster + safer)

Key improvements vs v1:
- Batch multiple tickets per model call (fewer round trips)
- Stronger JSON-only response contract + robust parsing
- Resume-friendly: skips rows that already have Solution filled
- Configurable batch size, sleep, retries, and model

Input : bugs_with_comments.xlsx  (needs columns: Summary, Details, Comments)
Output: bugs_enriched.xlsx       (adds: Solution, Final Status, References; also keeps Resolution Status/status for compatibility)

Run:
  python irt_enrich_excel_v2.py
  python irt_enrich_excel_v2.py --input bugs_with_comments.xlsx --output bugs_enriched.xlsx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _clean(text: Any) -> str:
    if not isinstance(text, str) or text in ("nan", "None", ""):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Solution" not in df.columns:
        df["Solution"] = ""
    if "Final Status" not in df.columns:
        df["Final Status"] = ""
    if "Resolution Status" not in df.columns:
        df["Resolution Status"] = ""
    if "status" not in df.columns:
        df["status"] = ""
    if "References" not in df.columns:
        df["References"] = ""
    return df


def _parse_json_loose(text: str) -> Any:
    """
    Prefer strict JSON; fallback to first JSON array/object found.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in output")
    return json.loads(match.group(1))


def _build_batch_prompt(items: list[dict[str, Any]]) -> str:
    # Keep the contract very explicit; model must return JSON array only.
    blob = []
    for it in items:
        blob.append(
            {
                "row_id": it["row_id"],
                "summary": it["summary"],
                "details": it["details"],
                "comments": it["comments"],
            }
        )
    return f"""You are an IRT (Incident Response Team) support analyst.
For each ticket below, analyze Summary (issue title), Details (issue description), and Comments (what we tried / workaround / fix).

Your goal:
- Write a clear Solution description:
  - If a permanent fix is described, say what the fix was.
  - If only a workaround is described, say what the workaround was.
  - If no fix/workaround is described, use "No solution documented."
- Set Final Status based on the evidence:
  - "Fixed"       = permanent fix confirmed
  - "Workaround"  = workaround given but not a confirmed permanent fix
  - "Unresolved"  = no solution / still failing / unknown outcome
  - "Rejected"    = ticket rejected / not a bug / wontfix (only if clearly stated)

Return ONLY valid JSON (no markdown, no extra text).
Return a JSON array with one object per input item:
[
  {{
    "row_id": <same row_id as input>,
    "solution": "<string, or 'No solution documented.'>",
    "final_status": "Fixed" | "Workaround" | "Unresolved" | "Rejected",
    "references": "<comma-separated URLs/IDs or 'None'>"
  }}
]

Tickets:
{json.dumps(blob, ensure_ascii=False)}
"""


def enrich_dataframe(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    batch_size: int,
    max_retries: int,
    request_sleep_seconds: float,
) -> pd.DataFrame:
    df = _ensure_cols(df)

    if "Summary" not in df.columns or "Comments" not in df.columns or "Details" not in df.columns:
        raise ValueError("Input file must contain 'Summary', 'Details', and 'Comments' columns")

    # Build list of rows needing enrichment
    todo: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        existing = str(row.get("Solution", "")).strip()
        if existing and existing.lower() not in ("nan", "none"):
            continue

        summary = str(row.get("Summary", "")).strip()
        details = str(row.get("Details", "")).strip()
        comments = str(row.get("Comments", "")).strip()

        # If no data, fill defaults
        if not summary or (not details and not comments):
            df.at[idx, "Solution"] = "No solution documented."
            df.at[idx, "Final Status"] = "Unresolved"
            df.at[idx, "Resolution Status"] = "Unresolved"
            df.at[idx, "status"] = "Unresolved"
            df.at[idx, "References"] = "None"
            continue

        todo.append(
            {
                "df_idx": idx,
                "row_id": int(idx),  # stable within file
                "summary": summary[:500],
                "details": _clean(details)[:1200],
                "comments": _clean(comments)[:1600],
            }
        )

    total = len(todo)
    if total == 0:
        return df

    print(f"  Rows needing enrichment: {total}")

    processed = 0
    for start in range(0, total, batch_size):
        batch = todo[start : start + batch_size]
        prompt = _build_batch_prompt(batch)

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=1800,
                )
                data = _parse_json_loose(resp.output_text or "")
                if not isinstance(data, list):
                    raise ValueError("Expected a JSON array")

                by_row_id = {int(o["row_id"]): o for o in data if isinstance(o, dict) and "row_id" in o}
                if len(by_row_id) == 0:
                    raise ValueError("No usable items returned")

                for it in batch:
                    out = by_row_id.get(int(it["row_id"]))
                    if not out:
                        continue
                    idx = it["df_idx"]
                    solution = str(out.get("solution", "")).strip() or "No solution documented."
                    final_status = str(out.get("final_status", "Unresolved")).strip() or "Unresolved"
                    refs = str(out.get("references", "None")).strip() or "None"

                    df.at[idx, "Solution"] = solution
                    df.at[idx, "Final Status"] = final_status
                    # Back-compat columns used by KB/query scripts
                    df.at[idx, "Resolution Status"] = final_status
                    df.at[idx, "status"] = final_status
                    df.at[idx, "References"] = refs

                break
            except Exception as e:
                last_err = e
                sleep_s = min(20.0, 2.0 * (1.6**attempt))
                time.sleep(sleep_s)
        else:
            # mark batch as failed but continue
            print(f"  ⚠️ Batch failed after retries: {last_err}")
            for it in batch:
                idx = it["df_idx"]
                df.at[idx, "Solution"] = "Unable to extract solution."
                df.at[idx, "Final Status"] = "Unresolved"
                df.at[idx, "Resolution Status"] = "Unresolved"
                df.at[idx, "status"] = "Unresolved"
                df.at[idx, "References"] = "None"

        processed += len(batch)
        if processed % (batch_size * 5) == 0 or processed == len(batch):
            print(f"  Progress: {processed}/{total}")

        # small pause to reduce rate-limit spikes
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bugs_with_comments.xlsx")
    parser.add_argument("--output", default="bugs_enriched.xlsx")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ENRICH_BATCH_SIZE", "15")))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL_EXTRACT", "gpt-4.1-mini"))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("ENRICH_MAX_RETRIES", "4")))
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.environ.get("REQUEST_SLEEP_SECONDS", "0.2")),
        help="Sleep between batches (seconds)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY missing (export it or set it in .env)")

    if not os.path.exists(args.input):
        raise SystemExit(f"❌ Input file not found: {args.input}")

    print()
    print("=" * 60)
    print("  STEP 1 (v2) — Enrich Excel with Solution + Final Status (+ References)")
    print("=" * 60)
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model : {args.model}")
    print(f"  Batch : {args.batch_size}")

    df = pd.read_excel(args.input)
    print(f"  Loaded: {len(df)} rows")

    client = OpenAI()
    df = enrich_dataframe(
        df=df,
        client=client,
        model=args.model,
        batch_size=max(1, args.batch_size),
        max_retries=max(1, args.retries),
        request_sleep_seconds=max(0.0, args.sleep),
    )

    df.to_excel(args.output, index=False)
    print()
    print(f"  ✅ Saved → {args.output}")


if __name__ == "__main__":
    main()

