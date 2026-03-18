## IRT RAG runbook

This folder contains a simple RAG pipeline:

- **Step 1**: Enrich Excel (`Summary` + `Comments`) → add `Solution`, `Resolution Status`/`status`, `References`
- **Step 2**: Embed + store in local Qdrant (`qdrant_storage/`)
- **Step 3**: Query (embed question → retrieve similar issues → generate answer)
- **Step 4**: Slack bot (optional)

### Prereqs

- Set your API key in the shell:

```bash
export OPENAI_API_KEY="your_key"
```

### Step 1 — Enrich Excel

Put your input file in this folder as `bugs_with_comments.xlsx` (must include columns `Summary` and `Comments`).

Run **v1** (simple, slower):

```bash
conda run -n bug_tracker python irt_enrich_excel.py
```

Run **v2** (batched, faster):

```bash
conda run -n bug_tracker python irt_enrich_excel_v2.py
```

Optional env vars:
- `OPENAI_MODEL_EXTRACT` (default `gpt-4.1-mini`)
- `ENRICH_BATCH_SIZE` (default `15`)
- `ENRICH_MAX_RETRIES` (default `4`)
- `REQUEST_SLEEP_SECONDS` (default `0.2`)

### Step 2 — Build the knowledge base (Qdrant local)

v1:

```bash
conda run -n bug_tracker python irt_rag_build_knowledge_base.py
```

v2 (incremental upserts; add `--recreate` to rebuild from scratch):

```bash
conda run -n bug_tracker python irt_rag_build_knowledge_base_v2.py
conda run -n bug_tracker python irt_rag_build_knowledge_base_v2.py --recreate
```

### Step 3 — Query

v1:

```bash
conda run -n bug_tracker python irt_rag_query.py --query "v2 dataset failed"
```

v2:

```bash
conda run -n bug_tracker python irt_rag_query_v2.py --query "v2 dataset failed this error ==> {error}" --min-score 0.30
```

### Step 4 — Slack bot (optional)

Set:
- `SLACK_BOT_TOKEN`
- `SLACK_APP_TOKEN`
- `OPENAI_API_KEY`

Then run:

```bash
conda run -n bug_tracker python irt_rag_slack_bot.py
```

