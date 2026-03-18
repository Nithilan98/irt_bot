"""
STEP 4: IRT Slack Bot — /irt slash command
Users type: /irt v2 dataset failed how to fix?
Bot replies with AI answer based on knowledge base.

Run: python step4_slack_bot.py
"""

import os, re, logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SLACK_BOT_TOKEN   = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN   = os.environ.get("SLACK_APP_TOKEN")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")
COLLECTION        = "irt_knowledge_base"
EMBED_MODEL       = "all-MiniLM-L6-v2"
STORAGE_DIR       = "./qdrant_storage"
TOP_K             = 5

app = App(token=SLACK_BOT_TOKEN)

# ── Load models once at startup ───────────────────────────────────────────────
print("⏳ Loading embedding model …")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model ready")

print("⏳ Connecting to Qdrant …")
qclient = QdrantClient(path=STORAGE_DIR)
count   = qclient.count(collection_name=COLLECTION).count
print(f"✅ Qdrant ready — {count} documents")

ai_client = OpenAI()


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search_kb(query: str) -> list:
    vec     = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = qclient.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=TOP_K,
        with_payload=True,
    )
    return [
        {
            "score"             : round(r.score, 3),
            "summary"           : r.payload.get("summary", ""),
            "solution"          : clean(r.payload.get("solution", "")),
            "resolution_status" : r.payload.get("resolution_status", ""),
            "severity"          : r.payload.get("severity", ""),
            "bug_category"      : r.payload.get("bug_category", ""),
            "environment"       : r.payload.get("environment", ""),
            "references"        : r.payload.get("references", "None"),
            "team"              : r.payload.get("team", ""),
        }
        for r in results
    ]


def generate_answer(query: str, hits: list) -> str:
    context = ""
    for i, h in enumerate(hits, 1):
        context += f"""
Issue #{i} (relevance: {h['score']})
  Summary    : {h['summary']}
  Category   : {h['bug_category']}
  Severity   : {h['severity']}
  Status     : {h['resolution_status']}
  Solution   : {h['solution']}
  References : {h['references']}
"""
    resp = ai_client.responses.create(
        model=os.environ.get("OPENAI_MODEL_SLACK", "gpt-4.1"),
        max_output_tokens=700,
        input=f"""
You are an IRT support assistant for ConverSight.

User asked: "{query}"

Relevant past issues:
{context}

Write a concise Slack response (max 300 words):
- State if this is a known issue
- Give the fix or workaround with bullet points
- Mention references if any
- Use *bold* for emphasis, not markdown headers
""",
    )
    return (resp.output_text or "").strip()


def build_blocks(query: str, answer: str, hits: list) -> list:
    icons = {"Fixed": "✅", "Partial": "⚠️", "Unresolved": "❌", "Rejected": "🚫"}

    hits_text = ""
    for h in hits[:3]:
        icon  = icons.get(h["resolution_status"], "❓")
        score = int(h["score"] * 100)
        bar   = "█" * (score // 10) + "░" * (10 - score // 10)
        hits_text += f"{icon} *{h['summary'][:65]}*\n"
        hits_text += f"   `{bar}` {score}% | _{h['resolution_status']}_ | {h['bug_category']}\n"
        if h["references"] not in ("None", "nan", ""):
            hits_text += f"   📎 {h['references'][:80]}\n"
        hits_text += "\n"

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔍 IRT Knowledge Base Response", "emoji": True}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Your question:*\n_{query}_"}
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*💡 Answer:*\n{answer}"}
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📋 Similar past issues:*\n\n{hits_text}"}
        },
        {
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"🤖 _IRT RAG Bot  •  {count} issues in KB  •  Top match: {int(hits[0]['score']*100)}%_"
            }]
        }
    ]


# ── /irt slash command ────────────────────────────────────────────────────────

@app.command("/irt")
def handle_irt(ack, respond, command):
    ack()   # must respond within 3 seconds

    query = command.get("text", "").strip()
    user  = command.get("user_id", "")

    if not query:
        respond(
            text="Please include a question. Example: `/irt v2 dataset failed`",
            response_type="ephemeral"
        )
        return

    log.info(f"/irt  user={user}  query={query[:80]}")

    try:
        hits = search_kb(query)

        if not hits or hits[0]["score"] < 0.3:
            respond(
                text=(
                    f"❌ *No similar issues found* for: _{query}_\n\n"
                    "This may be a new issue. Please create a ticket in the Bug Tracker."
                ),
                response_type="ephemeral"
            )
            return

        answer = generate_answer(query, hits)
        blocks = build_blocks(query, answer, hits)

        respond(
            text=f"IRT Knowledge Base: {query}",
            blocks=blocks,
            response_type="in_channel"   # visible to whole channel
        )

    except Exception as e:
        log.error(f"/irt error: {e}")
        respond(
            text=f"⚠️ Error: {str(e)[:100]}. Please try again.",
            response_type="ephemeral"
        )


# ── @mention handler ─────────────────────────────────────────────────────────

@app.event("app_mention")
def handle_mention(event, say):
    text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
    ts   = event.get("ts")

    if not text:
        say(
            text="Hi! Use `/irt your question` or mention me with a question.",
            thread_ts=ts
        )
        return

    try:
        hits = search_kb(text)
        if not hits or hits[0]["score"] < 0.3:
            say(
                text="❌ No similar issues found in knowledge base. This may be a new issue.",
                thread_ts=ts
            )
            return
        answer = generate_answer(text, hits)
        blocks = build_blocks(text, answer, hits)
        say(text=answer, blocks=blocks, thread_ts=ts)
    except Exception as e:
        say(text=f"⚠️ Error: {str(e)[:100]}", thread_ts=ts)


# ── start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key, val in [
        ("SLACK_BOT_TOKEN",   SLACK_BOT_TOKEN),
        ("SLACK_APP_TOKEN",   SLACK_APP_TOKEN),
        ("OPENAI_API_KEY",    OPENAI_API_KEY),
    ]:
        if not val:
            print(f"❌ {key} missing in .env")
            exit()

    print()
    print("=" * 55)
    print("  🤖  IRT RAG Slack Bot")
    print("=" * 55)
    print(f"  Command      : /irt <your question>")
    print(f"  Knowledge base: {count} documents")
    print(f"  Qdrant storage: {STORAGE_DIR}")
    print("=" * 55)
    print()

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
