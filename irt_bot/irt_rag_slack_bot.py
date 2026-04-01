"""
IRT RAG Slack Bot — v8
Socket Mode: NO URL needed. Just run this script and it connects.

Key changes from v7:
  - Automation categories are no longer hardcoded.
  - They live in automation_categories.json → loaded into Qdrant `automation_kb`.
  - Intent detection uses semantic search against that collection.
  - To add a new automation: edit automation_categories.json → re-run load_automation_kb.py.
  - Everything else (KB search, chat memory, ticket agent, streaming) unchanged.

Run:
    conda activate bug_tracker
    cd /home/user/workspace/python/script_new/irt_rag
    python load_automation_kb.py          # first time, or after updating JSON
    python irt_rag_slack_bot.py
"""

import os, re, time, logging, threading, json
from collections import defaultdict, deque
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN  = os.environ.get("SLACK_APP_TOKEN")
IRT_CHANNEL      = os.environ.get("IRT_SUPPORT_CHANNEL_ID", "C08BUMMH9B2")
TICKET_URL       = os.environ.get("TICKET_CREATE_URL", "https://conversight.slack.com/lists")
TICKET_LIST_ID   = os.environ.get("IRT_TICKET_LIST_ID", "")
COLLECTION       = "irt_knowledge_base"
AUTO_COLLECTION  = "automation_kb"           # ← new Qdrant collection
EMBED_MODEL      = "all-MiniLM-L6-v2"
STORAGE_DIR      = "./qdrant_storage"
TOP_K            = 5
MIN_SCORE        = 0.30
AUTO_MIN_SCORE   = 0.45   # threshold for automation semantic match
CHAT_HISTORY_LEN = 6

AUTOMATION_API_URL = (
    "http://api.conversight.ai/universe-engine/v2/api/resource/action/"
    "crn:dev:us:step_flow:9b505609-832c-453b-9e07-19897c59273e:"
    "standard:irtbot?action=irtbotautomation"
)
AUTOMATION_TOKEN = os.environ.get(
    "IRT_AUTOMATION_TOKEN",
    "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfZG9jIjp7InVzZXJJZCI6ImIzNDJlYjJmLTM5ZmYtNDY2NS1iOWMwLTg1ZDdiYjM2NDk0OSIsImF0aGVuYUlkIjoiYjM0MmViMmYtMzlmZi00NjY1LWI5YzAtODVkN2JiMzY0OTQ5Iiwib3JnSWQiOiI5YjUwNTYwOS04MzJjLTQ1M2ItOWUwNy0xOTg5N2M1OTI3M2UiLCJkZXZpY2VJZCI6IjEyMzQ1NiIsImRldmljZU5hbWUiOiJCcm93c2VyV2ViIiwiaXNUcmlhbFVzZXIiOmZhbHNlLCJpc0ZpcnN0VGltZUxvZ2luIjpmYWxzZSwic2Vzc2lvbklkIjoiY3MtOTg4ODFlMjktZWYzOS00ZmI5LTg0Y2ItMDlmODI4OGEwMjliIn0sImlhdCI6MTc3NDk0NzE1OX0.hdKXPR9sKxChI30o6UVv9X58VybyiBIA0Ep5itrj8zo"
)

app = App(token=SLACK_BOT_TOKEN)

# ── Load models once ──────────────────────────────────────────────────────────
print("⏳ Loading embedding model …")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model ready")

print("⏳ Connecting to Qdrant …")
qclient  = QdrantClient(path=STORAGE_DIR)
kb_count = qclient.count(collection_name=COLLECTION).count

# Verify automation_kb exists
try:
    auto_count = qclient.count(collection_name=AUTO_COLLECTION).count
    print(f"✅ Qdrant ready — {kb_count} KB docs, {auto_count} automation categories")
except Exception:
    print(f"✅ Qdrant ready — {kb_count} KB docs")
    print("⚠️  automation_kb collection not found — run load_automation_kb.py first!")
    auto_count = 0

ai = OpenAI()

STEPS = [
    "⏳  _Hold on, looking into this for you…_",
    "🔍  _Found some related cases, analysing…_",
    "✍️   _Almost there, putting together your answer…_",
]

# ── Conversation memory ───────────────────────────────────────────────────────
_history: dict = defaultdict(lambda: deque(maxlen=CHAT_HISTORY_LEN * 2))

def _conv_key(user: str, channel: str) -> str:
    return f"{user}::{channel}"

def _get_history(user: str, channel: str) -> list:
    return list(_history[_conv_key(user, channel)])

def _add_history(user: str, channel: str, role: str, content: str):
    _history[_conv_key(user, channel)].append({"role": role, "content": content})

def _clear_history(user: str, channel: str):
    key = _conv_key(user, channel)
    if key in _history:
        del _history[key]


# ── Pending clarifications ─────────────────────────────────────────────────────
_pending: dict = {}

def _save_pending(ts: str, query: str, user: str, channel: str):
    _pending[ts] = {"query": query, "user": user, "channel": channel}

def _get_pending(thread_ts: str) -> dict | None:
    return _pending.get(thread_ts)

def _clear_pending(ts: str):
    _pending.pop(ts, None)


# ── Automation state ──────────────────────────────────────────────────────────
_automation_state: dict = {}

def _get_auto_state(user: str, channel: str = None) -> dict | None:
    return _automation_state.get(user)

def _set_auto_state(user: str, channel: str = None, state: dict = None):
    _automation_state[user] = state

def _clear_auto_state(user: str, channel: str = None):
    _automation_state.pop(user, None)


# ── Last answer memory ────────────────────────────────────────────────────────
_last_answer: dict = {}

def _save_last_answer(user: str, question: str, answer: str, hits: list):
    _last_answer[user] = {"question": question, "answer": answer, "hits": hits[:1]}

def _get_last_answer(user: str) -> dict | None:
    return _last_answer.get(user)


# ── Ticket state ──────────────────────────────────────────────────────────────
_ticket_state: dict = {}

def _get_ticket_state(user: str) -> dict | None:
    return _ticket_state.get(user)

def _set_ticket_state(user: str, state: dict):
    _ticket_state[user] = state

def _clear_ticket_state(user: str):
    _ticket_state.pop(user, None)


# ═════════════════════════════════════════════════════════════════════════════
# AUTOMATION KB — semantic search & dynamic agent
# ═════════════════════════════════════════════════════════════════════════════

def search_automation_kb(query: str, top_k: int = 1) -> list:
    """
    Semantic search against the automation_kb Qdrant collection.
    Returns a list of matched category payloads (the full JSON dicts from the JSON file).
    """
    try:
        vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
        results = qclient.query_points(
            collection_name=AUTO_COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {"score": round(r.score, 3), **r.payload}
            for r in results
        ]
    except Exception as e:
        log.error(f"automation KB search error: {e}")
        return []


def detect_automation_from_kb(query: str) -> dict | None:
    """
    Semantic lookup: if the user's message matches an automation category
    above AUTO_MIN_SCORE, return that category dict (from the JSON).
    Returns None if no match is found.
    """
    hits = search_automation_kb(query, top_k=1)
    if not hits:
        return None
    top = hits[0]
    log.warning(f"automation KB match: '{top['category']}' score={top['score']}")
    if top["score"] >= AUTO_MIN_SCORE:
        return top
    return None


def _extract_json_from_message(message: str) -> dict:
    """
    Silently checks if the user's message contains a JSON object
    and extracts key-value pairs from it.
    Supports formats like: {"org_id": "abc", "extend_period": "2026-04-28"}
    Returns empty dict if no valid JSON found.
    """
    # Try to find a JSON block in the message (with or without code fences)
    json_patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",   # ```json {...} ```
        r"`(\{.*?\})`",                        # inline `{...}`
        r"(\{[^{}]*\})",                       # plain {...}
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, message, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and data:
                    log.warning(f"_extract_json_from_message: found JSON with keys={list(data.keys())}")
                    return {str(k).strip(): str(v).strip() for k, v in data.items()}
            except Exception:
                continue
    return {}


def _extract_field_with_ai(key: str, label: str, hint: str, message: str,
                            already_collected: dict = None) -> str | None:
    """
    Extracts a single field value from the user's message.
    Checks JSON format first, then regex, then GPT.
    already_collected: fields already saved — used to avoid reusing same email.
    """
    if already_collected is None:
        already_collected = {}

    msg_lower = message.lower().strip()

    # ── Step 0: JSON extraction (silent — not mentioned to user) ─────────────
    json_data = _extract_json_from_message(message)
    if json_data:
        # Try exact key match first
        if key in json_data:
            val = json_data[key].strip().rstrip(".,;")
            log.warning(f"_extract_field_with_ai (json-exact): key={key} → '{val}'")
            return val
        # Try label-based match (e.g. "Organisation ID" → org_id)
        label_lower = label.lower().replace(" ", "_")
        for jk, jv in json_data.items():
            if jk.lower().replace(" ", "_") in (key, label_lower):
                val = jv.strip().rstrip(".,;")
                log.warning(f"_extract_field_with_ai (json-label): key={key} jk={jk} → '{val}'")
                return val

    # ── Fast role extraction — handles typos like "uesr", "adimin", "usr" ──────
    if key == "role":
        msg_lower_stripped = msg_lower.strip()
        # Direct line match — check each line for role-like value
        for line in message.splitlines():
            line_clean = line.strip().lower().rstrip(".,;:")
            # Exact or close match
            if line_clean in ("user", "admin", "usr", "uesr", "adm", "adimin", "admn"):
                val = "admin" if line_clean.startswith("adm") else "user"
                log.warning(f"_extract_field_with_ai (role-fuzzy): '{line_clean}' → '{val}'")
                return val
            # Pattern: "role: user" or "role = admin"
            m = re.search(r"(?i)role\s*[:\-=]\s*(\w+)", line)
            if m:
                raw = m.group(1).lower()
                val = "admin" if raw.startswith("adm") else "user"
                log.warning(f"_extract_field_with_ai (role-pattern): '{raw}' → '{val}'")
                return val

    # Build alias list from the key
    key_aliases = {
        "org_id":        ["org_id", "orgid", "org", "organisation", "organization", "org id"],
        "dataset_id":    ["dataset_id", "datasetid", "dataset", "ds"],
        "tenant_id":     ["tenant_id", "tenantid", "tenant"],
        "user_id":       ["user_id", "userid", "user"],
        "extend_period": ["extend_period", "expiry", "expiry date", "expiry_date", "date", "extend till", "new date", "till"],
        "time_in_utc":   ["time_in_utc", "utc", "datetime", "time"],
        "time_in_minutes": ["time_in_minutes", "minutes", "timeout", "mins"],
        "refreshTime":   ["refreshtime", "refresh_time", "refresh time", "time"],
        "timezone":      ["timezone", "tz", "time zone"],
        "fetch_limit":   ["fetch_limit", "fetchlimit", "limit", "fetch limit"],
        "user_count":    ["user_count", "usercount", "count", "users"],
        "schema_to_activate": ["schema", "schema_to_activate"],
        "activate_type": ["activate_type", "type", "activation type"],
        "old_email":     ["old_email", "old email", "current email", "from", "old@", "old :"],
        "new_email":     ["new_email", "new email", "to", "new@", "new :"],
        "role":          ["role"],
    }.get(key, [key.replace("_", " "), key.replace("_", ""), key])

    for alias in key_aliases:
        # Email-aware pattern
        pattern = rf"(?i)\b{re.escape(alias)}\s*[:\-=@]?\s*([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{{2,}})"
        m = re.search(pattern, message)
        if m:
            val = m.group(1).strip().rstrip(".,;")
            log.warning(f"_extract_field_with_ai (regex-email): key={key} alias='{alias}' → '{val}'")
            return val
        # General pattern
        pattern = rf"(?i)\b{re.escape(alias)}\s*[:\-=]\s*(\S+)"
        m = re.search(pattern, message)
        if m:
            val = m.group(1).strip().rstrip(".,;")
            log.warning(f"_extract_field_with_ai (regex): key={key} alias='{alias}' → '{val}'")
            return val

    # ── Email position-based extraction ──────────────────────────────────────
    if key in ("old_email", "new_email"):
        email_pattern = r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        emails = re.findall(email_pattern, message)

        # Normalise space-broken emails if fewer than 2 found
        if len(emails) < 2:
            normalised = re.sub(
                r"([a-zA-Z0-9_.+\-]+)\s+([a-zA-Z0-9_.+\-]*@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
                r"\1\2", message
            )
            emails_normalised = re.findall(email_pattern, normalised)
            if len(emails_normalised) > len(emails):
                log.warning(f"_extract_field_with_ai (email-normalised): found {emails_normalised}")
                emails = emails_normalised

        if emails:
            # Filter out emails already collected in OTHER fields
            # so we don't reuse the same email for both old and new
            already_used = {
                v.lower() for k, v in already_collected.items()
                if k != key and "@" in str(v)
            }
            fresh_emails = [e for e in emails if e.lower() not in already_used]
            log.warning(
                f"_extract_field_with_ai (email-pos): key={key} "
                f"all={emails} already_used={already_used} fresh={fresh_emails}"
            )

            if key == "old_email":
                # Always take first email for old_email
                val = emails[0]
                log.warning(f"_extract_field_with_ai (email-pos): key=old_email → '{val}'")
                return val

            if key == "new_email":
                # Take first fresh email (not already used as old_email)
                if fresh_emails:
                    val = fresh_emails[0]
                    log.warning(f"_extract_field_with_ai (email-pos): key=new_email → '{val}' (fresh)")
                    return val
                # All emails are already used — only 1 unique email, ask separately
                log.warning(f"_extract_field_with_ai (email-pos): key=new_email — no fresh email found")
                return None

    # ── GPT fallback ──────────────────────────────────────────────────────────
    # Build field-specific instructions for better extraction
    extra_instructions = ""
    if key in ("old_email", "new_email"):
        pos = "first" if key == "old_email" else "second"
        extra_instructions = (
            f"\nThis is an email field. The message may contain two emails — "
            f"pick the {pos} one. "
            f"If there is only ONE email in the message and this is new_email, return NOT_FOUND.\n"
            f"Emails may have spaces inserted (e.g. 'john smith@email.com') — "
            f"remove the space and return the joined email."
        )
    elif key == "role":
        extra_instructions = (
            f"\nThis is a role field. Valid values are ONLY 'admin' or 'user'.\n"
            f"Common typos to correct: 'uesr'→'user', 'usr'→'user', 'adimin'→'admin', 'adm'→'admin'.\n"
            f"Return exactly 'admin' or 'user' — nothing else."
        )

    resp = ai.chat.completions.create(
        model      = os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens = 30,
        messages   = [
            {"role": "system", "content":
                f"You are extracting a specific field from a user message.\n\n"
                f"Field to extract: '{label}' (internal key: {key})\n"
                f"Example value format: {hint}"
                f"{extra_instructions}\n\n"
                f"The user may write values as:\n"
                f"  - Labelled:   'old: email@x.com', 'new: email@x.com'\n"
                f"  - Plain list: one value per line\n"
                f"  - Direct:     just the value itself\n\n"
                f"Rules:\n"
                f"  - Return ONLY the raw value\n"
                f"  - If not clearly present, return exactly: NOT_FOUND\n"
                f"  - Never guess or invent"},
            {"role": "user", "content": message},
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    log.warning(f"_extract_field_with_ai (gpt): key={key} → '{result}'")
    return None if (not result or result.upper() == "NOT_FOUND") else result


def _extract_all_fields_from_message(category_def: dict, message: str) -> dict:
    """
    Attempts to extract all field values that may already be in the trigger message.
    Returns a dict of {key: value} for values that were found.
    """
    collected = {}
    for field in category_def.get("fields", []):
        value = _extract_field_with_ai(
            field["key"], field["label"], field["hint"], message,
            already_collected=collected  # ← pass what's been collected so far
        )
        if value:
            collected[field["key"]] = value
            log.warning(f"pre-extracted {field['key']}='{value}'")

    # ── Email dedup fix: if old_email and new_email are the same value,
    # only one email was provided — keep old_email, clear new_email
    # so the bot asks for new_email separately
    if (collected.get("old_email") and collected.get("new_email")
            and collected["old_email"].lower() == collected["new_email"].lower()):
        del collected["new_email"]
        log.warning("pre-extracted: old_email == new_email — cleared new_email, will ask separately")

    return collected


def _is_field_required(field: dict, collected: dict) -> bool:
    """
    Determines if a field is required given the current collected values.
    Handles conditional required_when logic.
    """
    if field.get("required"):
        return True
    if "required_when" in field:
        for cond_key, cond_val in field["required_when"].items():
            if collected.get(cond_key, "").lower() == cond_val.lower():
                return True
    return False


def _build_payload(category_def: dict, collected: dict) -> dict:
    """
    Constructs the API payload from collected field values.
    Handles transforms (to_int, yes_no_to_bool, split_by_comma, special cases).
    """
    transforms = category_def.get("payload_transform", {})
    template   = category_def.get("payload_template", {})

    # Special case: Admin Email Changes — send all provided fields
    if template == "dynamic_all_provided":
        payload = {}
        for field in category_def.get("fields", []):
            k = field["key"]
            v = collected.get(k)
            if v:
                payload[k] = v
        return payload

    # Special case: Activate Dataset — nested schema structure
    if category_def["category"] == "Activate Dataset":
        activate_type = collected.get("activate_type", "current_schema")
        return {
            "dataset_id": collected["dataset_id"],
            "org_id":     collected["org_id"],
            "schema": {
                "schema_to_activate": collected["schema_to_activate"],
                f"activate_{activate_type}": True,
            }
        }

    # General case: substitute from template
    def _transform(key, val):
        t = transforms.get(key, "")
        if t == "to_int":
            try:
                return int(str(val).strip())
            except Exception:
                return val
        if t == "yes_no_to_bool":
            return str(val).lower().strip() in ("yes", "true", "1", "y")
        if t == "split_by_comma":
            return [v.strip() for v in str(val).split(",") if v.strip()]
        return val

    def _fill(obj):
        if isinstance(obj, dict):
            return {k: _fill(v) for k, v in obj.items()}
        if isinstance(obj, str) and obj.startswith("{") and obj.endswith("}"):
            key = obj[1:-1]
            raw = collected.get(key, "")
            return _transform(key, raw)
        return obj

    return _fill(template)


def call_automation_api(category: str, details: dict) -> dict:
    """Calls the IRT Automation API using requests library. Returns {ok, message}."""
    import requests as req_lib
    payload = json.dumps({"config": {"category": category, "details": details}})
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {AUTOMATION_TOKEN}",
    }
    try:
        resp = req_lib.post(
            AUTOMATION_API_URL,
            headers = headers,
            data    = payload,
            timeout = 30,
        )
        log.warning(f"automation API status={resp.status_code} response={resp.text[:200]}")
        if resp.status_code == 200:
            return {"ok": True, "message": resp.text[:300]}
        else:
            return {"ok": False, "message": f"API error {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        log.error(f"automation API error: {e}")
        return {"ok": False, "message": str(e)[:200]}


def automation_agent(user: str, channel: str, message: str, category_def: dict = None) -> str:
    """
    KB-driven agentic automation loop.

    Instead of a hardcoded category dict, it uses a `category_def` fetched
    from the Qdrant automation_kb collection (full JSON payload).

    On each call it either:
      - Starts a new automation session (if category_def given)
      - Handles cancel/confirm
      - Collects the next missing required field
      - Executes the API when all fields are ready and confirmed

    Returns a Slack-formatted string.
    """
    state = _get_auto_state(user, channel)

    # ── Cancel ────────────────────────────────────────────────────────────────
    if message.lower().strip() in ("cancel", "abort", "stop", "exit"):
        if state:
            cat_name = state.get("category_def", {}).get("category", "Automation")
            _clear_auto_state(user, channel)
            log.warning(f"automation_agent: session cancelled — user={user} category={cat_name}")
        return "❌ Automation cancelled. Ask me anything else!"

    # ── Start new session ─────────────────────────────────────────────────────
    if category_def and not state:
        # Pre-extract any field values already in the trigger message
        pre_collected = _extract_all_fields_from_message(category_def, message)
        state = {
            "category_def":       category_def,
            "collected":          pre_collected,
            "awaiting_confirm":   False,
            "owner":              user,
            "closed":             False,
        }
        _set_auto_state(user, channel, state)
        log.warning(
            f"[AUTO] SESSION START — user={user} category={category_def['category']} "
            f"pre_extracted={list(pre_collected.keys())}"
        )

    if not state:
        return "⚠️ No active automation session. Try describing what you want to do."

    cat_def   = state["category_def"]
    collected = state["collected"]
    fields    = cat_def.get("fields", [])

    # ── Handle confirmation ───────────────────────────────────────────────────
    if state.get("awaiting_confirm"):
        if message.lower().strip() in ("confirm", "yes", "proceed", "ok", "y"):
            try:
                details = _build_payload(cat_def, collected)
            except Exception as e:
                log.warning(f"[AUTO] PAYLOAD BUILD FAILED — user={user} category={cat_def['category']} error={e}")
                return f"⚠️ Failed to build payload: {e}\n_Please try again or type *cancel* to abort._"

            log.warning(
                f"[AUTO] API CALL START — user={user} category={cat_def['category']} "
                f"details={json.dumps(details)[:200]}"
            )
            result = call_automation_api(cat_def["category"], details)

            if result["ok"]:
                _clear_auto_state(user, channel)
                log.warning(f"[AUTO] API SUCCESS — user={user} category={cat_def['category']}")
                return (
                    f"✅ *{cat_def['category']}* completed successfully! "
                    f"The action has been applied — it may take a minute to reflect on the platform."
                )
            else:
                state["collected"]        = {}
                state["awaiting_confirm"] = False
                _set_auto_state(user, channel, state)
                log.warning(f"[AUTO] API FAILED — user={user} category={cat_def['category']} error={result['message'][:150]}")
                return (
                    f"❌ *{cat_def['category']}* failed.\n\n"
                    f"_Error:_ {result['message']}\n\n"
                    f"_Please provide all the details again to retry._"
                )
        else:
            cat_name = cat_def.get("category", "Automation")
            _clear_auto_state(user, channel)
            log.warning(f"[AUTO] CANCELLED AT CONFIRM — user={user} category={cat_name}")
            return "❌ Automation cancelled. Ask me anything else!"

    # ── Follow-up message — extract ALL missing fields from one message ───────
    if not state.get("just_started"):
        missing_now = [
            f for f in fields
            if f["key"] not in collected and _is_field_required(f, collected)
        ]
        if missing_now:
            # Try to extract every missing field from this single message
            extracted_any = False
            for field in missing_now:
                extracted = _extract_field_with_ai(
                    field["key"], field["label"], field["hint"], message,
                    already_collected=collected  # ← pass context so emails aren't reused
                )
                if extracted:
                    collected[field["key"]] = extracted
                    extracted_any = True

            # Email dedup: if both emails extracted and they're the same,
            # only one email was given — keep old, re-ask for new
            if (collected.get("old_email") and collected.get("new_email")
                    and collected["old_email"].lower() == collected["new_email"].lower()):
                del collected["new_email"]
                log.warning("follow-up: old_email == new_email — cleared new_email")
            state["collected"] = collected
            # Track consecutive failed extractions to avoid infinite loop
            if not extracted_any:
                state["_failed_extractions"] = state.get("_failed_extractions", 0) + 1
                log.warning(
                    f"[AUTO] FIELD EXTRACT FAILED — user={user} "
                    f"category={cat_def.get('category','')} "
                    f"attempt={state['_failed_extractions']} msg='{message[:60]}'"
                )
            else:
                state["_failed_extractions"] = 0
                log.warning(
                    f"[AUTO] FIELDS COLLECTED — user={user} "
                    f"collected={list(collected.keys())} msg='{message[:60]}'"
                )
            _set_auto_state(user, channel, state)
        else:
            # All fields already collected — user may be correcting a value
            for field in fields:
                extracted = _extract_field_with_ai(
                    field["key"], field["label"], field["hint"], message
                )
                if extracted:
                    collected[field["key"]] = extracted
                    state["collected"] = collected
                    state["_failed_extractions"] = 0
                    _set_auto_state(user, channel, state)
                    break

    state.pop("just_started", None)
    _set_auto_state(user, channel, state)

    # ── Find missing required fields ──────────────────────────────────────────
    missing = [
        f for f in fields
        if f["key"] not in collected and _is_field_required(f, collected)
    ]

    if missing:
        _set_auto_state(user, channel, state)

        # Show already collected fields as context
        collected_lines = ""
        if collected:
            lines = []
            for f in fields:
                if f["key"] in collected:
                    lines.append(f"   ✅ *{f['label']}:* `{collected[f['key']]}`")
            collected_lines = "\n".join(lines) + "\n\n"

        # Ask ALL missing fields at once — numbered list
        missing_lines = []
        for i, f in enumerate(missing, 1):
            missing_lines.append(f"   *{i}. {f['label']}*\n   _{f['hint']}_")
        missing_text = "\n\n".join(missing_lines)

        intro = "📝 *Please provide the following details:*" if len(missing) > 1 else f"📝 Please provide *{missing[0]['label']}*"

        # If extraction failed multiple times, show a clearer hint
        failed_count = state.get("_failed_extractions", 0)
        retry_hint = ""
        if failed_count >= 2:
            retry_hint = (
                f"\n\n⚠️ _I'm having trouble reading your input. "
                f"Please paste the value directly, e.g.:_ `{missing[0]['hint']}`"
            )

        return (
            f"{collected_lines}"
            f"{intro}\n\n"
            f"{missing_text}"
            f"{retry_hint}\n\n"
            f"_You can provide all of them in one message or one at a time._"
        )

    # ── All fields collected → show confirmation with buttons ─────────────────
    summary_lines = []
    for f in fields:
        if f["key"] in collected:
            summary_lines.append(f"   *{f['label']}:* `{collected[f['key']]}`")
    summary = "\n".join(summary_lines)

    state["awaiting_confirm"] = True
    _set_auto_state(user, channel, state)
    log.warning(
        f"[AUTO] READY TO CONFIRM — user={user} category={cat_def['category']} "
        f"fields={list(collected.keys())}"
    )
    # Return special marker so caller renders confirm/cancel buttons
    return "__CONFIRM__:" + f"🔧 *Ready to execute: {cat_def['category']}*\n\n{summary}"


def confirm_action_blocks(summary_text: str) -> list:
    """Builds a Slack block with the summary + Confirm / Cancel buttons."""
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": summary_text}
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅  Confirm", "emoji": True},
                    "style": "primary",
                    "action_id": "automation_confirm",
                    "value": "confirm",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌  Cancel", "emoji": True},
                    "style": "danger",
                    "action_id": "automation_cancel",
                    "value": "cancel",
                },
            ]
        }
    ]


def automation_info_response(category_def: dict) -> str:
    """
    Returns a Slack message describing the required inputs for a category.
    Built directly from the category_def payload — always accurate.
    """
    fields = category_def.get("fields", [])
    lines  = []
    for f in fields:
        req_tag = "" if f.get("required") else " _(optional)_"
        lines.append(f"• *{f['label']}*{req_tag} — `{f['hint']}`")
    fields_text = "\n".join(lines)
    cat = category_def["category"]
    return (
        f"*📋 Inputs required for {cat}:*\n\n"
        f"{fields_text}\n\n"
        f"_Just say *\"{cat.lower()}\"* and I'll guide you through it step by step._"
    )


# ═════════════════════════════════════════════════════════════════════════════
# EXISTING helpers (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _friendly_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return "The bot is receiving too many requests right now. Please try again in a moment."
    if "timeout" in msg or "timed out" in msg:
        return "The request took too long to process. Please try again."
    if "qdrant" in msg or "collection" in msg:
        return "The knowledge base is temporarily unavailable. Please try again shortly."
    if "openai" in msg or "api key" in msg or "authentication" in msg:
        return "The AI service is temporarily unavailable. Please try again shortly."
    if "channel_not_found" in msg or "not_in_channel" in msg:
        return "The bot doesn't have access to this channel. Please contact your workspace admin."
    return "Something went wrong. Please try again, or contact the IRT team if this persists."


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search_kb(query: str) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = qclient.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=TOP_K,
        with_payload=True,
    ).points
    return [
        {
            "score"            : round(r.score, 3),
            "summary"          : r.payload.get("summary", ""),
            "solution"         : clean(r.payload.get("solution", "")),
            "resolution_status": r.payload.get("final_status",
                                 r.payload.get("resolution_status",
                                 r.payload.get("status", ""))),
            "bug_category"     : r.payload.get("bug_category", ""),
            "severity"         : r.payload.get("severity", ""),
            "references"       : r.payload.get("references", "None"),
            "source"           : r.payload.get("source", "Excel"),
        }
        for r in results
    ]


def build_enriched_query(original_question: str, clarification_answer: str) -> str:
    resp = ai.chat.completions.create(
        model      = os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens = 60,
        messages   = [
            {"role": "system", "content":
                "Combine the original question and the clarification answer into ONE "
                "natural complete sentence usable as a search query. "
                "Put the clarification details first (version, environment), then the issue. "
                "Return ONLY the combined query — no explanation, no quotes."},
            {"role": "user", "content":
                f"Original question: {original_question}\n"
                f"Clarification answer: {clarification_answer}"},
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    return result if result else f"{clarification_answer} {original_question}"


def generate_answer(query: str, hits: list, history: list = None) -> str:
    context = ""
    for i, h in enumerate(hits, 1):
        sol = h["solution"]
        if not sol or sol in ("nan", "None", "No solution documented."):
            sol = "(No specific solution recorded)"
        tag = " [RCA]" if h.get("source") == "RCA" else ""
        context += f"""
Case {i}{tag} (relevance: {h['score']:.2f})
  Issue    : {h['summary']}
  Category : {h['bug_category']}
  Status   : {h['resolution_status']}
  Solution : {sol}
  Refs     : {h['references']}
"""
    system_prompt = f"""You are IRT Bot, an Incident Response Team support assistant for ConverSight.
ConverSight has two dataset versions — v1 (legacy) and v2 (current). Always tailor your answer to the correct version.

Relevant past cases:
{context}

How to respond:
1. Start with "Yes, the IRT team has seen this before." OR "This looks like a new issue — please raise it with the IRT team."
2. For each relevant case: "In a case where [issue], the fix was [exact solution]."
   Then: "This was a *permanent fix*." or "This was a *workaround*."
3. Write "*Steps to try:*" then 2-4 bullets ONLY from the Solution fields.
4. End with: "If this doesn't help, share your *Dataset name*, *Org ID*, *Environment*, and *current status* with the IRT team."

Rules: *bold* key terms. No "knowledge base" or "Case #1". Under 300 words."""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    resp = ai.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens=600,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def clarify_blocks(question: str, suggestions: list) -> list:
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"🤔 {question}"}}]
    if suggestions:
        buttons = []
        for i, s in enumerate(suggestions[:5]):
            buttons.append({
                "type": "button",
                "text": {"type": "plain_text", "text": s, "emoji": True},
                "action_id": f"clarify_reply_{i}",
                "value": s,
            })
        blocks.append({"type": "actions", "elements": buttons})
    return blocks


def handle_conversational(query: str, history: list = None) -> str:
    q = query.lower().strip().rstrip("!?.,")

    live_data_triggers = [
        "status of today", "today's ticket", "today tickets", "current status",
        "live data", "real time", "realtime", "counts of ticket", "ticket count",
        "how many ticket", "tickets closed", "tickets open", "tickets raised",
        "this week", "this month", "last week", "last month",
    ]
    if any(t in q for t in live_data_triggers):
        return (
            "📊 *Live data is not available yet.*\n\n"
            "I currently work with past incident data from the IRT knowledge base. "
            "Real-time ticket counts and live metrics are not connected yet.\n\n"
            "_For current ticket status please check the Bug Tracker directly._"
        )

    capability_triggers = {
        "what can you do", "what do you do", "who are you", "what are you",
        "help", "what is this", "how does this work", "tell me about yourself",
        "capabilities", "features",
    }
    if q in capability_triggers or any(t in q for t in [
        "what can you", "what do you", "capabilities", "can you do",
        "what are you able", "what are you capable",
    ]):
        return (
            "Hi! I'm *IRT Bot* — Conversight Immediate Response Team assistant.\n\n"
            "*🔍 I can help you diagnose issues:*\n"
            "Describe any ConverSight product problem — dataset failures, dataload errors, "
            "SME publish issues, notebook errors, connector failures, dashboard bugs — "
            "and I'll search past incidents to find what worked.\n\n"
            "*⚙️ I can do automations for you:*\n"
            "Like *Activate Dataset*, *Enable Athena IQ*, *Extend Trial Period*, "
            "*Increase User Count*, *Remove SME Duplicates* and more — "
            "just tell me what you need done.\n\n"
            "_Just describe your issue or tell me what you want done — I'll take it from there!_"
        )

    system_prompt = """You are IRT Bot — Conversight Immediate Response Team assistant.
Respond naturally. 2-3 sentences max. Be friendly. Use *bold* for emphasis.
Never mention "knowledge base" or internal system details."""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    resp = ai.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens=100,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def _sim_label(score: int) -> str:
    if score >= 85:
        return f"🟢 *{score}% similarity* — nearly identical issue"
    elif score >= 65:
        return f"🟡 *{score}% similarity* — clearly related"
    elif score >= 50:
        return f"🟠 *{score}% similarity* — loosely related"
    else:
        return f"🔴 *{score}% similarity* — vaguely related"


def _format_reference(ref: str) -> str:
    if not ref or str(ref).strip().lower() in (
        "none", "nan", "", "link", "n/a", "-", "null", "no reference"
    ):
        return ""
    ref = str(ref).strip()
    if re.match(r"^<https?://[^>]+>$", ref):
        return ref
    url_match = re.search(r"https?://\S+", ref)
    if url_match:
        url = url_match.group(0).rstrip(".,)>\"'")
        label = (
            "Asana ticket" if "asana.com" in url else
            "Slack thread" if "slack.com" in url else
            "GitHub" if "github.com" in url else
            "Jira ticket" if "jira" in url else
            "Google Doc" if "docs.google" in url else "Reference"
        )
        return f"<{url}|{label}>"
    if ref.lower() in ("link", "url", "ref", "reference", "ticket", "doc"):
        return ""
    return f"_{ref}_"


def build_blocks(query: str, answer: str, hits: list) -> list:
    icons      = {"Fixed": "✅", "Partial": "⚠️", "Workaround": "⚠️",
                  "Unresolved": "❌", "Rejected": "🚫"}
    res_labels = {"Fixed": "Fixed", "Partial": "Partial fix",
                  "Workaround": "Workaround", "Unresolved": "Unresolved",
                  "Rejected": "Rejected"}

    hits_text = ""
    for h in hits[:3]:
        icon  = icons.get(h["resolution_status"], "❓")
        label = res_labels.get(h["resolution_status"], h["resolution_status"])
        score = int(h["score"] * 100)
        bar   = "█" * (score // 10) + "░" * (10 - score // 10)
        src   = "  📄 _RCA doc_" if h.get("source") == "RCA" else ""
        ref   = _format_reference(h.get("references", ""))
        hits_text += f"{icon} *{h['summary'][:70]}*{src}\n"
        hits_text += f"   `{bar}` {score}% | _{label}_ | {h['bug_category']}\n"
        if ref:
            hits_text += f"   📎 {ref}\n"
        hits_text += "\n"

    return [
        {"type": "header", "text": {"type": "plain_text", "text": "🤖 IRT Bot", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Your question:*\n{query}"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*💡 Answer:*\n{answer}"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*📋 Similar past issues:*\n\n{hits_text}"}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"}},
    ]


def step_block(txt: str) -> list:
    return [{"type": "section", "text": {"type": "mrkdwn", "text": txt}}]


def automation_anchor_blocks(query: str) -> list:
    """
    Anchor message shown in the channel when an automation thread opens.
    Always contains a 🔴 Close Thread button — visible at the top of the thread
    so the user can cancel/close at any time without typing.
    """
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"⚙️ *Automation request:* {query}"
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "🔴  Close Thread", "emoji": True},
                "style": "danger",
                "action_id": "close_automation_thread",
                "value": "close",
            }
        }
    ]


# ═════════════════════════════════════════════════════════════════════════════
# QUERY ROUTER — now uses automation KB semantic match for AUTOMATE routing
# ═════════════════════════════════════════════════════════════════════════════

def analyze_query(query: str, history: list) -> dict:
    """
    Routes the user query to one of:
      search / clarify / chat / outofscope / automate / automateinfo / ticket

    Automation detection now uses semantic search against automation_kb
    FIRST (fast, no GPT call needed for obvious matches).
    Falls back to GPT routing for ambiguous cases.
    """

    q_lower = query.lower()

    # ── 1. Ticket creation fast path ──────────────────────────────────────────
    _ticket_triggers = [
        "create ticket", "create a ticket", "raise ticket", "raise a ticket",
        "log ticket", "log this", "log the issue", "open a ticket",
        "report this issue", "file a ticket", "file a bug",
        "create bug", "raise bug", "ticket for this", "ticket for the above",
        "create the ticket", "make a ticket",
    ]
    if any(t in q_lower for t in _ticket_triggers):
        return {"action": "ticket", "text": ""}

    # ── 2. Automation semantic match ──────────────────────────────────────────
    # Action verbs — broad list covers natural language variations
    action_verbs = [
        "extend", "increase", "change", "enable", "activate", "remove",
        "update", "get", "disable", "set", "add", "delete",
        "provide", "fetch", "show", "give", "find", "check",
        "count", "entity count", "user count",
    ]
    has_action_verb = any(v in q_lower for v in action_verbs)

    if has_action_verb and auto_count > 0:
        cat_def = detect_automation_from_kb(query)
        if cat_def:
            log.warning(f"analyze_query: AUTOMATE → {cat_def['category']} (score={cat_def['score']})")
            return {"action": "automate", "text": cat_def["category"], "category_def": cat_def}

    # ── 3. Info about an automation (what inputs does X need?) ────────────────
    info_triggers = ["what inputs", "what fields", "what do i need for", "how to use",
                     "what is needed for", "details for", "inputs for", "fields for"]
    if any(t in q_lower for t in info_triggers) and auto_count > 0:
        cat_def = detect_automation_from_kb(query)
        if cat_def:
            return {"action": "automateinfo", "text": cat_def["category"], "category_def": cat_def}

    # ── 4. GPT fallback router for search/clarify/chat/outofscope ────────────
    history_text = ""
    if history:
        for msg in history:
            role = "User" if msg["role"] == "user" else "Bot"
            history_text += f"{role}: {msg['content']}\n"

    # Build a concise list of automation category names for GPT context
    auto_categories_hint = ""
    if auto_count > 0:
        try:
            all_cats = qclient.scroll(
                collection_name=AUTO_COLLECTION, limit=50, with_payload=True
            )[0]
            cat_names = [p.payload.get("category", "") for p in all_cats]
            auto_categories_hint = "\n".join(f"- {c}" for c in cat_names if c)
        except Exception:
            pass

    history_block = ("History:\n" + history_text) if history_text else ""

    system_prompt = f"""IRT Bot query router for ConverSight support.
{history_block}
Message: "{query}"

Reply EXACTLY one of:
SEARCH: <query>
CLARIFY: <question>
SUGGESTIONS: opt1 | opt2 | opt3
CHAT:
OUTOFSCOPE:
AUTOMATE: <category name>
AUTOMATEINFO: <category name>

Automation categories available:
{auto_categories_hint}

Rules:
- v1/v2 explicitly in message → SEARCH immediately
- Greeting/thanks/capability questions → CHAT
- API keys, tokens, security, coding → OUTOFSCOPE
- Use AUTOMATE only if user clearly wants to execute an operation
- Use AUTOMATEINFO when user asks what inputs/fields are needed

MUST CLARIFY for version when ALL true:
1. Describes a dataset/dataload/SME/cluster issue
2. No version (v1 or v2) mentioned
3. Fix differs between v1 and v2

Examples NOT needing version: notebooks, connectors, storyboard, UI issues."""

    resp = ai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=120,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ],
    )
    result = (resp.choices[0].message.content or "").strip().replace("\\n", "\n")
    lines  = [l.strip() for l in result.strip().splitlines() if l.strip()]
    log.warning(f"analyze_query GPT result: {repr(result)}")

    if not lines:
        return {"action": "search", "text": query}

    first = lines[0].upper()

    if first.startswith("CLARIFY:"):
        question    = lines[0][8:].strip()
        suggestions = []
        for line in lines[1:]:
            if line.upper().startswith("SUGGESTIONS:"):
                suggestions = [s.strip() for s in line[12:].strip().split("|") if s.strip()]
                break
        return {"action": "clarify", "text": question, "suggestions": suggestions}

    if first.startswith("SEARCH:"):
        return {"action": "search", "text": lines[0][7:].strip() or query}

    if first.startswith("CHAT"):
        return {"action": "chat", "text": ""}

    if first.startswith("OUTOFSCOPE"):
        return {"action": "outofscope", "text": ""}

    # GPT said AUTOMATE — do semantic lookup to get the full category_def
    if first.startswith("AUTOMATE:"):
        cat_name = lines[0][9:].strip()
        cat_def  = detect_automation_from_kb(cat_name) or detect_automation_from_kb(query)
        if cat_def:
            return {"action": "automate", "text": cat_def["category"], "category_def": cat_def}
        return {"action": "search", "text": query}

    if first.startswith("AUTOMATEINFO:"):
        cat_name = lines[0][13:].strip()
        cat_def  = detect_automation_from_kb(cat_name) or detect_automation_from_kb(query)
        if cat_def:
            return {"action": "automateinfo", "text": cat_def["category"], "category_def": cat_def}
        return {"action": "search", "text": query}

    return {"action": "search", "text": query}


# ═════════════════════════════════════════════════════════════════════════════
# TICKET AGENT (unchanged from v7)
# ═════════════════════════════════════════════════════════════════════════════

TICKET_CATEGORY_OPTIONS = {
    "Data Load Failure V1":         "OptFI38PO4P",
    "Data Load Failure V2":         "Opt74DUH3XU",
    "Cluster Startup Failure":      "OptJ93QL2JS",
    "Notebook Launch Issues":       "OptUFPFJY50",
    "Scheduled Flows Struck":       "Opt0GZXQQ94",
    "URL Access Issues":            "OptXGU91C0D",
    "Recurring Storyboard Failure": "Opt48UMHNTD",
    "Deployment":                   "OptDHMPQYKR",
    "Cluster Struck Issues":        "OptB4RXY8ET",
    "Others":                       "OptOYUJQ53K",
}
TICKET_TEAM_OPTIONS = {
    "Testing":               "OptP2OSTCNM",
    "DevOps":                "OptYX0RVZMF",
    "Engineering (Backend)": "OptKXV6NQBE",
}
TICKET_ENV_OPTIONS = {
    "AWS Production": "OptBMCHH1RK",
    "GCP Staging":    "OptCUZ738HM",
    "GCP Production": "OptYKZKPJWD",
}


def _map_to_option_id(user_value: str, options: dict) -> str:
    v = user_value.lower().strip()
    for label, opt_id in options.items():
        if v == label.lower():
            return opt_id
    for label, opt_id in options.items():
        if v in label.lower() or label.lower() in v:
            return opt_id
    return list(options.values())[0]


def ticket_agent(user: str, message: str, client, channel: str) -> str:
    state = _get_ticket_state(user)

    if message.lower().strip() in ("cancel", "abort", "stop"):
        _clear_ticket_state(user)
        return "❌ Ticket creation cancelled."

    if not state:
        last = _get_last_answer(user)
        if not last:
            return (
                "⚠️ I don't have a recent issue to raise a ticket for.\n"
                "_Please describe your issue first, then request a ticket._"
            )
        top_hit  = last["hits"][0] if last["hits"] else {}
        question = last["question"]
        kb_category = top_hit.get("bug_category", "Others")
        q_lower = question.lower()
        if "v2" in q_lower and "v1" in kb_category.lower():
            category = kb_category.replace("V1", "V2").replace("v1", "V2")
        elif "v1" in q_lower and "v2" in kb_category.lower():
            category = kb_category.replace("V2", "V1").replace("v2", "V1")
        else:
            category = kb_category
        severity = (top_hit.get("severity") or "medium").lower()
        if severity not in ("high", "medium", "low"):
            severity = "medium"
        state = {
            "title": question[:120], "category": category, "severity": severity,
            "description": last["answer"][:500], "reporter": None, "team": None,
            "environment": None, "notes": None, "step": "reporter",
        }
        _set_ticket_state(user, state)
        return (
            f"🎫 *Creating a ticket for:*\n_{state['title']}_\n\n"
            f"I already have:\n"
            f"   *Category:* {state['category']}\n"
            f"   *Severity:* {state['severity'].capitalize()}\n\n"
            f"📝 *Your name?* _(Reporter)_"
        )

    step = state.get("step")

    if step == "reporter":
        state["reporter"] = message.strip()
        state["step"]     = "team"
        _set_ticket_state(user, state)
        return (
            f"✅ Reporter: *{state['reporter']}*\n\n"
            f"📝 *Team / Department?*\n"
            f"_{' / '.join(TICKET_TEAM_OPTIONS.keys())}_"
        )
    if step == "team":
        state["team"] = message.strip()
        state["step"] = "environment"
        _set_ticket_state(user, state)
        return (
            f"✅ Team: *{state['team']}*\n\n"
            f"📝 *Environment?*\n"
            f"_{' / '.join(TICKET_ENV_OPTIONS.keys())}_"
        )
    if step == "environment":
        state["environment"] = message.strip()
        state["step"]        = "notes"
        _set_ticket_state(user, state)
        return (
            f"✅ Environment: *{state['environment']}*\n\n"
            f"📝 *Any additional notes?*\n"
            f"_Type your notes or *skip*_"
        )
    if step == "notes":
        state["notes"] = "" if message.lower().strip() == "skip" else message.strip()
        state["step"]  = "confirm"
        _set_ticket_state(user, state)
        notes_line = f"\n   *Notes:* {state['notes']}" if state["notes"] else ""
        return (
            f"🎫 *Ready to create ticket:*\n\n"
            f"   *Title:* {state['title']}\n"
            f"   *Category:* {state['category']}\n"
            f"   *Severity:* {state['severity'].capitalize()}\n"
            f"   *Reporter:* {state['reporter']}\n"
            f"   *Team:* {state['team']}\n"
            f"   *Environment:* {state['environment']}"
            f"{notes_line}\n\n"
            f"Type *confirm* to create or *cancel* to abort."
        )
    if step == "confirm":
        if message.lower().strip() in ("confirm", "yes", "ok", "y"):
            result = create_slack_list_ticket(state, client)
            _clear_ticket_state(user)
            return result
        else:
            _clear_ticket_state(user)
            return "❌ Ticket creation cancelled."

    return "⚠️ Something went wrong. Please try again."


def _rich_text_block(text: str) -> list:
    return [{"type": "rich_text", "elements": [{
        "type": "rich_text_section",
        "elements": [{"type": "text", "text": text}]
    }]}]


def create_slack_list_ticket(state: dict, client) -> str:
    import datetime
    ticket_id   = f"IRT-{int(datetime.datetime.now().timestamp())}"
    severity    = (state.get("severity") or "medium").lower()
    if severity not in ("high", "medium", "low"):
        severity = "medium"
    description = state["description"][:500]
    notes_text  = state.get("notes") or ""
    if notes_text:
        description += f"\n\nAdditional notes: {notes_text}"

    category_id = _map_to_option_id(state.get("category", "Others"), TICKET_CATEGORY_OPTIONS)
    team_id     = _map_to_option_id(state.get("team", "Testing"),     TICKET_TEAM_OPTIONS)
    env_id      = _map_to_option_id(state.get("environment", "AWS Production"), TICKET_ENV_OPTIONS)

    if not TICKET_LIST_ID:
        return (
            "⚠️ *Ticket list not configured.*\n"
            "_Please set `IRT_TICKET_LIST_ID` in your `.env` file._"
        )

    try:
        payload = {
            "list_id": TICKET_LIST_ID,
            "column_values": [
                {"column_id": "Col08B8D4N7NG", "rich_text": _rich_text_block(state["title"])},
                {"column_id": "Col08B32LLAUV", "select": ["new"]},
                {"column_id": "Col08BJJ5S3U1", "select": [severity]},
                {"column_id": "Col08CWKZS679", "select": [category_id]},
                {"column_id": "Col08D8APUF61", "select": [team_id]},
                {"column_id": "Col08DZVB9P0Q", "select": [env_id]},
                {"column_id": "Col08D4V7CF54", "rich_text": _rich_text_block(description)},
            ]
        }
        resp = client.api_call("slackLists.items.create", json=payload)
        if resp.get("ok"):
            item_id    = resp.get("item", {}).get("id", ticket_id)
            ticket_url = (
                f"https://app.slack.com/client/TJKT125D0"
                f"/unified-files/list/{TICKET_LIST_ID}/r/{item_id}"
            )
            return (
                f"✅ *Ticket raised successfully!*\n\n"
                f"   *ID:* `{item_id}`\n"
                f"   *Title:* {state['title']}\n"
                f"   *Severity:* {severity.capitalize()}\n"
                f"   *Reporter:* {state['reporter']}\n"
                f"   *Status:* New\n\n"
                f"<{ticket_url}|📋 Open Ticket>\n\n"
                f"_📎 Add screenshots in the ticket thread — it helps IRT resolve faster._"
            )
        err = resp.get("error", "unknown")
        log.error(f"Slack List creation failed: {err}")
        if err in ("max_items_reached", "list_full", "too_many_items"):
            list_url = f"https://app.slack.com/client/TJKT125D0/unified-files/list/{TICKET_LIST_ID}"
            return (
                f"⚠️ *Could not create ticket — the Bugs Tracker list is full.*\n\n"
                f"<{list_url}|📋 Open Bugs Tracker> to archive old items."
            )
        return (
            f"⚠️ *Ticket creation failed.*\n_Error: {err}_\n\n"
            f"Please try again or raise the ticket manually."
        )
    except Exception as e:
        log.error(f"Slack List ticket error: {e}")
        return (
            f"⚠️ *Something went wrong while creating the ticket.*\n"
            f"_Error: {str(e)[:120]}_\n\nPlease try again or raise manually."
        )


# ═════════════════════════════════════════════════════════════════════════════
# STREAM RESPONSE (core dispatcher — same structure as v7, wires in new agent)
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_auto_response(response: str) -> tuple:
    """
    automation_agent() returns either:
      - A plain string  → wrap in step_block
      - "__CONFIRM__:..." → render confirm/cancel buttons
    Returns (text, blocks).
    """
    if isinstance(response, str) and response.startswith("__CONFIRM__:"):
        body = response[len("__CONFIRM__:"):]
        return body, confirm_action_blocks(body)
    return response, step_block(response)


def stream_response(
    client,
    channel: str,
    query: str,
    thread_ts: str = None,
    ephemeral_user: str = None,
    user_id: str = None,
) -> None:

    # ── Reset ─────────────────────────────────────────────────────────────────
    if query.lower().strip() in ("reset", "clear", "new", "start over"):
        if user_id:
            _clear_history(user_id, channel)
            _clear_auto_state(user_id, channel)
            _clear_ticket_state(user_id)
        kw = {"channel": channel, "text": "🔄 Conversation reset. Ask me anything!"}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        client.chat_postMessage(**kw)
        return

    # ── Active ticket session ─────────────────────────────────────────────────
    ticket_state = _get_ticket_state(user_id) if user_id else None
    if ticket_state:
        response = ticket_agent(user_id, query, client, channel)
        kw = {"channel": channel, "text": response, "blocks": step_block(response)}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        client.chat_postMessage(**kw)
        return

    # ── Active automation session ─────────────────────────────────────────────
    auto_state = _get_auto_state(user_id, channel) if user_id else None
    if auto_state:
        auto_thread_ts = auto_state.get("thread_ts") or thread_ts
        if query.lower().strip() in ("cancel", "abort", "stop", "exit"):
            _clear_auto_state(user_id, channel)
            cancel_text = "❌ Automation cancelled. Ask me anything else!"
            kw = {"channel": channel, "text": cancel_text, "blocks": step_block(cancel_text)}
            if auto_thread_ts:
                kw["thread_ts"] = auto_thread_ts
            client.chat_postMessage(**kw)
            return
        response = automation_agent(user_id, channel, query)
        final_text, final_blocks = _resolve_auto_response(response)
        # Always post follow-ups inside the automation thread
        kw = {"channel": channel, "text": final_text, "blocks": final_blocks}
        if auto_thread_ts:
            kw["thread_ts"] = auto_thread_ts
        client.chat_postMessage(**kw)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # EPHEMERAL PATH
    # ══════════════════════════════════════════════════════════════════════════
    if ephemeral_user:
        try:
            history_channel = f"{channel}:{thread_ts}" if thread_ts else channel
            history  = _get_history(user_id, history_channel) if user_id else []
            decision = analyze_query(query, history)

            if decision["action"] == "chat":
                answer = handle_conversational(query, history)
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", answer)
                final_text, final_blocks = answer, step_block(answer)

            elif decision["action"] == "ticket":
                response = ticket_agent(user_id, query, client, channel)
                final_text, final_blocks = response, step_block(response)

            elif decision["action"] == "automate":
                cat_def  = decision.get("category_def") or detect_automation_from_kb(query)
                response = automation_agent(user_id, channel, query, category_def=cat_def)
                final_text, final_blocks = _resolve_auto_response(response)

            elif decision["action"] == "automateinfo":
                cat_def    = decision.get("category_def") or detect_automation_from_kb(query)
                final_text = automation_info_response(cat_def) if cat_def else "⚠️ Automation not found."
                final_blocks = step_block(final_text)

            elif decision["action"] == "outofscope":
                final_text = (
                    "⚠️ *This is outside my scope.*\n\n"
                    "I help with ConverSight product issues and automations.\n"
                    "For API keys, security, or account questions contact your admin."
                )
                final_blocks = step_block(final_text)

            elif decision["action"] == "clarify":
                clarification = decision["text"]
                suggestions   = decision.get("suggestions", [])
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
                hint = f"\n_Quick answers: {' · '.join(suggestions)}_" if suggestions else ""
                final_text   = f"🤔 {clarification}{hint}"
                final_blocks = step_block(final_text)

            else:  # search
                gpt_query = decision["text"]
                search_q  = gpt_query if gpt_query else query
                hits = search_kb(search_q)
                if not hits or hits[0]["score"] < MIN_SCORE:
                    final_text   = "❌ *No similar issues found.*\n\nThis may be a new issue. Please create a ticket."
                    final_blocks = step_block(final_text)
                else:
                    answer = generate_answer(search_q, hits, history)
                    if user_id:
                        _add_history(user_id, history_channel, "user", search_q)
                        _add_history(user_id, history_channel, "assistant", answer)
                        _save_last_answer(user_id, search_q, answer, hits)
                    final_text, final_blocks = answer, build_blocks(search_q, answer, hits)

        except Exception as e:
            log.error(f"stream_response (ephemeral) error: {e}")
            final_text   = f"⚠️ {_friendly_error(e)}"
            final_blocks = step_block(final_text)

        client.chat_postEphemeral(
            channel=channel, user=ephemeral_user,
            text=final_text, blocks=final_blocks
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC PATH — animated loading → replace with answer
    # ══════════════════════════════════════════════════════════════════════════
    kw = {"channel": channel, "text": STEPS[0], "blocks": step_block(STEPS[0])}
    if thread_ts:
        kw["thread_ts"] = thread_ts
    r      = client.chat_postMessage(**kw)
    msg_ts = r.get("ts")

    stop_flag = {"done": False}

    def animate():
        steps = STEPS[1:]
        idx   = 0
        while True:
            for _ in range(25):
                if stop_flag["done"]:
                    return
                time.sleep(0.1)
            if stop_flag["done"]:
                return
            try:
                client.chat_update(
                    channel=channel, ts=msg_ts,
                    text=steps[idx], blocks=step_block(steps[idx])
                )
            except Exception:
                pass
            idx = (idx + 1) % len(steps)

    anim = threading.Thread(target=animate, daemon=True)
    anim.start()

    final_text, final_blocks = "", []
    try:
        history_channel = f"{channel}:{thread_ts}" if thread_ts else channel
        history  = _get_history(user_id, history_channel) if user_id else []
        decision = analyze_query(query, history)

        if decision["action"] == "chat":
            answer = handle_conversational(query, history)
            if user_id:
                _add_history(user_id, history_channel, "user", query)
                _add_history(user_id, history_channel, "assistant", answer)
            final_text, final_blocks = answer, step_block(answer)

        elif decision["action"] == "ticket":
            stop_flag["done"] = True
            anim.join(timeout=1)
            try:
                client.chat_delete(channel=channel, ts=msg_ts)
            except Exception:
                pass
            response = ticket_agent(user_id, query, client, channel)
            is_pure_dm = channel.startswith("D")
            if is_pure_dm or thread_ts:
                kw = {"channel": channel, "text": response, "blocks": step_block(response)}
                if thread_ts:
                    kw["thread_ts"] = thread_ts
                client.chat_postMessage(**kw)
            else:
                anchor = client.chat_postMessage(
                    channel=channel, text=f"🎫 Ticket request: {query}",
                    blocks=[{"type": "section", "text": {"type": "mrkdwn",
                        "text": f"🎫 *Ticket request:* {query}"}}]
                )
                client.chat_postMessage(channel=channel, text=response,
                    blocks=step_block(response), thread_ts=anchor["ts"])
                ticket_s = _get_ticket_state(user_id) or {}
                ticket_s["thread_ts"] = anchor["ts"]
                _set_ticket_state(user_id, ticket_s)
            return

        elif decision["action"] == "automate":
            cat_def = decision.get("category_def") or detect_automation_from_kb(query)
            stop_flag["done"] = True
            anim.join(timeout=1)
            response = automation_agent(user_id, channel, query, category_def=cat_def)
            final_text, final_blocks = _resolve_auto_response(response)

            # Determine if this is a pure 1:1 DM (channel ID starts with D)
            # Group DMs (mpim) and channels both use thread-based flow
            is_pure_dm = channel.startswith("D")

            if is_pure_dm:
                # Pure DM — no threads, just update the loading message in place
                try:
                    client.chat_update(channel=channel, ts=msg_ts,
                        text=final_text, blocks=final_blocks)
                except Exception:
                    client.chat_postMessage(channel=channel,
                        text=final_text, blocks=final_blocks)
            else:
                # Channel or group DM — use threads so conversation stays clean
                # Update loading message → becomes the automation anchor with Close Thread button
                try:
                    client.chat_update(
                        channel = channel, ts = msg_ts,
                        text    = f"⚙️ Automation request: {query}",
                        blocks  = automation_anchor_blocks(query)
                    )
                except Exception:
                    pass
                # Post first bot reply as thread under the anchor
                client.chat_postMessage(channel=channel, text=final_text,
                    blocks=final_blocks, thread_ts=msg_ts)
                # Save msg_ts as thread_ts so all follow-ups stay in the same thread
                current_state = _get_auto_state(user_id, channel) or {}
                current_state["thread_ts"] = msg_ts
                _set_auto_state(user_id, channel, current_state)
            return

        elif decision["action"] == "automateinfo":
            cat_def    = decision.get("category_def") or detect_automation_from_kb(query)
            final_text = automation_info_response(cat_def) if cat_def else "⚠️ Automation not found."
            final_blocks = step_block(final_text)

        elif decision["action"] == "outofscope":
            final_text = (
                "⚠️ *This is outside my scope.*\n\n"
                "I help with ConverSight product issues and automations.\n"
                "For API keys, security, or account questions contact your admin."
            )
            final_blocks = step_block(final_text)

        elif decision["action"] == "clarify":
            clarification = decision["text"]
            suggestions   = decision.get("suggestions", [])
            stop_flag["done"] = True
            anim.join(timeout=1)
            is_pure_dm = channel.startswith("D")

            if is_pure_dm:
                try:
                    client.chat_delete(channel=channel, ts=msg_ts)
                except Exception:
                    pass
                client.chat_postMessage(
                    channel=channel,
                    text=f"🤔 {clarification}",
                    blocks=clarify_blocks(clarification, suggestions),
                )
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
            else:
                if thread_ts:
                    anchor_ts = thread_ts
                    try:
                        client.chat_delete(channel=channel, ts=msg_ts)
                    except Exception:
                        pass
                else:
                    try:
                        client.chat_delete(channel=channel, ts=msg_ts)
                    except Exception:
                        pass
                    anchor = client.chat_postMessage(
                        channel=channel, text=f"*Question:* {query}",
                        blocks=[{"type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Question:* {query}"}}]
                    )
                    anchor_ts = anchor["ts"]

                sent = client.chat_postMessage(
                    channel=channel, text=f"🤔 {clarification}",
                    blocks=clarify_blocks(clarification, suggestions),
                    thread_ts=anchor_ts,
                )
                if user_id:
                    _save_pending(ts=sent["ts"],   query=query, user=user_id, channel=channel)
                    _save_pending(ts=anchor_ts,    query=query, user=user_id, channel=channel)
                    _pending[sent["ts"]]["clarify_ts"] = sent["ts"]
                    _pending[anchor_ts]["clarify_ts"]  = sent["ts"]
                    _pending[sent["ts"]]["anchor_ts"]  = anchor_ts
                    _pending[anchor_ts]["anchor_ts"]   = anchor_ts
            return

        else:  # search
            gpt_query = decision["text"]
            search_q  = gpt_query if gpt_query else query
            log.warning(f"search_q='{search_q[:80]}'")
            hits = search_kb(search_q)
            if not hits or hits[0]["score"] < MIN_SCORE:
                final_text   = (
                    "❌ *No similar issues found.*\n\n"
                    "This may be a new issue. Please create a ticket in the Bug Tracker.\n"
                    "_Type another question or *reset* to start fresh._"
                )
                final_blocks = step_block(final_text)
            else:
                answer = generate_answer(search_q, hits, history)
                if user_id:
                    _add_history(user_id, history_channel, "user", search_q)
                    _add_history(user_id, history_channel, "assistant", answer)
                    _save_last_answer(user_id, search_q, answer, hits)
                final_text, final_blocks = answer, build_blocks(search_q, answer, hits)

    except Exception as e:
        log.error(f"stream_response error: {e}")
        final_text   = f"⚠️ {_friendly_error(e)}"
        final_blocks = step_block(final_text)

    stop_flag["done"] = True
    anim.join(timeout=2)

    try:
        client.chat_update(channel=channel, ts=msg_ts, text=final_text, blocks=final_blocks)
    except Exception as e:
        log.error(f"chat_update failed: {e}")
        kw = {"channel": channel, "text": final_text, "blocks": final_blocks}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        client.chat_postMessage(**kw)


# ═════════════════════════════════════════════════════════════════════════════
# MODAL VIEW
# ═════════════════════════════════════════════════════════════════════════════

def irt_modal_view(title="Ask IRT Bot", prefill=""):
    return {
        "type": "modal",
        "callback_id": "irt_modal_submit",
        "title": {"type": "plain_text", "text": title},
        "submit": {"type": "plain_text", "text": "🔍 Search"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text":
                    "*Search the IRT Knowledge Base* 🔍\n"
                    "Describe your issue or tell me what you want done."}
            },
            {
                "type": "input",
                "block_id": "query_block",
                "label": {"type": "plain_text", "text": "Your question / issue"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "query_input",
                    "multiline": True,
                    "initial_value": prefill,
                    "placeholder": {"type": "plain_text",
                        "text": "e.g. v2 dataset stuck, extend trial period for org_123…"}
                }
            },
            {
                "type": "input",
                "block_id": "visibility_block",
                "label": {"type": "plain_text", "text": "Who sees the answer?"},
                "element": {
                    "type": "static_select",
                    "action_id": "visibility_select",
                    "initial_option": {"text": {"type": "plain_text", "text": "Only me (test)"}, "value": "ephemeral"},
                    "options": [
                        {"text": {"type": "plain_text", "text": "Only me (test)"}, "value": "ephemeral"},
                        {"text": {"type": "plain_text", "text": "Whole channel"},  "value": "in_channel"},
                    ]
                }
            }
        ]
    }


# ═════════════════════════════════════════════════════════════════════════════
# SLACK EVENT HANDLERS (unchanged from v7)
# ═════════════════════════════════════════════════════════════════════════════

@app.command("/irt")
def handle_irt(ack, command, client):
    ack()
    query   = command.get("text", "").strip()
    channel = command.get("channel_id", "")
    user    = command.get("user_id", "")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="Please add a question. Example: `/irt v2 dataset failed`")
        return
    log.warning(f"/irt u={user} q={query[:80]}")
    threading.Thread(target=stream_response, args=(client, channel, query),
        kwargs={"user_id": user}, daemon=True).start()


@app.command("/irt-test")
def handle_irt_test(ack, command, client):
    ack()
    query   = command.get("text", "").strip()
    channel = command.get("channel_id", "")
    user    = command.get("user_id", "")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="🧪 Test mode — only you see this.\nUsage: `/irt-test extend trial for org_123`")
        return
    log.warning(f"/irt-test u={user} q={query[:80]}")
    threading.Thread(target=stream_response, args=(client, channel, query),
        kwargs={"ephemeral_user": user, "user_id": user}, daemon=True).start()


@app.shortcut("ask_irt_bot")
def open_irt_modal(ack, shortcut, client):
    ack()
    client.views_open(trigger_id=shortcut["trigger_id"], view=irt_modal_view())


@app.view("irt_modal_submit")
def handle_modal_submit(ack, body, client, view):
    ack()
    user       = body["user"]["id"]
    query      = view["state"]["values"]["query_block"]["query_input"]["value"].strip()
    visibility = view["state"]["values"]["visibility_block"]["visibility_select"]["selected_option"]["value"]
    ephem_user = user if visibility == "ephemeral" else None
    channel    = (body.get("channel") or {}).get("id") or IRT_CHANNEL
    log.warning(f"modal u={user} vis={visibility} q={query[:80]}")
    threading.Thread(target=stream_response, args=(client, channel, query),
        kwargs={"ephemeral_user": ephem_user, "user_id": user}, daemon=True).start()


@app.action(re.compile(r"clarify_reply(_\d+)?"))
def handle_clarify_reply(ack, body, client):
    ack()
    user      = body["user"]["id"]
    channel   = body["channel"]["id"]
    value     = body["actions"][0]["value"]
    msg_ts    = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts", msg_ts)

    pending = _get_pending(msg_ts) or _get_pending(thread_ts)
    if pending:
        _clear_pending(msg_ts)
        _clear_pending(thread_ts)
        search_input = build_enriched_query(pending["query"], value)
    else:
        search_input = value

    threading.Thread(target=stream_response, args=(client, channel, search_input),
        kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()


@app.action("ask_another")
def handle_ask_another(ack, body, client):
    ack()
    tid = body.get("trigger_id")
    if tid:
        try:
            client.views_open(trigger_id=tid, view=irt_modal_view(title="Ask another question"))
        except Exception as e:
            log.error(f"ask_another error: {e}")


@app.action("create_ticket")
def handle_create_ticket(ack):
    ack()


@app.action("automation_confirm")
def handle_automation_confirm(ack, body, client):
    """User clicked the ✅ Confirm button on the automation summary card."""
    ack()
    user      = body["user"]["id"]
    channel   = body["channel"]["id"]
    msg_ts    = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts", msg_ts)

    # Get the category name from active state for a meaningful message
    auto_state = _get_auto_state(user, channel)
    category   = (auto_state or {}).get("category_def", {}).get("category", "Automation")

    # Replace buttons immediately with an "executing" message
    executing_text = (
        f"⚙️ *Automation request has been submitted!*\n\n"
        f"*{category}* is now being executed — "
        f"this may take a few seconds. I'll update you once it's done."
    )
    try:
        client.chat_update(
            channel = channel,
            ts      = msg_ts,
            text    = executing_text,
            blocks  = [{"type": "section", "text": {
                "type": "mrkdwn", "text": executing_text
            }}]
        )
    except Exception:
        pass

    def _execute():
        response = automation_agent(user, channel, "confirm")
        final_text, final_blocks = _resolve_auto_response(response)
        if "completed successfully" in final_text:
            log.warning(f"automation_confirm: completed — user={user} category={category}")
        # Replace the executing message with the final result
        try:
            client.chat_update(channel=channel, ts=msg_ts,
                text=final_text, blocks=final_blocks)
        except Exception:
            kw = {"channel": channel, "text": final_text, "blocks": final_blocks}
            if thread_ts != msg_ts:
                kw["thread_ts"] = thread_ts
            client.chat_postMessage(**kw)

        # After success — keep thread open, reset session so user can run again
        # with different inputs in the same thread
        if "completed successfully" in final_text:
            # Get the current state (may be cleared by automation_agent on success)
            # Re-create a fresh session with same category but empty collected
            # so user can immediately provide new inputs without @mentioning again
            current_state = _get_auto_state(user, channel)
            if not current_state:
                # Session was cleared on success — restore it as fresh for re-use
                # We need the category_def — get it from automation_kb
                cat_def = detect_automation_from_kb(category)
                if cat_def:
                    fresh_state = {
                        "category_def":     cat_def,
                        "collected":        {},
                        "awaiting_confirm": False,
                        "owner":            user,
                        "closed":           False,
                        "thread_ts":        thread_ts if thread_ts != msg_ts else msg_ts,
                    }
                    _set_auto_state(user, channel, fresh_state)
                    log.warning(f"[AUTO] SESSION RESET FOR REUSE — user={user} category={category} thread_ts={thread_ts}")
                    # Prompt user they can run again
                    try:
                        client.chat_postMessage(
                            channel   = channel,
                            thread_ts = thread_ts if thread_ts != msg_ts else msg_ts,
                            text      = f"_You can run *{category}* again with different inputs, or click 🔴 Close Thread when done._",
                            blocks    = [{"type": "context", "elements": [{
                                "type": "mrkdwn",
                                "text": f"_You can run *{category}* again with different inputs, or click 🔴 Close Thread when done._"
                            }]}]
                        )
                    except Exception:
                        pass

    threading.Thread(target=_execute, daemon=True).start()


@app.action("automation_cancel")
def handle_automation_cancel(ack, body, client):
    """User clicked the ❌ Cancel button on the confirm card."""
    ack()
    user      = body["user"]["id"]
    channel   = body["channel"]["id"]
    msg_ts    = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts", msg_ts)

    # Get category name before clearing for log + message
    auto_state = _get_auto_state(user, channel)
    category   = (auto_state or {}).get("category_def", {}).get("category", "Automation")

    _clear_auto_state(user, channel)
    log.warning(f"automation_cancel: session cancelled — user={user} category={category}")

    cancel_text = f"❌ *{category}* cancelled.\n_To start again, mention @irtbot with your request._"
    try:
        client.chat_update(
            channel = channel, ts = msg_ts,
            text    = cancel_text,
            blocks  = step_block(cancel_text)
        )
    except Exception:
        kw = {"channel": channel, "text": cancel_text, "blocks": step_block(cancel_text)}
        if thread_ts != msg_ts:
            kw["thread_ts"] = thread_ts
        client.chat_postMessage(**kw)

    # Issue 3: Post a thread end log message so everyone in the thread sees it closed
    if thread_ts and thread_ts != msg_ts:
        try:
            client.chat_postMessage(
                channel   = channel,
                thread_ts = thread_ts,
                text      = f"🔚 *Thread ended* — {category} was cancelled by <@{user}>.",
                blocks    = [{"type": "context", "elements": [{
                    "type": "mrkdwn",
                    "text": f"🔚 *Thread ended* — _{category}_ was cancelled by <@{user}>. Start a new request by mentioning @irtbot."
                }]}]
            )
        except Exception:
            pass


@app.action("close_automation_thread")
def handle_close_automation_thread(ack, body, client):
    """
    User clicked 🔴 Close Thread on the automation anchor message.
    Only the thread owner (user who started the automation) can close it.
    """
    ack()
    user      = body["user"]["id"]
    channel   = body["channel"]["id"]
    msg_ts    = body["message"]["ts"]

    # Check if session is still active and who owns it
    auto_state   = _get_auto_state(user, channel)
    still_active = bool(auto_state)

    # ── Issue 3 fix: only the session owner can close the thread ─────────────
    # If the session belongs to a different user, block silently with ephemeral
    if not still_active:
        # Session already cleared (completed or closed) — check all users' states
        # to see if another user owns this thread
        for uid, state in _automation_state.items():
            if state and state.get("thread_ts") == msg_ts and uid != user:
                # Another user owns this thread — block this user
                try:
                    client.chat_postEphemeral(
                        channel = channel,
                        user    = user,
                        text    = "⚠️ You can only close your own automation threads.",
                        blocks  = [{"type": "section", "text": {"type": "mrkdwn",
                            "text": "⚠️ *You can only close your own automation threads.*"
                        }}]
                    )
                except Exception:
                    pass
                return

    # Get the original query from the anchor message
    original_text = ""
    try:
        blocks = body["message"].get("blocks", [])
        if blocks:
            original_text = blocks[0].get("text", {}).get("text", "")
            original_text = re.sub(r"⚙️ \*Automation request:\* ", "", original_text)
    except Exception:
        pass

    # Clear session if still active
    if still_active:
        _clear_auto_state(user, channel)
        _clear_ticket_state(user)
        log.warning(f"[AUTO] THREAD CLOSED — user={user} channel={channel} msg_ts={msg_ts}")

    closed_text = f"⚙️ *Automation request:* {original_text}" if original_text else "⚙️ Automation request"

    # Always update the anchor to remove the Close Thread button
    try:
        client.chat_update(
            channel = channel,
            ts      = msg_ts,
            text    = "🔴 Thread closed",
            blocks  = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": closed_text}
                },
                {
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"🔴 *Thread closed by <@{user}>* — session cleared. Start a new request by mentioning @irtbot."
                    }]
                }
            ]
        )
    except Exception as e:
        log.error(f"close_automation_thread update error: {e}")

    # Only post the closing message in thread if session was still active
    if still_active:
        try:
            client.chat_postMessage(
                channel   = channel,
                thread_ts = msg_ts,
                text      = f"🔴 *This thread has been closed by <@{user}>.*\n_Start a new request by mentioning @irtbot in the channel._",
                blocks    = [{"type": "section", "text": {"type": "mrkdwn",
                    "text": f"🔴 *This thread has been closed by <@{user}>.*\n_Start a new request by mentioning @irtbot in the channel._"
                }}]
            )
        except Exception as e:
            log.error(f"close_automation_thread post error: {e}")


@app.action("show_all_automations")
def handle_show_all_automations(ack, body, client):
    """User clicked 'See all automations' — post the full list in thread."""
    ack()
    channel   = body["channel"]["id"]
    msg_ts    = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts", msg_ts)

    try:
        all_cats = qclient.scroll(
            collection_name=AUTO_COLLECTION, limit=50, with_payload=True
        )[0]
        names = [p.payload.get("category", "") for p in all_cats if p.payload.get("category")]
    except Exception:
        names = []

    if not names:
        client.chat_postMessage(channel=channel, thread_ts=msg_ts,
            text="⚠️ No automation categories found.")
        return

    lines = "\n".join(f"   • {n}" for n in names)
    full_text = f"*⚙️ All available automations ({len(names)}):*\n\n{lines}\n\n_Just tell me which one you need and I'll guide you through it._"
    client.chat_postMessage(
        channel   = channel,
        thread_ts = msg_ts,
        text      = full_text,
        blocks    = [{"type": "section", "text": {"type": "mrkdwn", "text": full_text}}]
    )


# ── Dedup guard ───────────────────────────────────────────────────────────────
_processed: set = set()
_processed_lock = threading.Lock()

def _already_processed(ts: str) -> bool:
    with _processed_lock:
        if ts in _processed:
            return True
        _processed.add(ts)
        if len(_processed) > 200:
            oldest = list(_processed)[:100]
            for t in oldest:
                _processed.discard(t)
        return False


@app.event("message")
def handle_dm(event, client):
    if event.get("bot_id") or event.get("subtype"):
        return

    raw_text     = event.get("text", "") or ""
    channel      = event.get("channel", "")
    user         = event.get("user", "")
    channel_type = event.get("channel_type", "")
    thread_ts    = event.get("thread_ts")
    event_ts     = event.get("ts", "")

    # ── Skip any message that contains a @mention — handle_mention owns those ─
    if re.search(r"<@[A-Z0-9]+>", raw_text):
        return

    query = clean(raw_text).strip()
    if not query or not user:
        return

    if _already_processed(event_ts):
        return

    # ── Thread reply — check this BEFORE the channel_type DM check ───────────
    # This catches replies in group DMs (mpim) and channels equally
    if thread_ts:
        is_dm_type = channel_type in ("im", "mpim")

        if not is_dm_type:
            # Channel thread — only respond if bot owns this thread AND it's not closed
            auto_state = _get_auto_state(user, channel)
            if auto_state and auto_state.get("thread_ts") == thread_ts and not auto_state.get("closed"):
                threading.Thread(target=stream_response, args=(client, channel, query),
                    kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()
                return

            # Clarification pending for this thread
            pending = _get_pending(thread_ts)
            if pending and pending["user"] == user:
                log.warning(f"thread_reply u={user} reply='{query[:50]}'")
                _clear_pending(thread_ts)
                _clear_pending(pending.get("clarify_ts", ""))
                enriched = build_enriched_query(pending["query"], query)
                threading.Thread(target=stream_response, args=(client, channel, enriched),
                    kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()

            # No active session and no pending clarification — ignore completely
            # This prevents old thread replies from leaking into KB search
            return
        else:
            # DM or group DM thread reply — check active automation session
            auto_state = _get_auto_state(user, channel)
            if auto_state:
                auto_thread_ts = auto_state.get("thread_ts") or thread_ts
                threading.Thread(target=stream_response, args=(client, channel, query),
                    kwargs={"thread_ts": auto_thread_ts, "user_id": user}, daemon=True).start()
                return
            # No active session — treat as plain DM message (fall through below)

    # ── Only respond to actual DMs and group DMs ──────────────────────────────
    if channel_type not in ("im", "mpim"):
        return

    log.warning(f"DM u={user} q={query[:80]}")
    threading.Thread(target=stream_response, args=(client, channel, query),
        kwargs={"user_id": user}, daemon=True).start()


def welcome_blocks(user: str) -> list:
    """
    Rich welcome card shown when someone @mentions the bot with no query,
    or says hi/hello. Shows what the bot can do with example prompts.
    """
    # Dynamically fetch automation category names from KB for the examples
    auto_examples = [
        "Extend Trial Period", "Activate Dataset",
        "Enable Athena IQ", "Increase User Count",
    ]
    try:
        if auto_count > 0:
            all_cats = qclient.scroll(
                collection_name=AUTO_COLLECTION, limit=50, with_payload=True
            )[0]
            names = [p.payload.get("category", "") for p in all_cats if p.payload.get("category")]
            if names:
                auto_examples = names[:4]
    except Exception:
        pass

    auto_list = "\n".join(f"   • {c}" for c in auto_examples)
    if auto_count > len(auto_examples):
        auto_list += f"\n   _...and {auto_count - len(auto_examples)} more_"

    return [
        # ── Header ────────────────────────────────────────────────────────────
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "👋 Hey! I'm IRT Bot", "emoji": True}
        },

        # ── Intro ─────────────────────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Hi <@{user}>! I'm the *Conversight Immediate Response Team assistant*.\n"
                    "Here's what I can do for you:"
                )
            }
        },

        {"type": "divider"},

        # ── Feature 1: Diagnose issues ────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*🔍 Diagnose & resolve product issues*\n"
                    "Describe any ConverSight problem and I'll search past incidents "
                    "to find what worked.\n\n"
                    "*Try:*\n"
                    "   `@irtbot v2 dataset stuck in DictionaryRequested`\n"
                    "   `@irtbot dataload failed in production`\n"
                    "   `@irtbot notebook not launching`\n"
                    "   `@irtbot SME publish failed`"
                )
            }
        },

        {"type": "divider"},

        # ── Feature 2: Automations ────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*⚙️ Run automations* _{auto_count} available_\n"
                    "I can execute IRT operations step-by-step — "
                    "just tell me what you need done.\n\n"
                    f"*Available actions:*\n{auto_list}\n\n"
                    "*Try:*\n"
                    "   `@irtbot extend trial period for org_123`\n"
                    "   `@irtbot activate dataset ds_456`"
                )
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "See all automations", "emoji": True},
                "action_id": "show_all_automations",
                "value": "show_all",
            }
        },

        {"type": "divider"},

        # ── Feature 3: Tickets ────────────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*🎫 Raise a ticket*\n"
                    "After I answer your question, just say *create ticket* "
                    "and I'll pre-fill a bug report from your issue.\n\n"
                    "*Try:*\n"
                    "   `@irtbot create ticket`  _(after describing an issue)_"
                )
            }
        },

        {"type": "divider"},

        # ── Tip ───────────────────────────────────────────────────────────────
        {
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (
                    "💡 *Tip:* You can also *DM me directly* for a private conversation "
                    "with full chat memory.  Type *reset* anytime to start fresh."
                )
            }]
        },
    ]


# ── Greeting phrases that should show the welcome card ───────────────────────
_GREETING_PHRASES = {
    "hi", "hello", "hey", "hii", "helo", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "what can you do", "what do you do", "who are you", "what are you",
    "help", "how does this work", "tell me about yourself",
    "capabilities", "features", "what can you help with",
}

def _is_greeting(text: str) -> bool:
    q = text.lower().strip().rstrip("!?.,")
    return q in _GREETING_PHRASES or len(text.split()) <= 2


def _open_thread_warning(user: str, channel: str, client) -> bool:
    """
    Checks if the user has an open automation thread in this channel.
    If yes → posts an ephemeral warning (only visible to that user) with a
             link to their open thread, and returns True (caller should return).
    If no  → returns False (caller proceeds normally).
    """
    auto_state = _get_auto_state(user, channel)
    if not auto_state or not auto_state.get("thread_ts"):
        return False

    thread_ts  = auto_state["thread_ts"]
    category   = auto_state.get("category_def", {}).get("category", "Automation")

    # Build a Slack deep-link to the thread
    # Format: https://slack.com/archives/{channel}/p{ts_without_dot}
    ts_clean   = thread_ts.replace(".", "")
    thread_url = f"https://slack.com/archives/{channel}/p{ts_clean}"

    warning_blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"⚠️ *You already have an open automation thread.*\n\n"
                    f"*{category}* is still in progress. "
                    f"Please complete or close it before starting a new request.\n\n"
                    f"<{thread_url}|👉 Go to your open thread>"
                )
            }
        }
    ]

    try:
        client.chat_postEphemeral(
            channel = channel,
            user    = user,
            text    = "⚠️ You have an open automation thread. Please complete or close it first.",
            blocks  = warning_blocks,
        )
    except Exception as e:
        log.error(f"_open_thread_warning ephemeral error: {e}")

    return True


@app.event("app_mention")
def handle_mention(event, client):
    text      = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
    channel   = event.get("channel", "")
    user      = event.get("user", "")
    ts        = event.get("ts")
    thread_ts = event.get("thread_ts")

    # Mark processed FIRST to block handle_dm from also firing
    if ts and _already_processed(ts):
        return

    if thread_ts:
        # ── Active automation session in this exact thread → route directly ───
        auto_state = _get_auto_state(user, channel)
        if auto_state and auto_state.get("thread_ts") == thread_ts and not auto_state.get("closed"):
            threading.Thread(target=stream_response, args=(client, channel, text),
                kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()
            return

        # ── Clarification pending → enrich and search ─────────────────────────
        pending = _get_pending(thread_ts)
        if pending and pending["user"] == user:
            _clear_pending(thread_ts)
            enriched = build_enriched_query(pending["query"], text)
            threading.Thread(target=stream_response, args=(client, channel, enriched),
                kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()
            return

        # ── Regular thread follow-up (no active session, no pending) ──────────
        # If this is a reply in an old/closed automation thread — ignore it
        # Only respond if it's a genuinely fresh thread interaction
        # (e.g. user @mentions bot inside a non-automation thread for a new question)
        if not auto_state:
            threading.Thread(target=stream_response, args=(client, channel, text),
                kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()
        return

    # ── Fresh top-level mention in the channel ────────────────────────────────
    # Clear any stale automation session — closed or abandoned ones should
    # not block or redirect new fresh requests into old threads.
    auto_state = _get_auto_state(user, channel)
    if auto_state and auto_state.get("thread_ts"):
        # Only clear if the thread is closed OR the session has no awaiting activity
        if auto_state.get("closed") or not auto_state.get("awaiting_confirm"):
            log.warning(f"handle_mention: clearing stale/closed auto session for u={user}")
            _clear_auto_state(user, channel)

    if _get_ticket_state(user):
        _clear_ticket_state(user)

    # ── No query or greeting → show rich welcome card in channel ─────────────
    if not text or _is_greeting(text):
        client.chat_postMessage(
            channel = channel,
            text    = f"Hi <@{user}>! Here's what I can do.",
            blocks  = welcome_blocks(user),
        )
        return

    threading.Thread(target=stream_response, args=(client, channel, text),
        kwargs={"thread_ts": ts, "user_id": user}, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    missing = [k for k, v in [
        ("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN),
        ("SLACK_APP_TOKEN", SLACK_APP_TOKEN),
        ("OPENAI_API_KEY",  os.environ.get("OPENAI_API_KEY")),
    ] if not v]
    if missing:
        [print(f"❌  {k} missing in .env") for k in missing]
        exit(1)

    print()
    print("=" * 65)
    print("  🤖  IRT RAG Slack Bot v8  — KB-Driven Automation Agent")
    print("=" * 65)
    print(f"  /irt <question>      → visible to whole channel")
    print(f"  /irt-test <question> → only you see it")
    print(f"  Ask IRT Bot button   → modal + live loading")
    print(f"  DM the bot           → chatbot with memory")
    print(f"  @mention bot         → reply in thread")
    print(f"  Automation           → {auto_count} categories from automation_kb")
    print(f"  Knowledge base       : {kb_count:,} documents")
    print(f"  Ticket list          : {'✅ ' + TICKET_LIST_ID if TICKET_LIST_ID else '❌ missing IRT_TICKET_LIST_ID'}")
    print(f"  Chat memory          : last {CHAT_HISTORY_LEN} turns per user")
    print()
    if auto_count == 0:
        print("  ⚠️  Run:  python load_automation_kb.py  to load automation categories")
    print("=" * 65)
    print()

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
