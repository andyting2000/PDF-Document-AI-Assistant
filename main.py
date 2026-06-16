from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import io
import pypdf
import psycopg2
from urllib import parse as urlparse
from urllib import error as urlerror
from urllib import request as urlrequest
import re
import socket
import time
import json
import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.delete("/delete-pdf")
def delete_pdf(filename: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ensure_pdf_files_table(cur)
        # Remove from pdf_files
        cur.execute("DELETE FROM pdf_files WHERE filename = %s;", (filename,))
        # Remove all embeddings for this PDF
        cur.execute(
            "DELETE FROM documents WHERE source_filename = %s;", (filename,))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "filename": filename}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


def load_env_file(file_name: str = ".env"):
    if not os.path.exists(file_name):
        return

    with open(file_name, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

HF_API_KEY = os.getenv("HF_API_KEY", "")
if not HF_API_KEY:
    raise RuntimeError("Missing HF_API_KEY. Add it to your .env file.")

HF_CHAT_COMPLETIONS_URL = os.getenv(
    "HF_CHAT_COMPLETIONS_URL",
    "https://router.huggingface.co/v1/chat/completions"
)
HF_EMBEDDINGS_URL = os.getenv(
    "HF_EMBEDDINGS_URL",
    "https://router.huggingface.co/v1/embeddings"
)
DEFAULT_HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model}"
HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE = os.getenv(
    "HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE",
    DEFAULT_HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE
)
HF_PROVIDER_POLICY = os.getenv("HF_PROVIDER_POLICY", "preferred").strip()

DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "math0619")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

DEFAULT_TEXT_MODEL = "openai/gpt-oss-120b"

CURATED_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def apply_provider_policy(model_name: str) -> str:
    model_name = model_name.strip()
    if not model_name or ":" in model_name or not HF_PROVIDER_POLICY:
        return model_name
    return f"{model_name}:{HF_PROVIDER_POLICY}"


TEXT_MODEL_CANDIDATES = [
    apply_provider_policy(model.strip()) for model in os.getenv(
        "HF_TEXT_MODELS",
        f"{DEFAULT_TEXT_MODEL},Qwen/Qwen3-235B-A22B-Instruct-2507,meta-llama/Llama-3.1-8B-Instruct"
    ).split(",") if model.strip()
]


TEXT_TIMEOUT_SECONDS = int(os.getenv("TEXT_TIMEOUT_SECONDS", "60"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBEDDING_TIMEOUT_SECONDS = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_PER_FILE_K = int(os.getenv("RETRIEVAL_PER_FILE_K", "2"))
EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
).strip()
DEFAULT_EMBEDDING_MODELS = f"{EMBEDDING_MODEL},thenlper/gte-large"
EMBEDDING_MODEL_CANDIDATES = [
    model.strip()
    for model in os.getenv("HF_EMBEDDING_MODELS", DEFAULT_EMBEDDING_MODELS).split(",")
    if model.strip()
]


def ensure_pdf_files_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_files (
            id SERIAL PRIMARY KEY,
            filename TEXT UNIQUE,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


_EMBEDDING_BACKEND_LOGGED = set()
_UNAVAILABLE_EMBEDDING_MODELS = set()


def _log_embedding_backend_once(model_name: str, endpoint_label: str, target_dimensions: int):
    key = (model_name, endpoint_label, target_dimensions)
    if key in _EMBEDDING_BACKEND_LOGGED:
        return
    _EMBEDDING_BACKEND_LOGGED.add(key)
    print(
        f"✅ Embedding backend in use: model={model_name} endpoint={endpoint_label} dimensions={target_dimensions}"
    )


def _is_retryable_model_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    retryable_markers = [
        "model_not_found",
        "model_decommissioned",
        "http 404",
        "not found",
        "does not exist",
        "no longer supported",
        "do not have access",
        "not available",
        "unsupported",
        "http 403",
        "access denied",
        "cloudflare",
        "banned your access",
    ]
    return any(marker in error_text for marker in retryable_markers)


def _compact_error_message(message: str, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", message).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True

    if isinstance(exc, urlerror.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)):
        return True

    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return True

    reason = getattr(exc, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    if reason and reason is not exc and _is_timeout_error(reason):
        return True

    cause = getattr(exc, "__cause__", None)
    if cause and cause is not exc and _is_timeout_error(cause):
        return True

    return False


def _extract_error_message(body_text: str, status_code: int) -> str:
    body_text = body_text or ""
    lowered = body_text.lower()

    if "<html" in lowered and ("cloudflare" in lowered or "access denied" in lowered):
        return f"HTTP {status_code}: Access denied by provider firewall (Cloudflare 1010)."

    try:
        parsed = json.loads(body_text)
    except Exception:
        return f"HTTP {status_code}: {_compact_error_message(body_text, max_len=280)}"

    if isinstance(parsed, dict):
        if isinstance(parsed.get("error"), dict):
            msg = parsed["error"].get("message")
            if msg:
                return f"HTTP {status_code}: {msg}"
        if parsed.get("message"):
            return f"HTTP {status_code}: {parsed['message']}"

    return f"HTTP {status_code}: {_compact_error_message(body_text, max_len=280)}"


def _hf_chat_completion(model: str, messages, **kwargs):
    timeout_seconds = kwargs.pop("timeout_seconds", 180)

    payload = {
        "model": model,
        "messages": messages,
    }

    for key in [
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ]:
        if key in kwargs and kwargs[key] is not None:
            payload[key] = kwargs[key]

    req = urlrequest.Request(
        HF_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as http_err:
        body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_extract_error_message(body, http_err.code))


def _hf_embedding_batch(text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS) -> List[List[float]]:
    return _hf_embedding_batch_with_model(EMBEDDING_MODEL, text_batch, timeout_seconds)


def _embedding_model_variants(model_name: str) -> List[str]:
    raw = (model_name or "").strip()
    if not raw:
        return []
    # For embedding endpoints, provider suffix variants (e.g. :preferred) often cause 404.
    return [raw]


def _validate_embedding_vectors(vectors: List[List[float]], text_batch: List[str], target_dimensions: int) -> List[List[float]]:
    resized_vectors: List[List[float]] = []
    for embedding in vectors:
        if target_dimensions:
            if len(embedding) < target_dimensions:
                # Keep dimensionality consistent with pgvector schema by right-padding with zeros.
                embedding = embedding + [0.0] * \
                    (target_dimensions - len(embedding))
            if len(embedding) > target_dimensions:
                # Keep vector size consistent with pgvector schema and query vectors.
                embedding = embedding[:target_dimensions]

        resized_vectors.append(embedding)

    if len(resized_vectors) != len(text_batch):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(resized_vectors)} vectors for {len(text_batch)} texts."
        )

    return resized_vectors


def _hf_embedding_batch_router(model_name: str, text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    payload = {
        "model": model_name,
        "input": text_batch,
        "dimensions": target_dimensions,
    }

    req = urlrequest.Request(
        HF_EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as http_err:
        body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_extract_error_message(body, http_err.code))

    vectors: List[List[float]] = []
    for item in data.get("data", []):
        embedding = [float(x) for x in item.get("embedding", [])]
        vectors.append(embedding)

    _log_embedding_backend_once(
        model_name, "router:/v1/embeddings", target_dimensions)

    return _validate_embedding_vectors(vectors, text_batch, target_dimensions)


def _mean_pool_token_embeddings(token_vectors: List[List[float]]) -> List[float]:
    if not token_vectors:
        return []

    dim = len(token_vectors[0])
    sums = [0.0] * dim
    count = 0
    for row in token_vectors:
        if not isinstance(row, list) or len(row) != dim:
            continue
        for idx, value in enumerate(row):
            sums[idx] += float(value)
        count += 1

    if count == 0:
        return []

    return [value / count for value in sums]


def _hf_embedding_batch_hf_inference(model_name: str, text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    # hf-inference route expects repository id, not provider suffix (e.g., ':preferred').
    base_model = model_name.split(":", 1)[0].strip()
    encoded_model = urlparse.quote(base_model, safe="/")
    template = HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE
    if "{model}" not in template or "/hf-inference/models/" not in template:
        print(
            "⚠️ Invalid HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE; using default router hf-inference model endpoint."
        )
        template = DEFAULT_HF_EMBEDDING_HF_INFERENCE_URL_TEMPLATE

    inference_url = template.format(
        model=encoded_model)
    payload = {
        "inputs": text_batch if len(text_batch) > 1 else text_batch[0],
        "options": {
            "wait_for_model": True,
        },
    }

    req = urlrequest.Request(
        inference_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as http_err:
        body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_extract_error_message(body, http_err.code))

    vectors: List[List[float]] = []
    if isinstance(data, list):
        # Single vector: [f1, f2, ...]
        if data and isinstance(data[0], (int, float)):
            vectors = [[float(x) for x in data]]
        # Batch vectors: [[...], [...]]
        elif data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
            vectors = [[float(x) for x in row] for row in data]
        # Token-level vectors: [[[...], [...]], ...] -> mean-pool per input
        elif data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            for token_vectors in data:
                pooled = _mean_pool_token_embeddings(token_vectors)
                if not pooled:
                    raise RuntimeError(
                        "Could not parse token-level embedding response.")
                vectors.append(pooled)

    if not vectors:
        raise RuntimeError(
            f"Unexpected hf-inference embedding response shape: {data}")

    _log_embedding_backend_once(
        model_name, "router:/hf-inference/models/{model}", target_dimensions)

    return _validate_embedding_vectors(vectors, text_batch, target_dimensions)


def _hf_embedding_batch_with_model(model_name: str, text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    try:
        return _hf_embedding_batch_hf_inference(model_name, text_batch, timeout_seconds, target_dimensions)
    except Exception as inference_error:
        if not _is_retryable_model_error(inference_error):
            raise

        compact_error = _compact_error_message(str(inference_error))
        print(
            f"⚠️ Router hf-inference endpoint failed for {model_name}; trying /v1/embeddings fallback. Error: {compact_error}"
        )
        return _hf_embedding_batch_router(model_name, text_batch, timeout_seconds, target_dimensions)


def create_embedding_with_fallback(text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    if not EMBEDDING_MODEL_CANDIDATES:
        raise RuntimeError("No embedding model candidates configured.")

    expanded_candidates: List[str] = []
    for base_model in EMBEDDING_MODEL_CANDIDATES:
        for variant in _embedding_model_variants(base_model):
            if variant not in expanded_candidates:
                expanded_candidates.append(variant)

    available_candidates = [
        model_name for model_name in expanded_candidates
        if model_name not in _UNAVAILABLE_EMBEDDING_MODELS
    ]
    if available_candidates:
        expanded_candidates = available_candidates

    last_error = None
    for idx, model_name in enumerate(expanded_candidates):
        try:
            return _hf_embedding_batch_with_model(model_name, text_batch, timeout_seconds, target_dimensions)
        except Exception as e:
            last_error = e
            compact_error = _compact_error_message(str(e))
            if _is_retryable_model_error(e):
                _UNAVAILABLE_EMBEDDING_MODELS.add(model_name)
                next_model = expanded_candidates[idx + 1] if idx + \
                    1 < len(expanded_candidates) else None
                if next_model:
                    print(
                        f"⚠️ Embedding model unavailable ({model_name}); trying fallback: {next_model}. Error: {compact_error}"
                    )
                    continue
            raise

    raise RuntimeError(
        "No working embedding models from candidates: "
        f"{expanded_candidates}. Last error: {last_error}. "
        "Tip: prefer documented HF Inference feature-extraction models such as "
        "intfloat/multilingual-e5-large or thenlper/gte-large in HF_EMBEDDING_MODELS."
    )


def embed_text_batch_with_backoff(text_batch: List[str], timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    try:
        return create_embedding_with_fallback(text_batch, timeout_seconds, target_dimensions)
    except Exception as exc:
        if not _is_timeout_error(exc) or len(text_batch) <= 1:
            raise

        split_point = max(1, len(text_batch) // 2)
        print(
            f"⚠️ Embedding batch of {len(text_batch)} texts timed out after {timeout_seconds}s; retrying as {split_point} + {len(text_batch) - split_point}."
        )
        left_vectors = embed_text_batch_with_backoff(
            text_batch[:split_point],
            timeout_seconds,
            target_dimensions,
        )
        right_vectors = embed_text_batch_with_backoff(
            text_batch[split_point:],
            timeout_seconds,
            target_dimensions,
        )
        return left_vectors + right_vectors


async def embed_texts_async(texts: List[str], batch_size: int = EMBED_BATCH_SIZE, target_dimensions: int = EMBEDDING_DIMENSIONS) -> List[List[float]]:
    if not texts:
        return []

    all_vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vectors = await asyncio.to_thread(
            embed_text_batch_with_backoff,
            batch,
            EMBEDDING_TIMEOUT_SECONDS,
            target_dimensions,
        )
        all_vectors.extend(batch_vectors)

    return all_vectors


def create_chat_completion_with_fallback(model_candidates, **kwargs):
    last_error = None
    for idx, model in enumerate(model_candidates):
        try:
            return _hf_chat_completion(model=model, **kwargs)
        except Exception as e:
            last_error = e
            compact_error = _compact_error_message(str(e))
            if _is_retryable_model_error(e):
                next_model = model_candidates[idx + 1] if idx + \
                    1 < len(model_candidates) else None
                if next_model:
                    print(
                        f"⚠️ Model unavailable ({model}); trying fallback: {next_model}. Error: {compact_error}"
                    )
                    continue
                print(
                    f"⚠️ Model unavailable and no fallbacks left ({model}). Error: {compact_error}")
                continue
            raise

    raise RuntimeError(
        f"No working models from candidates: {model_candidates}. Last error: {last_error}"
    )


def _completion_text(completion_payload: dict) -> str:
    try:
        content = completion_payload["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            f"Unexpected Hugging Face response shape: {completion_payload}") from e

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join([part for part in parts if part])

    return str(content)


def resolve_text_models(selected_model: Optional[str]) -> List[str]:
    model = apply_provider_policy((selected_model or "").strip())
    if not model:
        return TEXT_MODEL_CANDIDATES

    # Prioritize user-selected model, then keep configured fallbacks.
    ordered = [model] + [m for m in TEXT_MODEL_CANDIDATES if m != model]
    return ordered


def get_ui_text_models() -> List[str]:
    # Keep frontend options stable with 3 curated models.
    base_candidates = [m.split(":", 1)[0] for m in TEXT_MODEL_CANDIDATES]
    configured = [m for m in CURATED_TEXT_MODELS if m in base_candidates]
    for m in CURATED_TEXT_MODELS:
        if m not in configured:
            configured.append(m)
    return configured[:3]


def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def get_documents_embedding_dimension(cur) -> Optional[int]:
    cur.execute(
        """
        SELECT atttypmod, format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'documents'::regclass
          AND attname = 'embedding'
          AND attnum > 0
          AND NOT attisdropped
        LIMIT 1;
        """
    )
    row = cur.fetchone()
    if not row:
        return None

    atttypmod, formatted_type = row

    # Most reliable path: parse declared type text, e.g. "vector(384)".
    if isinstance(formatted_type, str):
        match = re.search(r"vector\((\d+)\)", formatted_type)
        if match:
            return int(match.group(1))

    # Fallback path for environments where format_type is unavailable/unexpected.
    if atttypmod is None:
        return None

    # Some pgvector builds encode typmod directly as n, others as n+4.
    if atttypmod > 4:
        return int(atttypmod - 4)
    if atttypmod > 0:
        return int(atttypmod)
    return None


def ensure_documents_metadata_columns(cur):
    # Backward-compatible metadata migration for existing documents table.
    cur.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_filename TEXT;")
    cur.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_page INTEGER;")
    cur.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_chunk INTEGER;")


def extract_pages_from_pdf(file_bytes) -> List[tuple]:
    pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages: List[tuple] = []
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append((page_number, page_text))
    return pages


def extract_text_from_pdf(file_bytes):
    pages = [page_text for _, page_text in extract_pages_from_pdf(file_bytes)]
    return "\n\n".join(pages)


def _normalize_pdf_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Merge words broken by line-wrap hyphenation: "state-of-the-\nart" -> "state-of-the-art"
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    # Keep paragraph boundaries while collapsing intra-paragraph hard wraps.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    sentence_like = re.split(r"(?<=[.!?])\s+", paragraph)
    sentence_like = [s.strip() for s in sentence_like if s and s.strip()]

    if not sentence_like:
        return []

    parts = []
    current = ""
    for sent in sentence_like:
        if len(sent) > max_chars:
            if current:
                parts.append(current)
                current = ""
            for i in range(0, len(sent), max_chars):
                parts.append(sent[i:i + max_chars].strip())
            continue

        candidate = sent if not current else f"{current} {sent}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sent

    if current:
        parts.append(current)

    return [p for p in parts if p.strip()]


def chunk_text(text, target_chars=1200, overlap_chars=220, min_chars=120):
    normalized = _normalize_pdf_text(text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(
        r"\n\n+", normalized) if p and p.strip()]
    units = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, target_chars))

    chunks = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        candidate = f"{current}\n\n{unit}"
        if len(candidate) <= target_chars:
            current = candidate
            continue

        chunks.append(current)

        # Carry lexical continuity into the next chunk for better retrieval recall.
        overlap_tail = current[-overlap_chars:].strip()
        if overlap_tail:
            current = f"{overlap_tail}\n\n{unit}"
        else:
            current = unit

        if len(current) > target_chars:
            oversized_parts = _split_long_paragraph(
                current.replace("\n\n", " "), target_chars)
            if oversized_parts:
                chunks.extend(oversized_parts[:-1])
                current = oversized_parts[-1]

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) >= min_chars]


@app.get("/debug-similarity")
async def debug_similarity(q: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ensure_documents_metadata_columns(cur)
        db_embedding_dimensions = get_documents_embedding_dimension(
            cur) or EMBEDDING_DIMENSIONS
        query_vector = (await embed_texts_async([q], target_dimensions=db_embedding_dimensions))[0]
        cur.execute("""
            SELECT content, embedding, source_filename, source_page, source_chunk,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT 3;
        """, (query_vector, query_vector))

        results = cur.fetchall()
        cur.close()
        conn.close()

        matches = []
        for r in results:
            vec_data = r[1]
            if isinstance(vec_data, str):
                vec_data = [float(x) for x in vec_data.strip('[]').split(',')]

            matches.append({
                "text": r[0],
                "vector": vec_data,
                "source_filename": r[2],
                "source_page": r[3],
                "source_chunk": r[4],
                "score": float(r[5])
            })

        return {"question_vector": query_vector, "matches": matches}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def admin_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vector Embedding Inspector</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background: #0d1117; color: #c9d1d9; }
            .container { max-width: 900px; margin: auto; background: #161b22; padding: 30px; border-radius: 10px; border: 1px solid #30363d; }
            input { width: 70%; padding: 12px; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: white; margin-right: 10px; }
            button { padding: 12px 24px; background: #238636; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
            button:hover { background: #2ea043; }
            .vector-box { background: #0d1117; color: #58a6ff; padding: 15px; border-radius: 6px; font-family: monospace; overflow-x: auto; white-space: nowrap; border: 1px solid #30363d; margin-top: 10px; }
            .result-card { background: #21262d; border: 1px solid #30363d; padding: 20px; margin-top: 20px; border-radius: 8px; }
            .score { font-size: 1.4em; font-weight: bold; color: #3fb950; float: right; }
            h2, h3 { color: #58a6ff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Vector Embedding Inspector</h2>
            <p>Type a question to calculate the Cosine Similarity between your query and the PDF chunks stored in PostgreSQL.</p>
            
            <div style="display: flex; margin-bottom: 30px;">
                <input type="text" id="query" placeholder="E.g., What is the total revenue?">
                <button onclick="inspectMath()">Calculate Similarity</button>
            </div>
            
            <div id="loading" style="display: none; color: #8b949e;">Calculating embedding similarity...</div>
            <div id="results"></div>
        </div>

        <script>
            async function inspectMath() {
                const q = document.getElementById('query').value;
                if (!q) return;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                const response = await fetch('/debug-similarity?q=' + encodeURIComponent(q));
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                // Show Question Vector
                let html = `<h3>1. Your Question's Embedding (First 20 dimensions)</h3>
                            <div class="vector-box">[${data.question_vector.slice(0, 20).map(n => n.toFixed(4)).join(', ')} ... ]</div>
                            <h3 style="margin-top: 40px;">2. Top Database Matches</h3>`;
                            
                // Show Chunk Vectors and Similarity Score
                data.matches.forEach((match, index) => {
                    const percentage = (match.score * 100).toFixed(2);
                    html += `
                    <div class="result-card">
                        <span class="score">${percentage}% Match</span>
                        <h4 style="margin-top: 0;">Rank ${index + 1}</h4>
                        <p style="color: #8b949e; line-height: 1.5;"><strong>PDF Chunk:</strong><br> "${match.text}"</p>
                        <p><strong>Chunk's Embedding:</strong></p>
                        <div class="vector-box">[${match.vector.slice(0, 20).map(n => n.toFixed(4)).join(', ')} ... ]</div>
                    </div>`;
                });
                
                document.getElementById('results').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ensure_documents_metadata_columns(cur)
        ensure_pdf_files_table(cur)
        db_embedding_dimensions = get_documents_embedding_dimension(
            cur) or EMBEDDING_DIMENSIONS

        async def process_single_pdf(file: UploadFile):
            file_start = time.perf_counter()
            content = await file.read()
            print(f"\n📄 Processing {file.filename}...")
            page_items = await asyncio.to_thread(extract_pages_from_pdf, content)
            chunk_items = []
            for page_number, page_text in page_items:
                chunks = chunk_text(page_text)
                for page_chunk_idx, chunk in enumerate(chunks, start=1):
                    if len(chunk.strip()) >= 10:
                        chunk_items.append({
                            "content": chunk,
                            "source_page": page_number,
                            "source_chunk": page_chunk_idx,
                        })
            rows = []
            if chunk_items:
                vectors = await embed_texts_async(
                    [item["content"] for item in chunk_items],
                    target_dimensions=db_embedding_dimensions,
                )
                rows = [
                    (
                        item["content"],
                        vector,
                        file.filename,
                        item["source_page"],
                        item["source_chunk"],
                    )
                    for item, vector in zip(chunk_items, vectors)
                ]
            elapsed = time.perf_counter() - file_start
            print(
                f"✅ Finished {file.filename} in {elapsed:.2f}s. Chunks saved: {len(rows)}")
            return {
                "filename": file.filename,
                "rows": rows,
                "saved": len(rows),
            }

        per_file_results = await asyncio.gather(*(process_single_pdf(file) for file in files))
        all_rows = []
        filenames = []
        saved_count = 0
        for item in per_file_results:
            filenames.append(item["filename"])
            saved_count += item["saved"]
            all_rows.extend(item["rows"])

        # Insert filenames into pdf_files table
        for fname in filenames:
            try:
                cur.execute(
                    "INSERT INTO pdf_files (filename) VALUES (%s) ON CONFLICT DO NOTHING;", (fname,))
            except Exception as e:
                print(f"PDF filename insert error: {e}")

        if all_rows:
            cur.executemany(
                """
                INSERT INTO documents (content, embedding, source_filename, source_page, source_chunk)
                VALUES (%s, %s, %s, %s, %s)
                """,
                all_rows
            )

        conn.commit()
        return {"filenames": filenames, "chunks_saved": saved_count, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        if _is_timeout_error(e):
            raise HTTPException(
                status_code=504,
                detail=(
                    "Embedding request timed out while processing the PDF. "
                    "The server retried with smaller batches, but the provider still did not respond in time."
                ),
            )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
# Endpoint to fetch PDF file list

# Provide both /pdf-list and /list-pdfs endpoints for compatibility


@app.get("/pdf-list")
@app.get("/list-pdfs")
def get_pdf_list():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ensure_pdf_files_table(cur)
        cur.execute(
            "SELECT filename, uploaded_at FROM pdf_files ORDER BY uploaded_at DESC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"pdf_files": [r[0] for r in rows]}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    summarize_these: List[ChatMessage] = []
    current_summary: str = ""
    selected_model: str = ""
    temperature: float = 0.5
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@app.get("/models")
def get_models():
    ui_models = get_ui_text_models()
    return {
        "text_models": ui_models,
        "default_text_model": ui_models[0] if ui_models else ""
    }


@app.delete("/clear")
def clear_database():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE documents;")
        cur.execute("TRUNCATE TABLE pdf_files;")

        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": "Database wiped successfully."}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


@app.post("/chat")
async def chat(request: ChatRequest):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        active_text_models = resolve_text_models(request.selected_model)

        new_summary = request.current_summary
        if request.summarize_these:
            summarize_prompt = f"""
            You are maintaining PDF document assistant's long-term memory for a PDF research chat.
            Update the memory using the new dialogue.

            Keep only durable information that helps future answers:
            - user goals, constraints, preferences, and definitions
            - key facts already established from PDFs
            - unresolved questions and follow-up tasks

            Remove filler, greetings, and repeated content.
            Keep this memory concise and structured in short bullet points.

            Previous Summary:
            {request.current_summary}

            New Dialogue to summarize:
            """
            for msg in request.summarize_these:
                summarize_prompt += f"{msg.role.capitalize()}: {msg.content}\n"

            try:
                summary_completion = create_chat_completion_with_fallback(
                    model_candidates=active_text_models,
                    messages=[{"role": "user", "content": summarize_prompt}],
                    max_tokens=300,
                    temperature=0.3,
                    timeout_seconds=TEXT_TIMEOUT_SECONDS
                )
                new_summary = _completion_text(summary_completion)
            except Exception as e:
                print(f"Summarization error: {e}")
                new_summary = request.current_summary

        ensure_documents_metadata_columns(cur)
        db_embedding_dimensions = get_documents_embedding_dimension(
            cur) or EMBEDDING_DIMENSIONS
        query_vector = (await embed_texts_async([request.question], target_dimensions=db_embedding_dimensions))[0]
        cur.execute("""
            WITH ranked AS (
                SELECT
                    content,
                    source_filename,
                    source_page,
                    source_chunk,
                    1 - (embedding <=> %s::vector) AS similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(source_filename, '__unknown__')
                        ORDER BY embedding <=> %s::vector
                    ) AS per_file_rank
                FROM documents
            )
            SELECT content, source_filename, source_page, source_chunk, similarity
            FROM ranked
            WHERE per_file_rank <= %s
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_vector, query_vector, RETRIEVAL_PER_FILE_K, RETRIEVAL_TOP_K))

        results = cur.fetchall()
        context_blocks = []
        if results:
            for idx, row in enumerate(results, start=1):
                raw_text, source_filename, source_page, source_chunk, similarity = row
                chunk_text = (raw_text or "").strip()
                if len(chunk_text) > 1800:
                    chunk_text = chunk_text[:1800] + " ..."

                source_label = source_filename or "unknown"
                page_label = source_page if source_page is not None else "?"
                chunk_label = source_chunk if source_chunk is not None else "?"
                context_blocks.append(
                    f"[Chunk {idx} | similarity={float(similarity):.4f} | source={source_label} | page={page_label} | chunk={chunk_label}]\n{chunk_text}"
                )

        context_text = "\n\n".join(
            context_blocks) if context_blocks else "No PDF chunks retrieved."

        system_prompt = f"""
        You are PDF document assistant.

        Your task is to answer using retrieved PDF text chunks first, and conversation memory second.

        PDF CONTEXT:
        {context_text}

        LONG-TERM MEMORY (Summary of older conversation): 
        {new_summary if new_summary else "No prior history."}

        Rules for extraction and reasoning:
        1. Treat PDF text as noisy OCR-like content. Resolve broken line wraps and hyphenation carefully before interpreting meaning.
        2. Prefer the most relevant chunks by similarity and content overlap with the question.
        3. If the question requests comparison across topics/documents, synthesize evidence from multiple source files when available.
        4. Do not invent facts that are not supported by the provided PDF context.
        5. If the answer is partial or missing in context, say exactly what is missing.

        Output format:
        1. Give a direct answer first in 2-6 sentences.
        2. Then add "Evidence from PDF:" with 2-5 bullet points.
        3. Each bullet must cite chunk ids like [Chunk 1] and include filename/page when available.
        4. If there is uncertainty, end with "Confidence:" set to High, Medium, or Low and one short reason.
        """

        llm_messages = [{"role": "system", "content": system_prompt}]

        for msg in request.history:
            llm_messages.append({"role": msg.role, "content": msg.content})

        llm_messages.append({"role": "user", "content": request.question})

        completion = create_chat_completion_with_fallback(
            model_candidates=active_text_models,
            messages=llm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            timeout_seconds=TEXT_TIMEOUT_SECONDS
        )

        return {
            "answer": _completion_text(completion),
            "new_summary": new_summary
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()
