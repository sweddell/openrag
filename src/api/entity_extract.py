"""
Entity extraction endpoint for OpenRAG.

Extracts entities and relationships from indexed document chunks using LLM,
and writes results to the orchestraite_entities_v1 index in OpenSearch.
"""

import base64
import datetime
import httpx
import json
import os
import re

from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENTITY_INDEX = "orchestraite_entities_v1"
_ENTITY_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "org_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "filter_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "created_at": {"type": "date"},
            "model": {"type": "keyword"},
            "prompt_version": {"type": "keyword"},
            "entity_count": {"type": "integer"},
            "relationship_count": {"type": "integer"},
            "entities": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "description": {"type": "text"},
                },
            },
            "relationships": {
                "type": "nested",
                "properties": {
                    "source": {"type": "keyword"},
                    "target": {"type": "keyword"},
                    "description": {"type": "text"},
                    "strength": {"type": "integer"},
                },
            },
        }
    }
}

_ENTITY_EXTRACTION_PROMPT = """\
Extract entities and relationships from the following text chunk.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "entities": [
    {{"name": "string", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|PRODUCT|EVENT|OTHER", "description": "string"}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "description": "string", "strength": 1}}
  ]
}}

Rules:
- Extract 3-8 entities maximum per chunk
- Only extract entities clearly present in the text
- Relationships must reference entities in the entities list
- strength is always 1
- Return empty arrays if no clear entities exist

TEXT:
{text}
"""


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class EntityExtractRequest(BaseModel):
    filter_id: str
    org_id: str
    filename_prefix: Optional[str] = None
    max_chunks: int = Field(default=100, ge=1, le=500)
    llm_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

async def entity_extract_endpoint(body: EntityExtractRequest):
    """
    Extract entities from indexed document chunks and write to orchestraite_entities_v1.

    Reads chunks matching filter_id (by filename prefix) from the documents index,
    runs LLM entity extraction per chunk, and writes results to orchestraite_entities_v1.
    """
    # Use OPENSEARCH_HOST and OPENSEARCH_PORT to construct URL (Docker-compatible)
    os_host = os.environ.get("OPENSEARCH_HOST", "opensearch")
    os_port = os.environ.get("OPENSEARCH_PORT", "9200")
    os_url = f"https://{os_host}:{os_port}".rstrip("/")
    os_user = os.environ.get("OPENSEARCH_USERNAME", "admin").strip() or "admin"
    os_pwd = (
        os.environ.get("OPENSEARCH_PASSWORD")
        or os.environ.get("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
        or ""
    ).strip()
    if not os_pwd:
        raise HTTPException(status_code=503, detail="OPENSEARCH_PASSWORD not configured")

    os_auth = "Basic " + base64.b64encode(f"{os_user}:{os_pwd}".encode()).decode()
    os_headers = {"Authorization": os_auth, "Content-Type": "application/json"}

    _ollama_raw = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_ENDPOINT") or "http://host.docker.internal:11434"
    # In Docker, use host.docker.internal; on host, it resolves to localhost
    ollama_url = _ollama_raw.rstrip("/")
    llm_model = body.llm_model or "graphrag-extractor:latest"
    # Strip ollama/ prefix for direct Ollama API
    ollama_model = llm_model.replace("ollama/", "")

    async with httpx.AsyncClient(verify=False, timeout=30.0) as os_client:
        # Ensure entity index exists
        check = await os_client.head(f"{os_url}/{_ENTITY_INDEX}", headers=os_auth and {"Authorization": os_auth})
        if check.status_code == 404:
            create = await os_client.put(
                f"{os_url}/{_ENTITY_INDEX}",
                json=_ENTITY_INDEX_MAPPING,
                headers=os_headers,
            )
            if create.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail=f"Failed to create entity index: {create.text}")

        # Fetch chunks matching this filter (by filename prefix or filter_id field)
        # Filename prefix format for orgs is {orgId}__, for projects is proj-{projectId}__
        prefix = body.filename_prefix or f"{body.org_id}__"
        query = {
            "size": body.max_chunks,
            "_source": ["document_id", "filename", "text", "page"],
            "query": {
                "bool": {
                    "should": [
                        {"prefix": {"filename": {"value": prefix}}},
                        {"term": {"filter_id": body.filter_id}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        chunks_resp = await os_client.post(
            f"{os_url}/documents/_search",
            json=query,
            headers=os_headers,
        )
        if not chunks_resp.is_success:
            raise HTTPException(status_code=502, detail=f"OpenSearch chunk query failed: {chunks_resp.text}")

        hits = chunks_resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return {"success": True, "extracted": 0, "total": 0, "message": "No matching chunks found"}

    extracted = 0
    errors = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as llm_client:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as os_write:
                for hit in hits:
                    src = hit["_source"]
                    chunk_id = hit["_id"]
                    doc_id = src.get("document_id", chunk_id)
                    filename = src.get("filename", "")
                    text = (src.get("text") or "").strip()
                    if not text or len(text) < 50:
                        continue

                    # Truncate to ~2500 chars to stay comfortably within num_ctx
                    text_trunc = text[:2500]
                    prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text_trunc)

                    try:
                        llm_resp = await llm_client.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": ollama_model,
                                "prompt": prompt,
                                "stream": False,
                                "options": {"temperature": 0, "num_ctx": 4096},
                            },
                        )
                        llm_resp.raise_for_status()
                        raw = llm_resp.json().get("response", "")

                        # Strip markdown fences
                        raw = re.sub(r'<.*?>', '', raw, flags=re.DOTALL).strip()
                        raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
                        raw = raw.rstrip('`').strip()
                        # Find the outermost JSON object by bracket counting
                        parsed = None
                        depth = 0
                        start = None
                        for ci, ch in enumerate(raw):
                            if ch == '{':
                                if start is None:
                                    start = ci
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0 and start is not None:
                                    try:
                                        parsed = json.loads(raw[start:ci + 1])
                                        break
                                    except json.JSONDecodeError:
                                        continue
                        if not parsed:
                            errors.append(f"{chunk_id}: failed to parse LLM response as JSON")
                            continue

                        entities = parsed.get("entities", [])
                        relationships = parsed.get("relationships", [])

                        entity_doc = {
                            "org_id": body.org_id,
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "filter_id": body.filter_id,
                            "filename": filename,
                            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                            "model": llm_model,
                            "prompt_version": "v1",
                            "entity_count": len(entities),
                            "relationship_count": len(relationships),
                            "entities": entities,
                            "relationships": relationships,
                        }

                        index_resp = await os_write.post(
                            f"{os_url}/{_ENTITY_INDEX}/_doc/{chunk_id}",
                            json=entity_doc,
                            headers=os_headers,
                        )
                        if index_resp.is_success:
                            extracted += 1
                        else:
                            errors.append(f"{chunk_id}: index error {index_resp.status_code}")

                    except (httpx.HTTPError, json.JSONDecodeError, Exception) as e:
                        errors.append(f"{chunk_id}: {type(e).__name__}: {e}")
                        continue

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Entity extraction error: {type(exc).__name__}: {exc}") from exc

    return {
        "success": True,
        "extracted": extracted,
        "total": len(hits),
        "errors": errors[:10],
        "filter_id": body.filter_id,
        "org_id": body.org_id,
    }
