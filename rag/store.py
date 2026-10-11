import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from models import Document
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


logger = logging.getLogger(__name__)


_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info("Conectando ao Qdrant em %s:%s", QDRANT_HOST, QDRANT_PORT)
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def _get_collection() -> None:
    client = _get_client()
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in collections:
        logger.info("Criando collection '%s'...", QDRANT_COLLECTION_NAME)
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )


def add_documents(
    docs_and_embeddings: list[tuple[Document, list[float]]],
    payload: dict | None = None,
) -> None:
    if not docs_and_embeddings:
        return

    _get_collection()
    client = _get_client()
    points = []

    for doc, embedding in docs_and_embeddings:
        point_payload = {
            "page_content": doc.page_content,
            **doc.metadata,
            **(payload or {}),
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=point_payload,
        ))

    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
    )
    logger.info("%d chunks indexados com sucesso.", len(points))


def query_as_string(query_text: str, n_results: int = 5) -> str:
    """Busca chunks e retorna como string formatada para o agente."""
    from rag.embedder import embed_query

    query_vector = embed_query(query_text)
    client = _get_client()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=n_results,
        with_payload=True,
    ).points

    if not results:
        return "Nenhum resultado encontrado nos documentos indexados."

    formatted = []
    for i, result in enumerate(results, start=1):
        payload = result.payload or {}
        source = payload.get("filename", "desconhecido")
        content = payload.get("page_content", "")
        score = round(result.score, 4)
        formatted.append(f"[{i}] (score={score}, source={source})\n{content}")

    return "\n\n---\n\n".join(formatted)


def delete_collection() -> None:
    client = _get_client()
    try:
        client.delete_collection(QDRANT_COLLECTION_NAME)
        logger.info("Collection '%s' removida.", QDRANT_COLLECTION_NAME)
    except Exception:
        logger.warning("Collection '%s' não existe, pulando.", QDRANT_COLLECTION_NAME)


def collection_info() -> dict:
    client = _get_client()
    info = client.get_collection(QDRANT_COLLECTION_NAME)
    return {
        "name": QDRANT_COLLECTION_NAME,
        "count": info.points_count,
    }