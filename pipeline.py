import logging
from pathlib import Path
import fitz
from models import Document
from rag.embedder import embed_documents
from rag.store import add_documents, collection_info, delete_collection
from config import CHUNK_SIZE, CHUNK_OVERLAP


logger = logging.getLogger(__name__)


# ── extração ───────────────────────────────────────────────────────────

def _is_fragmented(text: str) -> bool:
    """Detecta páginas com texto fragmentado (figuras, imagens)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    short_lines = sum(1 for l in lines if len(l) <= 3)
    return short_lines / len(lines) > 0.5


def _extract_text(file_path: Path) -> list[Document]:
    """Extrai texto de um PDF página por página."""
    documents = []

    with fitz.open(file_path) as pdf:
        total_pages = len(pdf)
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            if _is_fragmented(text):
                logger.warning("Página %d ignorada — texto fragmentado.", page_num)
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                }
            ))

    return documents


def _extract_full_text(file_path: Path) -> str:
    """Extrai texto completo do PDF como string única."""
    full_text = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text = page.get_text()
            if text.strip() and not _is_fragmented(text):
                full_text.append(text)
    return "\n".join(full_text)


# ── chunking ───────────────────────────────────────────────────────────

def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # separador vazio: divide caractere a caractere (str.split("") lança ValueError)
    parts = list(text) if separator == "" else text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and len(separators) > 1:
                sub = _recursive_split(part, chunk_size, chunk_overlap, separators[1:])
                chunks.extend(sub[:-1])
                current = sub[-1] if sub else ""
            else:
                current = part

    if current:
        chunks.append(current)

    if chunk_overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            result.append(tail + " " + chunks[i])
        return result

    return chunks


def _chunk_documents(docs: list[Document]) -> list[Document]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    for doc in docs:
        doc_chunks = _recursive_split(
            doc.page_content,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            separators,
        )
        for i, chunk in enumerate(doc_chunks):
            if chunk.strip():
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                    }
                ))

    return chunks


# ── ingestão pública ───────────────────────────────────────────────────

def ingest_pdf(file_path: str | Path) -> dict:
    """
    Processa e indexa um PDF no Qdrant.

    Returns:
        Dicionário com resumo do processamento e texto completo.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Esperado arquivo .pdf, recebido: {path.suffix}")

    logger.info("Ingerindo: %s", path.name)

    full_text = _extract_full_text(path)
    docs = _extract_text(path)
    logger.info("%d página(s) extraída(s).", len(docs))

    chunks = _chunk_documents(docs)
    logger.info("%d chunk(s) gerado(s).", len(chunks))

    embedded = embed_documents(chunks)
    add_documents(embedded)

    return {
        "filename": path.name,
        "pages": len(docs),
        "chunks": len(chunks),
        "full_text": full_text,
    }


def status() -> None:
    """Exibe informações sobre o índice atual."""
    info = collection_info()
    logger.info("Collection: %s | Chunks: %s", info["name"], info["count"])