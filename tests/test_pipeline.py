"""Testes dos helpers de extração e chunking do pipeline."""

import pytest

import pipeline
from models import Document


# ── detecção de texto fragmentado ──────────────────────────────────────

def test_is_fragmented_detects_figure_like_pages():
    text = "\n".join(["a", "b", "12", "x", "y", "Uma linha normal de texto"])
    assert pipeline._is_fragmented(text) is True


def test_is_fragmented_accepts_regular_text_and_short_pages():
    regular = "\n".join(f"Linha de texto número {i} com conteúdo." for i in range(10))
    assert pipeline._is_fragmented(regular) is False
    # menos de 5 linhas nunca é considerado fragmentado
    assert pipeline._is_fragmented("a\nb\nc\nd") is False


# ── chunking ───────────────────────────────────────────────────────────

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def test_recursive_split_respects_chunk_size_without_overlap():
    text = "\n\n".join(f"Parágrafo {i} " + "palavra " * 10 for i in range(20))
    chunks = pipeline._recursive_split(text, 100, 0, SEPARATORS)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # sem overlap, nenhum conteúdo é perdido
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_recursive_split_adds_overlap_from_previous_chunk():
    text = "\n\n".join(["A" * 50, "B" * 50, "C" * 50])
    chunks = pipeline._recursive_split(text, 60, 10, SEPARATORS)

    assert chunks[0] == "A" * 50
    assert chunks[1] == "A" * 10 + " " + "B" * 50
    assert chunks[2] == "B" * 10 + " " + "C" * 50


def test_chunk_documents_preserves_metadata(monkeypatch):
    monkeypatch.setattr(pipeline, "CHUNK_SIZE", 60)
    monkeypatch.setattr(pipeline, "CHUNK_OVERLAP", 0)
    doc = Document(
        page_content="\n\n".join(["A" * 50, "B" * 50]),
        metadata={"filename": "paper.pdf", "page": 3},
    )

    chunks = pipeline._chunk_documents([doc])

    assert [c.page_content for c in chunks] == ["A" * 50, "B" * 50]
    for i, chunk in enumerate(chunks):
        assert chunk.metadata == {
            "filename": "paper.pdf",
            "page": 3,
            "chunk_index": i,
            "total_chunks": 2,
        }


# ── validação de entrada ───────────────────────────────────────────────

def test_ingest_pdf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_pdf(tmp_path / "inexistente.pdf")


def test_ingest_pdf_rejects_non_pdf(tmp_path):
    path = tmp_path / "notas.txt"
    path.write_text("conteúdo")
    with pytest.raises(ValueError, match=".txt"):
        pipeline.ingest_pdf(path)


def test_ingest_pdf_end_to_end_with_mocked_index(tmp_path, monkeypatch):
    import fitz

    path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Attention is all you need.")
        pdf.save(path)

    indexed = []
    monkeypatch.setattr(pipeline, "embed_documents", lambda docs: [(d, [0.0]) for d in docs])
    monkeypatch.setattr(pipeline, "add_documents", indexed.extend)

    result = pipeline.ingest_pdf(path)

    assert result["filename"] == "paper.pdf"
    assert result["pages"] == 1
    assert result["chunks"] == len(indexed) == 1
    assert "Attention is all you need." in result["full_text"]
    assert indexed[0][0].metadata["page"] == 1
