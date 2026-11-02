"""Index searchable textbook PDFs into a local Qdrant database using Ollama embeddings."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import ollama
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "books" / "raw"
DB_DIR = ROOT / "data" / "qdrant"
COLLECTION = "textbook_chunks"
EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS = 1600
OVERLAP_CHARS = 220
EMBED_BATCH_SIZE = 16


def chunk_page(text: str, max_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split one PDF page into overlapping chunks while preferring whitespace boundaries."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start + int(max_chars * 0.65), end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def read_pdfs() -> list[dict[str, Any]]:
    pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Copy at least one searchable textbook PDF into books/raw/ and retry."
        )

    chunks: list[dict[str, Any]] = []
    blank_pages = 0
    for pdf_path in pdf_paths:
        relative_path = pdf_path.relative_to(PDF_DIR).as_posix()
        print(f"Reading: {relative_path}")
        try:
            with fitz.open(pdf_path) as document:
                for page_index, page in enumerate(document):
                    # sort=True often improves reading order for normal, searchable PDFs.
                    page_text = page.get_text("text", sort=True)
                    page_chunks = chunk_page(page_text)
                    if not page_chunks:
                        blank_pages += 1
                    for chunk_index, text in enumerate(page_chunks):
                        # Stable IDs make repeated indexing replace matching chunks instead of duplicating them.
                        point_id = str(uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{relative_path}|pdf-page-{page_index + 1}|chunk-{chunk_index}",
                        ))
                        chunks.append({
                            "id": point_id,
                            "text": text,
                            "source": relative_path,
                            "book_title": pdf_path.stem,
                            "pdf_page": page_index + 1,
                            "chunk_index": chunk_index,
                        })
        except Exception as exc:
            print(f"WARNING: Could not process {relative_path}: {exc}", file=sys.stderr)

    print(f"Found {len(pdf_paths)} PDF(s) and extracted {len(chunks)} text chunk(s).")
    if blank_pages:
        print(
            f"NOTE: {blank_pages} page(s) produced no selectable text. These may be blank pages or scanned images; "
            "OCR/layout processing may be needed for those pages."
        )
    if not chunks:
        raise RuntimeError("No text chunks were extracted. Check whether the PDFs contain selectable text.")
    return chunks


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Verify Ollama and get the first vector batch before altering an existing index.
    chunks = read_pdfs()
    first_batch = chunks[:EMBED_BATCH_SIZE]
    try:
        first_response = ollama.embed(model=EMBED_MODEL, input=[item["text"] for item in first_batch])
        first_vectors = first_response.embeddings
    except Exception as exc:
        raise RuntimeError(
            f"Could not create embeddings with Ollama model '{EMBED_MODEL}'. "
            f"Check that Ollama is running and run: ollama pull {EMBED_MODEL}. Details: {exc}"
        ) from exc

    if not first_vectors:
        raise RuntimeError("Ollama returned no embeddings.")
    vector_size = len(first_vectors[0])

    # Rebuilding from all PDFs keeps the index consistent when source books change.
    # Close any running chat.py process before this step; Qdrant local mode locks its storage path.
    client = QdrantClient(path=str(DB_DIR))
    try:
        if client.collection_exists(COLLECTION):
            client.delete_collection(collection_name=COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            if start == 0:
                vectors = first_vectors
            else:
                response = ollama.embed(model=EMBED_MODEL, input=[item["text"] for item in batch])
                vectors = response.embeddings

            points = []
            for item, vector in zip(batch, vectors, strict=True):
                points.append(models.PointStruct(
                    id=item["id"],
                    vector=vector,
                    payload={
                        "text": item["text"],
                        "source": item["source"],
                        "book_title": item["book_title"],
                        "pdf_page": item["pdf_page"],
                        "chunk_index": item["chunk_index"],
                    },
                ))
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            done = min(start + len(batch), len(chunks))
            print(f"Embedded and indexed {done}/{len(chunks)} chunks")
    finally:
        client.close()

    print(f"\nDone. Index is stored locally at: {DB_DIR}")
    print("You can now run: python src/chat.py")


if __name__ == "__main__":
    main()
