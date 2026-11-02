# Local Textbook RAG Starter (Windows)

A minimal local-first retrieval-augmented generation project for searchable textbook PDFs.

## Development status

This project is the first prototype, and development is still in progress. It provides a starting point for experimenting with textbook retrieval and question answering; the ingestion pipeline and retrieval design are still being refined.

Data ingestion is currently taking a substantial share of development time. Textbooks combine formulas, diagrams, text, and tables, and their meaning often depends on how those elements relate to one another. An equation needs its variable definitions, a diagram needs its labels and surrounding explanation, and a table needs its row and column structure. Extracting readable text alone does not establish that the content is correct or complete.

This requires granular planning and experimentation with how each content type is extracted, represented, and checked against the original pages. Preserving reading order, connections between related elements, and source references is a central consideration. Errors or omissions at this stage carry through to retrieval and answer generation, so ingestion quality is a priority before expanding downstream functionality.

Other areas under consideration and planned changes include:

- **Chunking and embedding:** the key question is how to divide textbook content into useful retrieval units without separating definitions, derivations, or explanations from the material they describe. Chunk boundaries, overlap, metadata, and embedding choices need experimentation together, with representative textbook questions used to assess whether the retrieved context is sufficient.
- **LangChain and/or LangGraph:** these frameworks are being studied to understand where they could help organize the pipeline and its processing steps. Their adoption and scope remain open decisions while the workflow requirements become clearer.
- **Model inference:** a move from Ollama to Hugging Face Transformers is planned, with closer control over model loading and inference configuration as an intended direction. The transition will require checking model compatibility, memory requirements, and how inference integrates with retrieval; it is not yet implemented.
- **Development environment:** a move from Windows to Windows Subsystem for Linux (WSL) is imminent. This will shift development to a Linux environment and will require validating dependencies, GPU access, file paths, and setup instructions there.

The stack and setup instructions below describe the current Windows/Ollama prototype. They will be updated as these changes are implemented.

## Current stack

- Python + PyMuPDF: extracts text page by page from searchable PDFs.
- Ollama + `nomic-embed-text`: creates local embedding vectors.
- Qdrant local mode: persists the vector index under `data/qdrant/` without Docker.
- Ollama + `qwen3:4b`: generates answers from retrieved passages.

## 1. Prerequisites

1. Install Python 3.11 or newer from https://www.python.org/downloads/ and enable the option to add Python to PATH.
2. Install Ollama for Windows from https://ollama.com/download.
3. Ensure the NVIDIA driver is current. Ollama's Windows documentation lists supported Windows/driver requirements: https://github.com/ollama/ollama/blob/main/docs/windows.mdx

Open PowerShell in this project directory.

## 2. Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either adjust the execution policy for your user according to your organization's policy, or use Command Prompt and run `.venv\Scripts\activate.bat`.

## 3. Download models

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
ollama list
```

The model downloads use the internet, but ingestion and inference run locally after setup. `qwen3:4b` is the default generation model. If it is too slow or memory-heavy, download `qwen3:1.7b` and change `CHAT_MODEL` in `src/chat.py` to `qwen3:1.7b`.

## 4. Add textbooks

Copy searchable PDFs into:

```text
books\raw\
```

Subfolders are supported. Keep your originals backed up; this project only reads the PDFs.

## 5. Build the index

Close any running `chat.py` process first, then run:

```powershell
python src/ingest.py
```

The first ingestion extracts each page, splits page text into overlapping chunks, obtains embeddings in batches, and stores them locally in `data\qdrant\`.

**Important:** running ingestion rebuilds the collection from all PDFs under `books/raw/`. Do not run ingestion and chat at the same time because Qdrant local mode uses a local storage lock.

## 6. Ask questions

```powershell
python src/chat.py
```

Examples:

- `Explain the difference between a process and a thread using my textbooks.`
- `Which assumptions does the derivation of the Nyquist sampling theorem require?`
- `Compare the two books' definitions of time complexity.`

Type `exit` to stop. Answers show source labels and a separate retrieved-source list. Citations use physical PDF page numbers, which may differ from printed page numbering.

## Limitations of this MVP

- Best suited to searchable text PDFs. Scanned pages with no text layer need OCR.
- PDF text extraction may misread equations, complex tables, columns, headers, or footnotes. Manually inspect representative pages before trusting answers.
- Chunks do not cross PDF page boundaries, which simplifies page citation but can split a thought across pages.
- Retrieval is semantic-only at present. A later iteration should add BM25/keyword retrieval, reranking, chapter metadata, and an evaluation set.
- This is a CLI prototype, not yet a web UI.

## Troubleshooting

- `ollama` not found: reopen PowerShell after installing Ollama and verify `ollama --version`.
- Model not found: rerun the matching `ollama pull ...` command.
- Out of memory / very slow responses: change `CHAT_MODEL` to `qwen3:1.7b`, keep `NUM_CTX = 4096`, and close GPU-heavy applications.
- No chunks extracted: verify that text can be selected in the PDF viewer. Scanned PDFs will need OCR.
- Poor equation results: inspect the extracted text. If symbols were lost in PDF extraction, retrieval cannot repair the source text; add layout-aware processing for those books.
