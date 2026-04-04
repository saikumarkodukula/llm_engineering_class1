# Installation Guide

This project runs locally with Python and can optionally use Ollama and Neo4j for the full classroom experience.

## Requirements

- Python 3.10 or newer
- Terminal access
- Internet access for the first model download if the local models are not cached yet

Optional but recommended:

- Ollama for the preferred local generation path
- Neo4j for the graph and hybrid demos

## 1. Create A Virtual Environment

macOS or Linux:

```bash
cd "/path/to/class 1"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows Command Prompt:

```cmd
cd "C:\path\to\class 1"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd "C:\path\to\class 1"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is not available on Windows, replace it with `python`.

## 2. Optional Environment File

Copy the sample environment file:

```bash
cp .env.example .env
```

On Windows:

```cmd
copy .env.example .env
```

Use `.env` when you want to override the default model or add Neo4j or OpenAI credentials.

The sample environment file currently uses `gpt-5-mini` for the optional OpenAI path.

## 3. Optional Ollama Setup

Install Ollama from the official site:

- https://ollama.com/download

Recommended model pulls for this project:

```bash
ollama pull llama3.2:latest
```

The code uses a sentence-transformer through `chromadb` for embeddings, so you do not need a separate Ollama embedding model to run the current RAG flow.

## 4. Optional Neo4j Setup

Install and start Neo4j if you want menu options `3` and `4`.

Then set these values in `.env`:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

If Neo4j is not configured, the graph-backed menu options are skipped with a message instead of crashing.

## 5. Add Or Replace Source Documents

The project chooses documents in this order:

1. PDFs from `sample_data/pdfs/`
2. text files from `sample_data/`
3. built-in fallback documents inside `rag_chromadb_demo.py`

To teach with your own content:

- place `.pdf` files in `sample_data/pdfs/`
- or place `.txt` files in `sample_data/`

The current repository is already set up for weight-loss and obesity-guideline examples.

## 6. Run The App

Main menu:

```bash
python3 main.py
```

Direct launchers:

```bash
python3 run_prompt_demo.py
python3 run_rag_demo.py
python3 run_hybrid_demo.py
python3 run_neo4j_graph_demo.py
python3 run_both_demos.py
```

## Common Troubleshooting

### `pip` or `python3` not found

Try using the interpreter directly from the virtual environment:

```bash
.venv/bin/python main.py
```

On Windows:

```cmd
.venv\Scripts\python.exe main.py
```

### PowerShell blocks activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Ollama is installed but the app still falls back

Make sure the Ollama app or local server is running, then retry the demo.

### First run is slow

That is expected if the Hugging Face model or embedding model needs to download and cache locally.

### RAG answers look unrelated

Check which documents were loaded:

- PDFs take priority over text files
- stale or mixed-domain files in `sample_data/` can change retrieval quality

### Graph demos do not start

Check:

- Neo4j is running
- `.env` has valid `NEO4J_*` values
- the `neo4j` Python package is installed from `requirements.txt`
