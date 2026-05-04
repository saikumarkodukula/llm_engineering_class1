# LLM Engineering Class 1

This repository is a classroom demo for prompt engineering, retrieval-augmented generation (RAG), and graph-based grounding using weight-loss and obesity-management content.

The codebase is intentionally small and split into focused modules so students can trace each stage of the pipeline:

1. Prompt the model directly
2. Retrieve relevant evidence from ChromaDB
3. Add graph facts from Neo4j
4. Generate a grounded answer

## Project Focus

The current dataset and demo defaults are centered on weight-loss guidance:

- built-in fallback documents discuss safe and sustainable weight loss
- `sample_data/` contains local text files used when PDFs are not available
- `sample_data/pdfs/` can hold guideline PDFs for richer retrieval
- `graph_db_store.py` seeds Neo4j with curated weight-loss relationships and adds simple extracted relations from ingested text

This makes the demos easier to explain because the retrieval output is tied to one domain instead of mixed general-purpose examples.

## Demos

Run the menu app:

```bash
python3 main.py
```

Menu options:

- `1` Prompt engineering examples using health-guidance prompts
- `2` ChromaDB RAG demo over the local weight-loss corpus
- `3` Hybrid vector + graph demo using ChromaDB and Neo4j together
- `4` Neo4j graph retrieval demo
- `5` Run all demos in sequence
- `6` Run the RAG evaluation scorecard
- `7` Run the LangChain RAG strategies demo

Direct launchers:

```bash
python3 run_prompt_demo.py
python3 run_rag_demo.py
python3 run_hybrid_demo.py
python3 run_neo4j_graph_demo.py
python3 run_both_demos.py
python3 run_rag_evaluation.py
python3 run_langchain_rag_demo.py
```

## Repository Layout

- `main.py` menu entry point for the classroom app
- `prompt_engineering_examples.py` prompt-technique examples
- `llm_utils.py` shared provider loading and text generation helpers
- `rag_chromadb_demo.py` ingestion, chunking, vector storage, retrieval, and RAG prompt construction
- `rag_evaluation.py` golden-dataset RAG evaluation and scorecard reporting
- `langchain_rag_demo.py` LangChain-based chunking strategies and retrieval strategies including HyDE
- `graph_db_store.py` Neo4j schema, seed facts, extraction helpers, ingestion, and graph querying
- `hybrid_graph_rag_demo.py` combines vector hits and graph facts before generation
- `neo4j_graph_demo.py` pure graph retrieval demo
- `demo_output_utils.py` console-formatting helpers
- `sample_data/` local `.txt` source files
- `sample_data/pdfs/` optional PDF corpus for richer classroom examples
- `INSTALLATION.md` local setup instructions

## Provider Behavior

The app resolves the language-model provider in this order:

1. `LLM_PROVIDER` if explicitly set to `ollama`, `openai`, or `huggingface`
2. local Ollama if the Ollama server is reachable
3. local Hugging Face fallback

Important notes:

- Ollama is the preferred local runtime for classroom use
- OpenAI is optional and only used when `OPENAI_API_KEY` is configured
- `.env.example` now defaults to `gpt-5-mini` for the optional OpenAI path
- Hugging Face remains available as an offline fallback after the model is cached locally

## Data Flow

### Prompt demo

The prompt demo sends different prompt styles to the same model so students can compare how phrasing changes the result.

### RAG demo

The RAG demo:

1. loads PDF or text documents
2. chunks the text
3. embeds the chunks with a sentence-transformer
4. stores them in ChromaDB
5. retrieves the best matching chunks
6. builds a grounded prompt
7. asks the model for a final answer

### RAG evaluation

The RAG evaluation workflow:

1. loads a golden dataset from `sample_data/rag_evaluation_dataset.json`
2. runs retrieval against the ChromaDB collection
3. measures Hit@3, Precision@5, Recall@5, MRR, and NDCG@5
4. optionally generates answers with the active LLM provider
5. scores faithfulness, correctness, relevancy, completeness, and format compliance
6. records latency, failure rate, and simple business-facing signals
7. saves the report to `rag_evaluation_report.json`

### LangChain RAG strategies

The LangChain demo adds:

1. chunking choices: fixed, semantic, recursive, structure-based, and LLM-based
2. retrieval choices: similarity, MMR, and HyDE
3. a separate persisted Chroma store under `chroma_storage_langchain/`
4. an interactive classroom flow for comparing strategy tradeoffs

### Graph demo

The graph demo:

1. creates Neo4j constraints
2. loads curated weight-loss seed facts
3. ingests chunk nodes and simple entity nodes
4. extracts lightweight subject-relation-object triples
5. queries matching graph facts for a user question

### Hybrid demo

The hybrid demo retrieves vector context first, then uses the question plus retrieved text to find matching Neo4j facts. Both evidence streams are placed into one final prompt.

## Configuration

Copy the example environment file if you want local overrides:

```bash
cp .env.example .env
```

Common settings:

- `LLM_PROVIDER`
- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `HF_MODEL_NAME`
- `HF_FALLBACK_MODEL_NAME`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Dependencies

Key packages used in the project:

- `transformers` and `torch` for the Hugging Face fallback path
- `chromadb` for vector storage and retrieval
- `sentence-transformers` for embeddings
- `pypdf` for PDF extraction
- `neo4j` for graph storage and querying
- `python-dotenv` for local configuration
- `openai` for the optional cloud provider

## Code Comments

All defined functions in the Python modules now include function-level docstrings, and the more instructional parts of the retrieval and hybrid flows include inline comments where the logic benefits from explanation.

## Known Limits

- The graph extraction is regex-based and intentionally lightweight
- PDF extraction quality depends on the source PDF text layer
- The first local model run may require downloads before offline use works
- Neo4j-backed demos require a running Neo4j instance and valid credentials
- This is a teaching project, not a clinical decision-support system
