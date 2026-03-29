# LLM Engineering Class 1

This project is a simple teaching starter for your first LLM engineering class.
It covers three big ideas:

1. How to load and use an LLM from Ollama or Hugging Face
2. How prompt engineering changes model output
3. How a basic RAG pipeline works with ChromaDB
4. How a simple knowledge graph can represent entities and relationships

The project is intentionally small, split into separate files, and commented so students can follow the flow without reading one large script.

## What This Project Demonstrates

- Loading a local Ollama model or a Hugging Face instruction model
- Sending prompts to the model
- Comparing different prompting styles
- Creating embeddings for documents
- Storing document vectors in ChromaDB
- Retrieving top-k matching documents
- Sending retrieved context back to the LLM

## Current Project Flow

When you run `python3 main.py`, the program shows a menu:

- `1` runs prompt engineering examples
- `2` runs the ChromaDB RAG demo
- `3` runs the knowledge graph demo
- `4` runs all demos in sequence

This keeps the first class simple because you can teach one concept at a time.

## Runtime Order

The classroom project now uses this provider order:

1. Local Ollama
2. Local Hugging Face fallback

Recommended local models:

- `llama3.2:latest` for prompt generation
- `nomic-embed-text:latest` for embeddings used by ChromaDB

## Installation

Students need to install the project requirements on their own machines before running the demos.

Use the platform guide here:

- `INSTALLATION.md`
  Step-by-step setup for both macOS and Windows.

## Project Files

- `main.py`
  Main classroom entry point. It shows the menu and decides which demo to run.

- `run_prompt_demo.py`
  Direct PyCharm Run/Debug file for prompt engineering only.

- `run_rag_demo.py`
  Direct PyCharm Run/Debug file for the RAG demo only.

- `run_both_demos.py`
  Direct PyCharm Run/Debug file for running all demos without menu input.

- `run_knowledge_graph_demo.py`
  Direct PyCharm Run/Debug file for the knowledge graph demo only.

- `llm_utils.py`
  Shared helper file for loading the configured LLM provider and generating answers.

- `prompt_engineering_examples.py`
  Contains multiple prompt techniques so students can compare outputs.

- `rag_chromadb_demo.py`
  Contains the full RAG pipeline: load documents, embed them, store them, retrieve top-k results, and ask the LLM using retrieved context.

- `knowledge_graph_demo.py`
  Contains a small in-memory knowledge graph demo with triples, entity lookup, and simple relation-path search.

- `.env.example`
  Sample configuration file. You can copy it to `.env` and change the model name without editing Python files.

- `sample_data/`
  Folder for local `.txt` files. If text files are present here, the RAG demo uses them instead of the built-in sample documents.

- `llm_engineering_class_demo.ipynb`
  Notebook version of the classroom demo for teaching cell by cell.

- `requirements.txt`
  Project dependencies with comments describing what each package is used for.

## Code-Level Explanation

### `main.py`

This file keeps the classroom flow simple.

- Prints a menu
- Accepts user input
- Runs prompt engineering only, RAG only, or both

Why this matters:
- Students see that orchestration can stay very small
- The real logic is moved into dedicated files, which makes teaching easier

### `llm_utils.py`

This file is the shared model utility layer.

What it does:

- Loads environment variables using `python-dotenv`
- Chooses Ollama first when available
- Falls back to a local Hugging Face model when Ollama is unavailable
- Can optionally use OpenAI later if you add credentials
- Sends prompts to the active provider and returns the generated answer

Important teaching idea:
- An LLM workflow is not magic. At a basic level, it is:
  prompt text -> selected provider -> generated answer

Why this file exists:
- Prompt engineering and RAG both use the same model
- Shared logic should live in one place instead of being repeated

### `prompt_engineering_examples.py`

This file is for showing how prompt wording affects model output.

It currently includes:

- Zero-shot prompting
  Ask the model to do a task with no examples.

- One-shot prompting
  Give one example, then ask the model to continue the same pattern.

- Few-shot prompting
  Give multiple examples so the pattern is even clearer.

- Role prompting
  Tell the model to act like a teacher or expert.

- Step-by-step prompting
  Ask the model to reason through the task in steps.

- Structured output prompting
  Ask the model to answer in a fixed format.

Important teaching idea:
- The same model can produce different results depending on prompt structure
- Better prompts often improve clarity, format, and task completion

Current observation:
- Ollama with `llama3.2:latest` gives better classroom prompt results than the smaller Hugging Face fallback
- The Hugging Face fallback still exists so the project can run offline without Ollama

### `rag_chromadb_demo.py`

This file teaches a basic Retrieval-Augmented Generation pipeline.

It is organized in clear steps:

#### Step 1: Load documents

Two choices exist in the code:

- `get_sample_documents()`
  Returns a small built-in list of demo documents

- `load_documents_from_text_files()`
  Loads `.txt` files from `sample_data/`

Then:

- `get_documents_for_demo()`
  Uses local files if available
  Falls back to built-in sample documents if not

Teaching point:
- RAG can work on local documents, not only hardcoded strings

#### Step 2: Build the vector store

`build_vector_store()` does the following:

- Creates a persistent ChromaDB client
- Creates or loads a Chroma collection named `class_notes`
- Uses `SentenceTransformerEmbeddingFunction`
- Embeds each document into vectors
- Stores document ids, text, and metadata in ChromaDB

Teaching point:
- LLMs do not search raw text directly
- Documents are converted into embeddings so similarity search becomes possible

#### Step 3: Retrieve top-k results

`retrieve_top_k()` does the following:

- Accepts a user query
- Sends the query to ChromaDB
- Returns the most relevant documents

Teaching point:
- This is the retrieval part of RAG
- `top_k` means how many relevant chunks you want back

#### Step 4: Build the final RAG prompt

`build_rag_prompt()` does the following:

- Joins the retrieved documents into one context block
- Adds the user question
- Instructs the model to answer only from the retrieved context

Teaching point:
- RAG still uses prompting
- The difference is that the prompt now contains retrieved knowledge

#### Step 5: Generate the final answer

`run_rag_demo()` does the following:

- Builds the vector store
- Loads the LLM
- Accepts a classroom question
- Retrieves top-k matching documents
- Prints retrieved results
- Creates the final grounded prompt
- Sends it to the LLM
- Prints the final answer

Teaching point:
- RAG is not a separate model
- It is a pipeline: retrieve relevant context first, then prompt the LLM with that context

## Packages And What Each One Does

### Core LLM packages

- `ollama`
  The local runtime used by the project through its HTTP API. Students install the Ollama app separately.

- `transformers`
  Hugging Face library used for the fallback local model path.

- `torch`
  Backend deep learning framework required to run the Hugging Face fallback locally.

### Embedding and retrieval packages

- `sentence-transformers`
  Used for embedding text into vectors so similar text can be retrieved by meaning.

- `chromadb`
  Vector database used to store embeddings and perform top-k similarity search.

### Configuration package

- `python-dotenv`
  Loads values from a `.env` file so model configuration can be changed without editing code.

### Optional cloud package

- `openai`
  Optional if you later want to switch the classroom demo to the OpenAI API.

### Optional teaching package

- `jupyter`
  Optional if you want to present the notebook file in a notebook interface.

## Requirements File

The `requirements.txt` file now groups packages by purpose:

- model loading and fallback execution
- model execution
- embeddings
- vector database
- environment configuration
- optional notebook support

This makes it easier for students to understand why each dependency exists.

## Installation

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional environment setup:

```bash
cp .env.example .env
```

Then edit `.env` if you want a different Hugging Face model.

## Running The Project

Run the terminal version:

```bash
python3 main.py
```

Run the PyCharm-friendly launchers:

```bash
python3 run_prompt_demo.py
python3 run_rag_demo.py
python3 run_both_demos.py
```

Run the notebook version:

- Open `llm_engineering_class_demo.ipynb` in Jupyter or VS Code
- Run the cells step by step during class

## Data For The RAG Demo

You have two ways to supply documents:

### Built-in demo documents

These are already included in `rag_chromadb_demo.py`.

### Local text files

Put `.txt` files inside the `sample_data/` folder.

Example:

- `sample_data/chromadb_notes.txt`
- `sample_data/intro_to_llms.txt`

If these files exist, the RAG demo uses them automatically.

## What Has Already Been Fixed In This Project

These issues were already handled:

- Split the project into separate files for easier teaching
- Added comments throughout the code
- Added `.env` support for model selection
- Added local text-file loading for RAG
- Added a notebook version
- Fixed compatibility issues with newer `transformers` versions by using `AutoTokenizer` and `AutoModelForSeq2SeqLM`
- Cached the Hugging Face generation model locally
- Cached the sentence-transformer embedding model locally
- Updated the Chroma embedding setup to use `local_files_only=True` so the embedding model can be loaded offline after caching
- Added safer `top_k` handling
- Added clearer error handling for first-time model downloads

## Current Models Used

- Generation model:
  `google/flan-t5-base`

- Embedding model:
  `all-MiniLM-L6-v2`

Why these were chosen:

- They are well-known classroom-friendly models
- They are easy to explain
- They are good enough for a first demonstration

## Known Limitations

- `google/flan-t5-base` is not a very strong model by current standards
- Some prompt outputs may be weak, repetitive, or factually wrong
- The first run on a new machine may require internet access to download models
- The embedding model must be cached once before fully offline use
- This RAG example uses whole text files as documents and does not yet chunk long documents

These are acceptable for class 1 because the goal is to explain the workflow, not build a production system.


## Quick Classroom Commands

Install:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
python3 main.py
```

Run prompt demo directly:

```bash
python3 run_prompt_demo.py
```

Run RAG demo directly:

```bash
python3 run_rag_demo.py
```

Run both demos directly:

```bash
python3 run_both_demos.py
```

Create `.env`:

```bash
cp .env.example .env
```

## Good First-Class Discussion Points

- Why do better prompts matter?
- Why does the same model fail on some tasks?
- Why do we need embeddings?
- Why store vectors in a database?
- What does top-k retrieval mean?
- Why does RAG reduce hallucination risk?
- Why is RAG still dependent on prompt quality?

## Summary

This project now gives you:

- a terminal demo
- a notebook demo
- prompt engineering examples
- a working ChromaDB RAG example
- configurable model selection
- local sample documents
- code comments for students
- documented package roles
- documented code flow

For a first LLM engineering class, this is enough to teach the core ideas clearly without unnecessary complexity.
