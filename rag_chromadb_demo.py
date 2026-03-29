"""
RAG demo using ChromaDB.

This file shows the complete classroom flow:
1. Create a few sample documents
2. Convert them into vector embeddings
3. Store them inside ChromaDB
4. Query the top-k most relevant chunks
5. Send the retrieved context to the LLM
"""

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from llm_utils import ask_llm, load_llm


# We keep the database folder inside the project so students can inspect it.
CHROMA_PATH = Path("chroma_storage")
DATA_DIR = Path("sample_data")


def get_sample_documents() -> List[Dict[str, str]]:
    """
    Return a small in-memory dataset for demonstration.

    In a real project, these could come from PDFs, web pages,
    company documents, or classroom notes.
    """
    return [
        {
            "id": "doc_1",
            "text": (
                "Transformers are deep learning models that are very effective "
                "for natural language processing tasks such as translation, "
                "summarization, question answering, and text generation."
            ),
        },
        {
            "id": "doc_2",
            "text": (
                "ChromaDB is a vector database used to store embeddings. "
                "It helps retrieve the most relevant text chunks for a query."
            ),
        },
        {
            "id": "doc_3",
            "text": (
                "Prompt engineering is the process of designing better prompts "
                "so a language model can produce more accurate and useful results."
            ),
        },
        {
            "id": "doc_4",
            "text": (
                "Retrieval-Augmented Generation, or RAG, combines information "
                "retrieval with text generation. The retriever fetches useful "
                "context, and the language model uses that context to answer."
            ),
        },
    ]


def load_documents_from_text_files() -> List[Dict[str, str]]:
    """
    Load documents from local .txt files.

    This lets students replace the in-memory examples with their own notes.
    Each text file becomes one document in the vector database.
    """
    documents: List[Dict[str, str]] = []

    if not DATA_DIR.exists():
        return documents

    for text_file in sorted(DATA_DIR.glob("*.txt")):
        file_text = text_file.read_text(encoding="utf-8").strip()

        if not file_text:
            continue

        documents.append(
            {
                "id": text_file.stem,
                "text": file_text,
            }
        )

    return documents


def get_documents_for_demo() -> List[Dict[str, str]]:
    """
    Prefer local text files when they exist.

    This makes the classroom demo more realistic because you can drop files
    into the folder without changing the code.
    """
    file_documents = load_documents_from_text_files()

    if file_documents:
        return file_documents

    return get_sample_documents()


def build_vector_store() -> Any:
    """
    Create a persistent ChromaDB collection and add sample documents.

    We use a sentence-transformer embedding model so every text chunk
    is converted into a numerical vector.
    """
    # PersistentClient stores the database on disk.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # This embedding model is widely used for simple semantic search demos.
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        local_files_only=True,
    )

    # If the collection already exists, ChromaDB returns it.
    collection = client.get_or_create_collection(
        name="class_notes",
        embedding_function=embedding_function,
    )

    documents = get_documents_for_demo()

    # Reset old demo content so repeated classroom runs stay predictable.
    existing_ids = [doc["id"] for doc in documents]
    try:
        collection.delete(ids=existing_ids)
    except Exception:
        # If documents do not exist yet, delete may fail.
        # That is fine for a first-time run.
        pass

    # Add documents, metadata, and ids into the vector database.
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[{"source": "class_demo"} for _ in documents],
    )

    return collection


def retrieve_top_k(collection: Any, user_query: str, top_k: int = 2) -> List[str]:
    """
    Query ChromaDB and return the top-k most relevant documents.
    """
    # This prevents asking for more results than the collection contains.
    safe_top_k = min(top_k, max(1, collection.count()))

    results = collection.query(
        query_texts=[user_query],
        n_results=safe_top_k,
    )

    # ChromaDB returns a nested structure.
    # We extract the list of retrieved document texts for the first query.
    return list(results["documents"][0])


def build_rag_prompt(user_query: str, retrieved_docs: List[str]) -> str:
    """
    Create the final prompt that includes retrieved context.
    """
    context_block = "\n\n".join(retrieved_docs)

    prompt = f"""
    Answer the question using only the context below.
    If the answer is not in the context, say "I could not find it in the retrieved documents."

    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """.strip()

    return prompt


def run_rag_demo() -> None:
    """
    Run the full RAG pipeline from vector storage to final LLM answer.
    """
    print("\nCreating or loading the ChromaDB vector store...")
    collection = build_vector_store()
    print(f"Loaded {collection.count()} documents into ChromaDB.")

    # Load the same Hugging Face model used in the prompt engineering demo.
    generator = load_llm()

    # Ask the user for a question so they can test retrieval live in class.
    user_query = input(
        "\nEnter a question for the RAG demo\n"
        "(example: What is ChromaDB used for?): "
    ).strip()

    if not user_query:
        user_query = "What is ChromaDB used for?"

    top_k = 2
    retrieved_docs = retrieve_top_k(collection, user_query=user_query, top_k=top_k)

    print("\nTop-k retrieved documents:\n")
    for index, doc in enumerate(retrieved_docs, start=1):
        print(f"{index}. {doc}\n")

    rag_prompt = build_rag_prompt(user_query, retrieved_docs)
    rag_answer = ask_llm(generator, rag_prompt, max_new_tokens=160)

    print("=" * 80)
    print("FINAL RAG PROMPT SENT TO THE LLM")
    print("=" * 80)
    print(rag_prompt)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(rag_answer)
