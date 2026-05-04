"""
RAG demo using ChromaDB.

This file shows the complete classroom flow:
1. Create or load weight-loss guidance documents
2. Convert them into vector embeddings
3. Store them inside ChromaDB
4. Query the top-k most relevant chunks
5. Send the retrieved context to the LLM
"""

from pathlib import Path
import re
from typing import Any, Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from demo_output_utils import print_banner, print_key_value, print_section
from llm_utils import ask_llm, load_llm


# We keep the database folder inside the project so students can inspect it.
CHROMA_PATH = Path("chroma_storage")
COLLECTION_NAME = "weight_loss_guidance"
DATA_DIR = Path("sample_data")
PDF_DIR = DATA_DIR / "pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
ADULT_TERMS = {"adult", "adults", "weight loss", "obesity", "calorie", "diet"}
CHILD_TERMS = {"child", "children", "adolescent", "young people", "teen"}


def get_sample_documents() -> List[Dict[str, str]]:
    """
    Return a tiny fallback corpus used when no local files are present.

    Parameters:
    - None.

    Returns:
    - List[Dict[str, str]]: A short list of built-in documents with `id` and
      `text` fields.
    """
    return [
        {
            "id": "doc_1",
            "text": (
                "Safe weight loss usually requires a calorie deficit, regular "
                "physical activity, and a balanced eating pattern that can be "
                "maintained over time."
            ),
        },
        {
            "id": "doc_2",
            "text": (
                "Very-low-calorie diets should not be used as routine obesity "
                "management without clinical support and a nutritionally "
                "complete plan."
            ),
        },
        {
            "id": "doc_3",
            "text": (
                "Long-term weight management is more likely when people use "
                "flexible dietary changes instead of highly restrictive plans "
                "that are difficult to sustain."
            ),
        },
        {
            "id": "doc_4",
            "text": (
                "Guidelines often recommend multicomponent care for obesity, "
                "including nutrition support, physical activity, behavior "
                "change strategies, and professional follow-up when needed."
            ),
        },
    ]


def chunk_text_for_rag(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split long text into overlapping chunks for vector retrieval.

    Parameters:
    - text: The raw text to split into chunks.
    - chunk_size: The target maximum size of each chunk in characters.
    - chunk_overlap: The number of characters to overlap between chunks.

    Returns:
    - List[str]: The ordered list of chunk strings.
    """
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            last_space = cleaned_text.rfind(" ", start, end)
            if last_space > start + 100:
                end = last_space

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def load_documents_from_text_files() -> List[Dict[str, Any]]:
    """
    Load `.txt` files from `sample_data/` and convert them into chunks.

    Parameters:
    - None.

    Returns:
    - List[Dict[str, Any]]: A list of chunk records, each containing `id`,
      `text`, and metadata such as file name and chunk index.
    """
    documents: List[Dict[str, Any]] = []

    if not DATA_DIR.exists():
        return documents

    for text_file in sorted(DATA_DIR.glob("*.txt")):
        file_text = text_file.read_text(encoding="utf-8").strip()

        if not file_text:
            continue

        for chunk_index, chunk_text in enumerate(chunk_text_for_rag(file_text), start=1):
            documents.append(
                {
                    "id": f"{text_file.stem}_chunk_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        "document_name": text_file.name,
                        "source_path": str(text_file),
                        "chunk_index": chunk_index,
                    },
                }
            )

    return documents


def load_documents_from_pdf_files() -> List[Dict[str, Any]]:
    """
    Load PDF files from `sample_data/pdfs/` and chunk them page by page.

    Parameters:
    - None.

    Returns:
    - List[Dict[str, Any]]: A list of chunk records extracted from PDF pages,
      including page and chunk metadata.
    """
    documents: List[Dict[str, Any]] = []

    if not PDF_DIR.exists():
        return documents

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        return documents

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires `pypdf`. Run `pip install -r requirements.txt`."
        ) from exc

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            for chunk_index, chunk_text in enumerate(
                chunk_text_for_rag(page_text),
                start=1,
            ):
                documents.append(
                    {
                        "id": f"{pdf_file.stem}_page_{page_number}_chunk_{chunk_index}",
                        "text": chunk_text,
                        "metadata": {
                            "document_name": pdf_file.name,
                            "source_path": str(pdf_file),
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                    }
                )

    return documents


def get_documents_for_demo() -> List[Dict[str, Any]]:
    """
    Choose the best available document source for the demo.

    Parameters:
    - None.

    Returns:
    - List[Dict[str, Any]]: The document chunks selected for the current demo
      run.
    """
    pdf_documents = load_documents_from_pdf_files()
    if pdf_documents:
        return pdf_documents

    file_documents = load_documents_from_text_files()

    if file_documents:
        return file_documents

    return get_sample_documents()


def build_vector_store() -> Any:
    """
    Build or refresh the ChromaDB collection used by the RAG demo.

    Parameters:
    - None.

    Returns:
    - Any: The ChromaDB collection object that stores the embedded chunks.
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
        name=COLLECTION_NAME,
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
        metadatas=[
            doc.get("metadata", {"source": "class_demo"})
            for doc in documents
        ],
    )

    return collection


def score_retrieved_chunk(user_query: str, chunk_text: str) -> int:
    """
    Apply a lightweight reranking score to make classroom retrieval cleaner.

    Parameters:
    - user_query: The user's question.
    - chunk_text: One retrieved chunk being rescored.

    Returns:
    - int: A higher score for chunks that better match the classroom intent.
    """
    lowered_query = user_query.lower()
    lowered_chunk = chunk_text.lower()
    score = 0

    query_mentions_children = any(term in lowered_query for term in CHILD_TERMS)
    query_mentions_adults = any(term in lowered_query for term in {"adult", "adults"})

    if any(term in lowered_chunk for term in ADULT_TERMS):
        score += 2

    if query_mentions_adults and any(term in lowered_chunk for term in {"adult", "adults"}):
        score += 3

    if not query_mentions_children and any(term in lowered_chunk for term in CHILD_TERMS):
        score -= 4

    if "safely" in lowered_query and any(
        phrase in lowered_chunk
        for phrase in {
            "do not",
            "should not",
            "harmful",
            "clinical support",
            "nutritionally complete",
        }
    ):
        score += 2

    if re.search(r"\b\d+\s*kcal\b", lowered_chunk):
        score += 1

    return score


def retrieve_top_k(collection: Any, user_query: str, top_k: int = 2) -> List[str]:
    """
    Query ChromaDB and return the top-k most relevant chunk texts.

    Parameters:
    - collection: The ChromaDB collection to query.
    - user_query: The user question used for semantic search.
    - top_k: The number of final chunks to return after reranking.

    Returns:
    - List[str]: The selected chunk texts used as RAG context.
    """
    retrieved_records = retrieve_top_k_records(
        collection=collection,
        user_query=user_query,
        top_k=top_k,
    )
    return [record["text"] for record in retrieved_records]


def retrieve_top_k_records(
    collection: Any,
    user_query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB and return top-k retrieved records with metadata.

    Parameters:
    - collection: The ChromaDB collection to query.
    - user_query: The user question used for semantic search.
    - top_k: The number of final records to return after reranking.

    Returns:
    - List[Dict[str, Any]]: Retrieved record dictionaries with text, metadata,
      distance, and rerank score.
    """
    # Pull a slightly larger candidate set, then rerank it for cleaner demos.
    safe_top_k = min(max(top_k * 3, top_k), max(1, collection.count()))

    results = collection.query(
        query_texts=[user_query],
        n_results=safe_top_k,
        include=["documents", "metadatas", "distances"],
    )

    candidate_records: List[Dict[str, Any]] = []
    documents = list(results["documents"][0])
    metadatas = list(results["metadatas"][0])
    distances = list(results["distances"][0])

    for document, metadata, distance in zip(documents, metadatas, distances):
        candidate_records.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
                "rerank_score": score_retrieved_chunk(user_query, document),
            }
        )

    ranked_records = sorted(
        candidate_records,
        key=lambda record: (
            record["rerank_score"],
            -(record["distance"] or 0.0),
        ),
        reverse=True,
    )

    return ranked_records[:top_k]


def build_rag_prompt(user_query: str, retrieved_docs: List[str]) -> str:
    """
    Assemble the final grounded prompt sent to the language model.

    Parameters:
    - user_query: The original user question.
    - retrieved_docs: The retrieved chunk texts that will become the context.

    Returns:
    - str: The final prompt text sent to the generation model.
    """
    context_block = "\n\n".join(retrieved_docs)

    prompt = f"""
    Answer the question using only the context below.
    Write a short, classroom-friendly summary in plain language.
    Prefer 3 to 5 clear bullet points when the question asks for guidance, methods, or recommendations.
    Do not copy long phrases from the context.
    Combine overlapping points into one clean summary.
    If the context includes warnings or limits, include them briefly.
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
    Run the end-to-end RAG demo with presentation-friendly console output.

    Parameters:
    - None.

    Returns:
    - None: This function prints the demo stages and final answer to the console.
    """
    print_banner("ChromaDB RAG Demo")
    print_section("Stage 1: Build Vector Store")
    collection = build_vector_store()
    print_key_value("Collection name", COLLECTION_NAME)
    print_key_value("Chunks loaded", str(collection.count()))

    print_section("Stage 2: Load Generation Model")
    generator = load_llm()
    print_key_value("Provider", generator["provider"])
    print_key_value("Model", generator["model_name"])

    user_query = input(
        "\nEnter a question for the RAG demo\n"
        "(example: What supports safe and sustainable weight loss?): "
    ).strip()

    if not user_query:
        user_query = "What supports safe and sustainable weight loss?"

    top_k = 2
    retrieved_docs = retrieve_top_k(collection, user_query=user_query, top_k=top_k)

    print_section("Stage 3: Retrieved Vector Context")
    print_key_value("User question", user_query)
    print_key_value("Top-k", str(top_k))
    for index, doc in enumerate(retrieved_docs, start=1):
        print(f"\n[Retrieved Chunk {index}]\n{doc}\n")

    rag_prompt = build_rag_prompt(user_query, retrieved_docs)
    rag_answer = ask_llm(generator, rag_prompt, max_new_tokens=260)

    print_section("Stage 4: Final Prompt Sent To The LLM")
    print(rag_prompt)

    print_section("Stage 5: Final Grounded Answer")
    print(rag_answer)
