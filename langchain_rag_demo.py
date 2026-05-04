"""
LangChain-based RAG demo with multiple chunking and retrieval strategies.

This module implements the chunking approaches highlighted in the supplied PDF:
1. fixed-size chunking
2. semantic chunking
3. recursive chunking
4. document structure-based chunking
5. LLM-based chunking

It also exposes multiple retrieval paths:
1. naive similarity retrieval
2. MMR retrieval
3. HyDE retrieval
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from demo_output_utils import print_banner, print_key_value, print_section
from llm_utils import ask_llm, load_huggingface_llm, load_llm


DATA_DIR = Path("sample_data")
PDF_DIR = DATA_DIR / "pdfs"
LANGCHAIN_CHROMA_DIR = Path("chroma_storage_langchain")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEMANTIC_SIMILARITY_THRESHOLD = 0.72
LLM_CHUNK_DELIMITER = "\n---CHUNK---\n"


def get_langchain_embeddings() -> HuggingFaceEmbeddings:
    """
    Build the shared LangChain embedding model used by the new RAG flow.

    Parameters:
    - None.

    Returns:
    - HuggingFaceEmbeddings: The embedding adapter for LangChain.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_sample_langchain_documents() -> List[Document]:
    """
    Return a fallback corpus when no local files are available.

    Parameters:
    - None.

    Returns:
    - List[Document]: Small built-in documents for a predictable demo fallback.
    """
    return [
        Document(
            page_content=(
                "Safe weight loss usually requires a calorie deficit, regular "
                "physical activity, and balanced nutrition that can be sustained."
            ),
            metadata={"document_name": "fallback_doc_1.txt", "source_path": "built_in"},
        ),
        Document(
            page_content=(
                "Very-low-calorie diets should not be routine obesity treatment "
                "without clinical support and a nutritionally complete plan."
            ),
            metadata={"document_name": "fallback_doc_2.txt", "source_path": "built_in"},
        ),
    ]


def load_langchain_documents() -> List[Document]:
    """
    Load text and PDF files into LangChain Document objects.

    Parameters:
    - None.

    Returns:
    - List[Document]: Raw documents before chunking.
    """
    documents: List[Document] = []

    if DATA_DIR.exists():
        for text_file in sorted(DATA_DIR.glob("*.txt")):
            text = text_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_name": text_file.name,
                        "source_path": str(text_file),
                        "source_type": "text",
                    },
                )
            )

    if PDF_DIR.exists():
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires `pypdf`. Run `pip install -r requirements.txt`."
            ) from exc

        for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
            reader = PdfReader(str(pdf_file))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "document_name": pdf_file.name,
                            "source_path": str(pdf_file),
                            "source_type": "pdf",
                            "page_number": page_number,
                        },
                    )
                )

    return documents or get_sample_langchain_documents()


def chunk_documents_fixed(documents: Sequence[Document]) -> List[Document]:
    """
    Chunk documents into equal-size windows with overlap.

    Parameters:
    - documents: The raw documents.

    Returns:
    - List[Document]: Fixed-size chunks.
    """
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(list(documents))


def chunk_documents_recursive(documents: Sequence[Document]) -> List[Document]:
    """
    Chunk documents recursively along natural separators first.

    Parameters:
    - documents: The raw documents.

    Returns:
    - List[Document]: Recursive chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def split_by_document_structure(text: str) -> List[str]:
    """
    Split text into sections using headings and paragraph breaks.

    Parameters:
    - text: The raw document text.

    Returns:
    - List[str]: Structure-aware sections.
    """
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized_text.split("\n")]
    sections: List[str] = []
    current_lines: List[str] = []

    def flush_section() -> None:
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(section_text)
            current_lines.clear()

    for line in lines:
        is_heading = (
            len(line) < 90
            and line
            and (
                line.endswith(":")
                or line.isupper()
                or bool(re.match(r"^\d+(\.\d+)*[\)\.]?\s+", line))
            )
        )
        if not line:
            flush_section()
            continue
        if is_heading:
            flush_section()
            current_lines.append(line)
            continue
        current_lines.append(line)

    flush_section()
    return sections


def chunk_documents_structure_based(documents: Sequence[Document]) -> List[Document]:
    """
    Chunk documents by structure first, then recursively if sections are too long.

    Parameters:
    - documents: The raw documents.

    Returns:
    - List[Document]: Structure-aware chunks.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunked_documents: List[Document] = []

    for document in documents:
        sections = split_by_document_structure(document.page_content) or [document.page_content]
        for section_index, section in enumerate(sections, start=1):
            section_doc = Document(
                page_content=section,
                metadata={**document.metadata, "section_index": section_index},
            )
            chunked_documents.extend(recursive_splitter.split_documents([section_doc]))

    return chunked_documents


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentence-like units.

    Parameters:
    - text: The raw text.

    Returns:
    - List[str]: Sentence-like segments.
    """
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute cosine similarity for two dense vectors.

    Parameters:
    - vector_a: First vector.
    - vector_b: Second vector.

    Returns:
    - float: Cosine similarity score.
    """
    numerator = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = sum(value * value for value in vector_a) ** 0.5
    norm_b = sum(value * value for value in vector_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def chunk_documents_semantic(
    documents: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    """
    Build semantic chunks by grouping adjacent sentences with similar embeddings.

    Parameters:
    - documents: The raw documents.
    - embeddings: The embedding model used to compare sentences.

    Returns:
    - List[Document]: Semantic chunks.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunked_documents: List[Document] = []

    for document in documents:
        sentences = sentence_split(document.page_content)
        if not sentences:
            continue

        if len(sentences) == 1:
            chunked_documents.extend(recursive_splitter.split_documents([document]))
            continue

        sentence_embeddings = embeddings.embed_documents(sentences)
        current_sentences = [sentences[0]]
        previous_embedding = sentence_embeddings[0]

        for sentence, sentence_embedding in zip(sentences[1:], sentence_embeddings[1:]):
            similarity = cosine_similarity(previous_embedding, sentence_embedding)
            candidate_text = " ".join(current_sentences + [sentence])
            if similarity >= SEMANTIC_SIMILARITY_THRESHOLD and len(candidate_text) <= CHUNK_SIZE * 1.35:
                current_sentences.append(sentence)
            else:
                chunked_documents.extend(
                    recursive_splitter.split_documents(
                        [
                            Document(
                                page_content=" ".join(current_sentences),
                                metadata=dict(document.metadata),
                            )
                        ]
                    )
                )
                current_sentences = [sentence]
            previous_embedding = sentence_embedding

        if current_sentences:
            chunked_documents.extend(
                recursive_splitter.split_documents(
                    [
                        Document(
                            page_content=" ".join(current_sentences),
                            metadata=dict(document.metadata),
                        )
                    ]
                )
            )

    return chunked_documents


def split_text_with_llm(generator: Any, text: str) -> List[str]:
    """
    Ask the active LLM to split text into semantically isolated chunks.

    Parameters:
    - generator: The loaded LLM provider config.
    - text: The text segment to split.

    Returns:
    - List[str]: LLM-proposed chunk texts.
    """
    prompt = f"""
    Split the following document into semantically complete chunks for RAG.
    Rules:
    - keep each chunk self-contained
    - do not summarize
    - preserve the original wording
    - return only the chunks
    - separate chunks using the exact delimiter {LLM_CHUNK_DELIMITER.strip()}

    Document:
    {text}
    """.strip()
    response = ask_llm(generator, prompt, max_new_tokens=500)
    candidate_chunks = [chunk.strip() for chunk in response.split(LLM_CHUNK_DELIMITER) if chunk.strip()]
    if not candidate_chunks:
        return [text]
    return candidate_chunks


def chunk_documents_llm_based(documents: Sequence[Document]) -> List[Document]:
    """
    Chunk documents with an LLM, then enforce size limits recursively.

    Parameters:
    - documents: The raw documents.

    Returns:
    - List[Document]: LLM-guided chunks, or recursive chunks if no generator loads.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        generator = load_llm()
    except Exception:
        try:
            generator = load_huggingface_llm()
        except Exception:
            return chunk_documents_recursive(documents)

    pre_split_docs = chunk_documents_structure_based(documents)
    chunked_documents: List[Document] = []

    for document in pre_split_docs:
        if len(document.page_content) <= CHUNK_SIZE:
            chunked_documents.append(document)
            continue

        llm_chunks = split_text_with_llm(generator, document.page_content[:3500])
        for llm_chunk_index, llm_chunk in enumerate(llm_chunks, start=1):
            chunked_documents.extend(
                recursive_splitter.split_documents(
                    [
                        Document(
                            page_content=llm_chunk,
                            metadata={
                                **document.metadata,
                                "llm_chunk_index": llm_chunk_index,
                            },
                        )
                    ]
                )
            )

    return chunked_documents


def choose_chunking_strategy(
    strategy_name: str,
    documents: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    """
    Dispatch to the requested chunking strategy.

    Parameters:
    - strategy_name: The selected strategy label.
    - documents: Raw documents.
    - embeddings: Shared embeddings for semantic chunking.

    Returns:
    - List[Document]: Chunked documents.
    """
    strategy_name = strategy_name.strip().lower()
    strategies: Dict[str, Callable[..., List[Document]]] = {
        "fixed": chunk_documents_fixed,
        "semantic": lambda docs: chunk_documents_semantic(docs, embeddings),
        "recursive": chunk_documents_recursive,
        "structure": chunk_documents_structure_based,
        "llm": chunk_documents_llm_based,
    }
    if strategy_name not in strategies:
        return chunk_documents_recursive(documents)
    return strategies[strategy_name](documents)


def enrich_chunk_metadata(chunks: Sequence[Document]) -> List[Document]:
    """
    Add stable chunk identifiers into metadata.

    Parameters:
    - chunks: The chunked documents.

    Returns:
    - List[Document]: Documents with explicit chunk ids in metadata.
    """
    seen_counts: Dict[str, int] = defaultdict(int)
    enriched_documents: List[Document] = []

    for chunk in chunks:
        document_name = chunk.metadata.get("document_name", "unknown")
        seen_counts[document_name] += 1
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = f"{Path(document_name).stem}_chunk_{seen_counts[document_name]}"
        enriched_documents.append(Document(page_content=chunk.page_content, metadata=metadata))

    return enriched_documents


def build_langchain_vector_store(
    chunking_strategy: str,
) -> Tuple[Chroma, HuggingFaceEmbeddings, List[Document]]:
    """
    Build a LangChain Chroma vector store for the selected chunking strategy.

    Parameters:
    - chunking_strategy: The chosen chunking strategy.

    Returns:
    - Tuple[Chroma, HuggingFaceEmbeddings, List[Document]]: Vector store,
      embeddings, and final chunk list.
    """
    embeddings = get_langchain_embeddings()
    raw_documents = load_langchain_documents()
    chunked_documents = enrich_chunk_metadata(
        choose_chunking_strategy(chunking_strategy, raw_documents, embeddings)
    )

    persist_directory = LANGCHAIN_CHROMA_DIR / chunking_strategy
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=f"weight_loss_guidance_{chunking_strategy}",
    )

    return vector_store, embeddings, chunked_documents


def retrieve_with_similarity(
    vector_store: Chroma,
    user_query: str,
    top_k: int,
) -> List[Document]:
    """
    Retrieve top documents with standard similarity search.

    Parameters:
    - vector_store: The LangChain vector store.
    - user_query: The user query.
    - top_k: Number of results.

    Returns:
    - List[Document]: Retrieved documents.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever.invoke(user_query)


def retrieve_with_mmr(
    vector_store: Chroma,
    user_query: str,
    top_k: int,
) -> List[Document]:
    """
    Retrieve diverse documents with maximal marginal relevance.

    Parameters:
    - vector_store: The LangChain vector store.
    - user_query: The user query.
    - top_k: Number of results.

    Returns:
    - List[Document]: Retrieved documents.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, top_k)},
    )
    return retriever.invoke(user_query)


def build_hyde_document(generator: Any, user_query: str) -> str:
    """
    Generate a hypothetical answer document for HyDE retrieval.

    Parameters:
    - generator: The loaded generator.
    - user_query: The original question.

    Returns:
    - str: The hypothetical answer passage.
    """
    prompt = f"""
    Write a short hypothetical reference passage that would likely answer the question.
    The passage should sound like a document excerpt, not a direct chat answer.

    Question:
    {user_query}

    Hypothetical passage:
    """.strip()
    return ask_llm(generator, prompt, max_new_tokens=180).strip()


def retrieve_with_hyde(
    vector_store: Chroma,
    user_query: str,
    top_k: int,
    generator: Any | None = None,
) -> Tuple[List[Document], str]:
    """
    Retrieve documents using HyDE.

    Parameters:
    - vector_store: The LangChain vector store.
    - user_query: The original user question.
    - top_k: Number of results.

    Returns:
    - Tuple[List[Document], str]: Retrieved documents and the hypothetical passage.
    """
    if generator is None:
        try:
            generator = load_llm()
        except Exception:
            generator = load_huggingface_llm()

    hypothetical_document = build_hyde_document(generator, user_query)
    retrieved_docs = vector_store.similarity_search(hypothetical_document, k=top_k)
    return retrieved_docs, hypothetical_document


def build_langchain_rag_prompt(user_query: str, retrieved_docs: Sequence[Document]) -> str:
    """
    Assemble the final grounded prompt for the generation model.

    Parameters:
    - user_query: The original question.
    - retrieved_docs: The retrieved LangChain documents.

    Returns:
    - str: Prompt text for the LLM.
    """
    context_blocks = []
    for document in retrieved_docs:
        source_name = document.metadata.get("document_name", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        context_blocks.append(
            f"[Source: {source_name} | Chunk: {chunk_id}]\n{document.page_content}"
        )

    context_block = "\n\n".join(context_blocks)
    return f"""
    Answer the question using only the retrieved context below.
    Write a short classroom-friendly answer.
    Use 3 to 5 bullet points when the question asks for guidance or recommendations.
    If the answer is not supported by the context, say "I could not find it in the retrieved documents."

    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """.strip()


def choose_retrieval_strategy() -> str:
    """
    Ask the user which retrieval strategy to use.

    Parameters:
    - None.

    Returns:
    - str: Strategy key.
    """
    print("\nRetrieval strategies:")
    print("1. similarity")
    print("2. mmr")
    print("3. hyde")
    choice = input("\nChoose a retrieval strategy [1-3] (default 3): ").strip()
    return {"1": "similarity", "2": "mmr", "3": "hyde"}.get(choice, "hyde")


def choose_chunking_strategy_interactive() -> str:
    """
    Ask the user which chunking strategy to use.

    Parameters:
    - None.

    Returns:
    - str: Strategy key.
    """
    print("\nChunking strategies:")
    print("1. fixed")
    print("2. semantic")
    print("3. recursive")
    print("4. structure")
    print("5. llm")
    choice = input("\nChoose a chunking strategy [1-5] (default 3): ").strip()
    return {
        "1": "fixed",
        "2": "semantic",
        "3": "recursive",
        "4": "structure",
        "5": "llm",
    }.get(choice, "recursive")


def run_langchain_rag_demo() -> None:
    """
    Run the LangChain RAG demo with selectable chunking and retrieval paths.

    Parameters:
    - None.

    Returns:
    - None: Prints the demo stages and final answer to the console.
    """
    print_banner("LangChain RAG Demo")
    chunking_strategy = choose_chunking_strategy_interactive()
    retrieval_strategy = choose_retrieval_strategy()

    print_section("Stage 1: Build LangChain Vector Store")
    vector_store, _embeddings, chunks = build_langchain_vector_store(chunking_strategy)
    print_key_value("Chunking strategy", chunking_strategy)
    print_key_value("Chunks loaded", str(len(chunks)))
    print_key_value("Persist directory", str(LANGCHAIN_CHROMA_DIR / chunking_strategy))

    print_section("Stage 2: Load Generation Model")
    try:
        generator = load_llm()
    except Exception:
        generator = load_huggingface_llm()
    print_key_value("Provider", generator["provider"])
    print_key_value("Model", generator["model_name"])

    user_query = input(
        "\nEnter a LangChain RAG question\n"
        "(example: What warnings are given about very-low-calorie diets?): "
    ).strip()
    if not user_query:
        user_query = "What warnings are given about very-low-calorie diets?"

    print_section("Stage 3: Retrieve Context")
    hypothetical_document = ""
    if retrieval_strategy == "mmr":
        retrieved_docs = retrieve_with_mmr(vector_store, user_query, top_k=3)
    elif retrieval_strategy == "hyde":
        retrieved_docs, hypothetical_document = retrieve_with_hyde(
            vector_store,
            user_query,
            top_k=3,
            generator=generator,
        )
    else:
        retrieved_docs = retrieve_with_similarity(vector_store, user_query, top_k=3)

    print_key_value("Retrieval strategy", retrieval_strategy)
    print_key_value("User question", user_query)
    if hypothetical_document:
        print("\n[HyDE Hypothetical Passage]\n")
        print(hypothetical_document)

    for index, document in enumerate(retrieved_docs, start=1):
        source_name = document.metadata.get("document_name", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        print(
            f"\n[Retrieved Chunk {index}] [{source_name} | {chunk_id}]\n"
            f"{document.page_content}\n"
        )

    rag_prompt = build_langchain_rag_prompt(user_query, retrieved_docs)
    rag_answer = ask_llm(generator, rag_prompt, max_new_tokens=240)

    print_section("Stage 4: Final Prompt Sent To The LLM")
    print(rag_prompt)

    print_section("Stage 5: Final Grounded Answer")
    print(rag_answer)

