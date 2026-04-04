"""
Hybrid retrieval demo that combines vector search with a Neo4j-backed graph.

This classroom example combines ChromaDB retrieval with Neo4j-backed
graph evidence before sending grounded context to the LLM.
"""

from typing import Any, Dict, List

from demo_output_utils import print_banner, print_key_value, print_section
from graph_db_store import (
    close_neo4j_driver,
    get_neo4j_driver,
    ingest_documents_into_graph,
    query_graph_facts,
)
from llm_utils import ask_llm, load_llm
from rag_chromadb_demo import build_vector_store, get_documents_for_demo


def retrieve_vector_context(
    collection: Any,
    user_query: str,
    top_k: int = 3,
) -> List[Dict[str, str]]:
    """
    Retrieve vector-search matches for the hybrid demo.

    Parameters:
    - collection: The ChromaDB collection to query.
    - user_query: The question used for semantic retrieval.
    - top_k: The number of vector hits to return.

    Returns:
    - List[Dict[str, str]]: A list of vector hits, each containing `id` and `text`.
    """
    safe_top_k = min(top_k, max(1, collection.count()))
    results = collection.query(
        query_texts=[user_query],
        n_results=safe_top_k,
    )

    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    # Keep the result small and explicit so the later prompt is easy to inspect.
    return [
        {
            "id": doc_id,
            "text": text,
        }
        for doc_id, text in zip(ids, documents)
    ]


def build_hybrid_prompt(
    user_query: str,
    vector_hits: List[Dict[str, str]],
    graph_facts: List[str],
) -> str:
    """
    Build the final prompt that combines vector and graph evidence.

    Parameters:
    - user_query: The original user question.
    - vector_hits: The retrieved vector chunks.
    - graph_facts: The matching Neo4j graph facts.

    Returns:
    - str: The final grounded prompt sent to the generation model.
    """
    # Separate vector context from graph facts so students can see both signals.
    vector_block = "\n\n".join(
        f"[Vector Source: {hit['id']}]\n{hit['text']}" for hit in vector_hits
    )
    graph_block = "\n".join(f"- {fact}" for fact in graph_facts) or "- None found"

    return f"""
    Answer the question using only the grounded evidence below.
    Use both the vector context and the graph facts when they are relevant.
    If the answer is not supported by the evidence, say "I could not find enough grounded evidence."

    Vector context:
    {vector_block}

    Knowledge graph facts:
    {graph_block}

    Question:
    {user_query}

    Answer:
    """.strip()


def run_hybrid_graph_rag_demo() -> None:
    """
    Run the full hybrid demo that uses both ChromaDB and Neo4j.

    Parameters:
    - None.

    Returns:
    - None: This function prints the demo stages and final answer to the console.
    """
    print_banner("Hybrid Vector + Neo4j Demo")
    print_section("Stage 1: Build Vector Store")
    collection = build_vector_store()
    print_key_value("Vector chunks loaded", str(collection.count()))

    documents = get_documents_for_demo()
    print_section("Stage 2: Ingest Graph Data Into Neo4j")
    neo4j_driver = None
    try:
        neo4j_driver = get_neo4j_driver()
        ingest_documents_into_graph(neo4j_driver, documents)
        print_key_value("Graph chunks ingested", str(len(documents)))

        print_section("Stage 3: Load Generation Model")
        generator = load_llm()
        print_key_value("Provider", generator["provider"])
        print_key_value("Model", generator["model_name"])

        user_query = input(
            "\nEnter a hybrid retrieval question\n"
            "(example: What guidance supports safe weight loss and what should be avoided?): "
        ).strip()

        if not user_query:
            user_query = "What guidance supports safe weight loss and what should be avoided?"

        vector_hits = retrieve_vector_context(collection, user_query=user_query, top_k=3)
        graph_query_text = user_query + " " + " ".join(hit["text"] for hit in vector_hits)
        graph_facts = query_graph_facts(neo4j_driver, graph_query_text)

        hybrid_prompt = build_hybrid_prompt(user_query, vector_hits, graph_facts)
        hybrid_answer = ask_llm(generator, hybrid_prompt, max_new_tokens=180)

        print_section("Stage 4: Vector Retrieval Results")
        print_key_value("User question", user_query)
        print_key_value("Vector hits", str(len(vector_hits)))
        for index, hit in enumerate(vector_hits, start=1):
            print(f"\n[Vector Hit {index}] [{hit['id']}]\n{hit['text']}\n")

        print_section("Stage 5: Neo4j Graph Facts")
        print_key_value("Graph facts", str(len(graph_facts)))
        if graph_facts:
            for index, fact in enumerate(graph_facts, start=1):
                print(f"{index}. {fact}")
        else:
            print("No graph facts were found for this question.")

        print_section("Stage 6: Final Hybrid Prompt")
        print(hybrid_prompt)

        print_section("Stage 7: Final Grounded Answer")
        print(hybrid_answer)
    except RuntimeError as exc:
        print(f"Hybrid graph demo could not start: {exc}")
    finally:
        close_neo4j_driver(neo4j_driver)
