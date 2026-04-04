"""
Neo4j-backed graph retrieval demo for classroom use.
"""

from demo_output_utils import print_banner, print_key_value, print_section
from graph_db_store import (
    close_neo4j_driver,
    get_neo4j_driver,
    ingest_documents_into_graph,
    query_graph_facts,
)
from rag_chromadb_demo import get_documents_for_demo


def run_neo4j_graph_demo() -> None:
    """
    Run the pure Neo4j graph demo with structured console output.

    Parameters:
    - None.

    Returns:
    - None: This function prints graph facts for the user's question.
    """
    print_banner("Neo4j Graph Demo")
    print_section("Stage 1: Prepare Graph Documents")
    documents = get_documents_for_demo()
    print_key_value("Chunks prepared", str(len(documents)))

    print_section("Stage 2: Ingest Into Neo4j")
    driver = None
    try:
        driver = get_neo4j_driver()
        ingest_documents_into_graph(driver, documents)
        print("Stored documents, chunks, entities, and simple relations in Neo4j.")

        user_question = input(
            "\nEnter a graph database question\n"
            "(example: Which recommendations support safe weight loss?): "
        ).strip()

        if not user_question:
            user_question = "Which recommendations support safe weight loss?"

        graph_facts = query_graph_facts(driver, user_question)

        print_section("Stage 3: Retrieved Neo4j Facts")
        print_key_value("User question", user_question)
        print_key_value("Graph facts", str(len(graph_facts)))
        if graph_facts:
            for index, fact in enumerate(graph_facts, start=1):
                print(f"{index}. {fact}")
        else:
            print("I could not find matching graph facts in Neo4j.")
    except RuntimeError as exc:
        print(f"Neo4j graph demo could not start: {exc}")
    finally:
        close_neo4j_driver(driver)
