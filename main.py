"""
Entry point for the LLM engineering classroom demo.

This file gives students a simple menu so they can run:
1. Prompt engineering examples in a health-guidance domain
2. ChromaDB + RAG examples over weight-loss documents
3. Hybrid graph + vector grounding demo for weight-loss guidance
4. Neo4j graph database demo over curated and extracted guidance facts
5. Run all demos in sequence

"""

from demo_output_utils import print_banner, print_section
from graph_db_store import is_neo4j_configured
from hybrid_graph_rag_demo import run_hybrid_graph_rag_demo
from neo4j_graph_demo import run_neo4j_graph_demo
from prompt_engineering_examples import run_prompt_engineering_demo
from rag_chromadb_demo import run_rag_demo


def main() -> None:
    """
    Show the classroom menu and dispatch to the selected demo.

    Parameters:
    - None.

    Returns:
    - None: This function reads menu input and runs the chosen demo.
    """
    print_banner("LLM Engineering Class Demo")
    print_section("Choose A Demo")
    print("1. Prompt engineering examples")
    print("2. ChromaDB RAG demo")
    print("3. Hybrid graph + vector grounding demo")
    print("4. Neo4j graph database demo")
    print("5. Run all demos")

    choice = input("\nEnter 1, 2, 3, 4, or 5: ").strip()

    if choice == "1":
        run_prompt_engineering_demo()
    elif choice == "2":
        run_rag_demo()
    elif choice == "3":
        if is_neo4j_configured():
            run_hybrid_graph_rag_demo()
        else:
            print(
                "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and "
                "NEO4J_PASSWORD in your .env file before running option 3."
            )
    elif choice == "4":
        if is_neo4j_configured():
            run_neo4j_graph_demo()
        else:
            print(
                "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and "
                "NEO4J_PASSWORD in your .env file before running option 4."
            )
    elif choice == "5":
        run_prompt_engineering_demo()
        run_rag_demo()
        if is_neo4j_configured():
            run_hybrid_graph_rag_demo()
            run_neo4j_graph_demo()
        else:
            print(
                "Neo4j is not configured. Graph-backed demos were skipped. "
                "Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your "
                ".env file to enable options 3 and 4."
            )
    else:
        print("Invalid choice. Please run the program again and enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
