"""
Entry point for the LLM engineering classroom demo.

This file gives students a simple menu so they can run:
1. Prompt engineering examples
2. ChromaDB + RAG examples
3. Knowledge graph example
4. Run all demos in sequence

"""

from knowledge_graph_demo import run_knowledge_graph_demo
from prompt_engineering_examples import run_prompt_engineering_demo
from rag_chromadb_demo import run_rag_demo


def main() -> None:
    """Show a tiny menu so the class can choose what to run."""
    print("\nLLM Engineering Class Demo")
    print("1. Prompt engineering examples")
    print("2. ChromaDB RAG demo")
    print("3. Knowledge graph demo")
    print("4. Run all demos")

    choice = input("\nEnter 1, 2, 3, or 4: ").strip()

    if choice == "1":
        run_prompt_engineering_demo()
    elif choice == "2":
        run_rag_demo()
    elif choice == "3":
        run_knowledge_graph_demo()
    elif choice == "4":
        run_prompt_engineering_demo()
        run_rag_demo()
        run_knowledge_graph_demo()
    else:
        print("Invalid choice. Please run the program again and enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
