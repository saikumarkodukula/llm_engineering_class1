"""
PyCharm-friendly launcher for running both demos in sequence.

This is useful for class rehearsals when you want one script that
executes prompt engineering, RAG, and the knowledge graph example.
"""

from knowledge_graph_demo import run_knowledge_graph_demo
from prompt_engineering_examples import run_prompt_engineering_demo
from rag_chromadb_demo import run_rag_demo


if __name__ == "__main__":
    run_prompt_engineering_demo()
    run_rag_demo()
    run_knowledge_graph_demo()
