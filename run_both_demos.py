"""
PyCharm-friendly launcher for running all classroom demos in sequence.

This is useful for class rehearsals when you want one script that
executes prompt engineering, RAG, and the Neo4j-backed graph demos.
"""

from hybrid_graph_rag_demo import run_hybrid_graph_rag_demo
from graph_db_store import is_neo4j_configured
from neo4j_graph_demo import run_neo4j_graph_demo
from prompt_engineering_examples import run_prompt_engineering_demo
from rag_chromadb_demo import run_rag_demo


if __name__ == "__main__":
    run_prompt_engineering_demo()
    run_rag_demo()
    if is_neo4j_configured():
        run_hybrid_graph_rag_demo()
        run_neo4j_graph_demo()
