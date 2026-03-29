"""
Simple knowledge graph demo for classroom teaching.

This file uses a tiny in-memory graph of entities and relationships so
students can see how graph-style retrieval differs from vector retrieval.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


Triple = Tuple[str, str, str]
Graph = Dict[str, List[Tuple[str, str]]]


def get_sample_triples() -> List[Triple]:
    """Return a small set of entity-relationship triples for the demo."""
    return [
        ("LLM", "answers", "Questions"),
        ("Prompt Engineering", "improves", "LLM Output"),
        ("RAG", "combines", "Retrieval"),
        ("RAG", "combines", "Generation"),
        ("RAG", "uses", "Retriever"),
        ("Retriever", "searches", "ChromaDB"),
        ("ChromaDB", "is_a", "Vector Database"),
        ("Embeddings", "power", "Semantic Search"),
        ("Retriever", "uses", "Embeddings"),
        ("Knowledge Graph", "stores", "Entities"),
        ("Knowledge Graph", "stores", "Relationships"),
        ("Knowledge Graph", "connects", "Entities"),
    ]


def build_graph(triples: List[Triple]) -> Graph:
    """Build a simple adjacency list from the triples."""
    graph: Graph = {}
    for source, relation, target in triples:
        graph.setdefault(source, []).append((relation, target))
        graph.setdefault(target, [])
    return graph


def print_triples(triples: List[Triple]) -> None:
    """Print the graph facts in a readable format."""
    print("\nKnowledge graph triples:\n")
    for source, relation, target in triples:
        print(f"- ({source}) -[{relation}]-> ({target})")


def find_entity_matches(graph: Graph, text: str) -> List[str]:
    """Find graph entities mentioned in the user's text."""
    lowered_text = text.lower()
    return [entity for entity in graph if entity.lower() in lowered_text]


def answer_entity_question(graph: Graph, entity: str) -> str:
    """List the direct facts connected to an entity."""
    facts = graph.get(entity, [])
    if not facts:
        return f"I could not find any graph facts for '{entity}'."

    lines = [f"Facts connected to {entity}:"]
    for relation, target in facts:
        lines.append(f"- {entity} {relation} {target}")
    return "\n".join(lines)


def find_relationship_path(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    """
    Find a short undirected path between two entities.

    We treat edges as traversable in both directions so students can ask
    relation questions even if the triple direction does not match the wording.
    """
    if start == goal:
        return [start]

    neighbors: Dict[str, List[Tuple[str, str, str]]] = {entity: [] for entity in graph}
    for source, facts in graph.items():
        for relation, target in facts:
            neighbors[source].append((target, relation, "forward"))
            neighbors[target].append((source, relation, "backward"))

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        for neighbor, relation, direction in neighbors.get(current, []):
            if neighbor in visited:
                continue

            if direction == "forward":
                step = f"{current} -[{relation}]-> {neighbor}"
            else:
                step = f"{current} <-[{relation}]- {neighbor}"

            new_path = path + [step]
            if neighbor == goal:
                return new_path

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def answer_graph_question(graph: Graph, question: str) -> str:
    """Answer a small set of classroom-friendly graph questions."""
    matched_entities = find_entity_matches(graph, question)

    if "related" in question.lower() and len(matched_entities) >= 2:
        start, goal = matched_entities[0], matched_entities[1]
        path = find_relationship_path(graph, start, goal)
        if not path:
            return f"I could not find a graph path between {start} and {goal}."
        return "Relationship path:\n- " + "\n- ".join(path[1:])

    if matched_entities:
        return answer_entity_question(graph, matched_entities[0])

    sample_entities = ", ".join(sorted(graph.keys()))
    return (
        "I could not match that question to a graph entity.\n"
        f"Try asking about one of these: {sample_entities}"
    )


def run_knowledge_graph_demo() -> None:
    """Run the classroom knowledge graph demo."""
    triples = get_sample_triples()
    graph = build_graph(triples)

    print("\nBuilding a simple knowledge graph...")
    print(f"Loaded {len(triples)} triples across {len(graph)} entities.")
    print_triples(triples)

    user_question = input(
        "\nEnter a graph question\n"
        "(examples: What facts are connected to RAG? or How is RAG related to ChromaDB?): "
    ).strip()

    if not user_question:
        user_question = "How is RAG related to ChromaDB?"

    graph_answer = answer_graph_question(graph, user_question)

    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH ANSWER")
    print("=" * 80)
    print(graph_answer)
