"""
Neo4j-backed graph storage for document chunks, entities, and relations.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


load_dotenv()


Triple = Tuple[str, str, str]

NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j").strip()

RELATION_PATTERNS = [
    (re.compile(r"(.+?)\s+is\s+(?:an?\s+|the\s+)?(.+)", re.IGNORECASE), "IS_A"),
    (re.compile(r"(.+?)\s+uses\s+(.+)", re.IGNORECASE), "USES"),
    (re.compile(r"(.+?)\s+combines\s+(.+)", re.IGNORECASE), "COMBINES"),
    (re.compile(r"(.+?)\s+recommends\s+(.+)", re.IGNORECASE), "RECOMMENDS"),
    (re.compile(r"(.+?)\s+requires\s+(.+)", re.IGNORECASE), "REQUIRES"),
    (re.compile(r"(.+?)\s+supports\s+(.+)", re.IGNORECASE), "SUPPORTS"),
    (re.compile(r"(.+?)\s+helps\s+(.+)", re.IGNORECASE), "HELPS"),
    (re.compile(r"(.+?)\s+reduces\s+(.+)", re.IGNORECASE), "REDUCES"),
    (re.compile(r"(.+?)\s+includes\s+(.+)", re.IGNORECASE), "INCLUDES"),
    (re.compile(r"(.+?)\s+stores\s+(.+)", re.IGNORECASE), "STORES"),
    (re.compile(r"(.+?)\s+connects\s+(.+)", re.IGNORECASE), "CONNECTS"),
    (re.compile(r"(.+?)\s+improves\s+(.+)", re.IGNORECASE), "IMPROVES"),
    (re.compile(r"(.+?)\s+powers\s+(.+)", re.IGNORECASE), "POWERS"),
    (re.compile(r"(.+?)\s+searches\s+(.+)", re.IGNORECASE), "SEARCHES"),
    (re.compile(r"(.+?)\s+answers\s+(.+)", re.IGNORECASE), "ANSWERS"),
    (re.compile(r"(.+?)\s+should\s+be\s+less\s+than\s+(.+)", re.IGNORECASE), "SHOULD_BE_LESS_THAN"),
    (re.compile(r"(.+?)\s+should\s+not\s+be\s+used\s+for\s+(.+)", re.IGNORECASE), "AVOIDS"),
    (re.compile(r"(.+?)\s+can\s+be\s+harmful\s+for\s+(.+)", re.IGNORECASE), "MAY_HARM"),
    (re.compile(r"(.+?)\s+is\s+part\s+of\s+(.+)", re.IGNORECASE), "PART_OF"),
]

TECHNICAL_PHRASES = [
    "knowledge graph",
    "vector database",
    "semantic search",
    "prompt engineering",
    "retrieval augmented generation",
    "retrieval",
    "generation",
    "embeddings",
    "retriever",
    "transformers",
    "llm",
]

STOP_TERMS = {
    "and",
    "are",
    "can",
    "does",
    "for",
    "how",
    "the",
    "use",
    "used",
    "what",
    "when",
    "where",
    "which",
    "with",
}

WEIGHT_LOSS_SEED_TRIPLES: List[Triple] = [
    ("Safe Weight Loss", "REQUIRES", "Calorie Deficit"),
    ("Safe Weight Loss", "REQUIRES", "Regular Physical Activity"),
    ("Safe Weight Loss", "REQUIRES", "Balanced Diet"),
    ("Calorie Deficit", "SHOULD_BE_LESS_THAN", "Energy Expenditure"),
    ("Low-Calorie Diet", "SUPPORTS", "Sustainable Weight Loss"),
    ("Balanced Diet", "SUPPORTS", "Long-Term Weight Management"),
    ("Regular Physical Activity", "SUPPORTS", "Weight Loss"),
    ("Regular Physical Activity", "SUPPORTS", "Overall Health"),
    ("Very-Low-Calorie Diet", "AVOIDS", "Routine Obesity Management"),
    ("Very-Low-Calorie Diet", "REQUIRES", "Clinical Support"),
    ("Very-Low-Calorie Diet", "REQUIRES", "Nutritionally Complete Plan"),
    ("Very-Low-Calorie Diet", "PART_OF", "Multicomponent Weight Management Strategy"),
    ("Restrictive Diet", "MAY_HARM", "Long-Term Weight Management"),
    ("Flexible Dietary Changes", "SUPPORTS", "Safe Weight Loss"),
    ("Professional Guidance", "SUPPORTS", "Safe Weight Loss"),
]

CURATED_ENTITIES = {
    entity
    for triple in WEIGHT_LOSS_SEED_TRIPLES
    for entity in (triple[0], triple[2])
}
CURATED_RELATIONS = {
    relation
    for _, relation, _ in WEIGHT_LOSS_SEED_TRIPLES
}


def is_neo4j_configured() -> bool:
    """
    Check whether the environment has the minimum settings for Neo4j.

    Parameters:
    - None.

    Returns:
    - bool: `True` if URI, username, and password are available.
    """
    return bool(NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD)


def get_neo4j_driver() -> Any:
    """
    Create the Neo4j driver used by the graph-backed demos.

    Parameters:
    - None.

    Returns:
    - Any: The Neo4j driver object used to open sessions and run Cypher queries.
    """
    if not is_neo4j_configured():
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and "
            "NEO4J_PASSWORD in your environment or .env file."
        )

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "The Neo4j driver is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:
        driver.close()
        raise RuntimeError(
            "Could not connect to Neo4j at "
            f"{NEO4J_URI}. Make sure the Neo4j server is running, the Bolt "
            "port is reachable, and the username/password are correct."
        ) from exc

    return driver


def normalize_graph_text(value: str) -> str:
    """
    Clean up text before storing it as a graph entity or value.

    Parameters:
    - value: The raw text to normalize.

    Returns:
    - str: A trimmed, whitespace-normalized string.
    """
    collapsed = re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}\"'")
    return collapsed[:200]


def sentence_split(text: str) -> List[str]:
    """
    Split a chunk into sentence-like spans for rule-based extraction.

    Parameters:
    - text: The raw chunk text.

    Returns:
    - List[str]: A list of sentence-like strings.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_entities_from_text(text: str) -> List[str]:
    """
    Extract simple entity candidates from text without extra NLP libraries.

    Parameters:
    - text: The chunk text to scan.

    Returns:
    - List[str]: A sorted list of detected entity names.
    """
    entities = set()

    pattern = r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
    for match in re.finditer(pattern, text):
        candidate = normalize_graph_text(match.group(0))
        if len(candidate) >= 2:
            entities.add(candidate)

    lowered_text = text.lower()
    for phrase in TECHNICAL_PHRASES:
        if phrase.lower() in lowered_text:
            if phrase == "llm":
                entities.add("LLM")
            else:
                entities.add(phrase.title())

    if "chromadb" in lowered_text:
        entities.add("ChromaDB")

    return sorted(entities)


def extract_triples_from_text(text: str) -> List[Triple]:
    """
    Extract simple subject-relation-object triples using regex rules.

    Parameters:
    - text: The chunk text to scan.

    Returns:
    - List[Triple]: A list of `(source, relation, target)` tuples.
    """
    triples: List[Triple] = []

    for sentence in sentence_split(text):
        normalized_sentence = normalize_graph_text(sentence)
        if not normalized_sentence:
            continue

        for pattern, relation in RELATION_PATTERNS:
            match = pattern.search(normalized_sentence)
            if not match:
                continue

            source = normalize_graph_text(match.group(1))
            target = normalize_graph_text(match.group(2))
            if source and target and source != target:
                triples.append((source, relation, target))
            break

    return triples


def ensure_graph_schema(driver: Any) -> None:
    """
    Create Neo4j constraints so duplicate ids are avoided during ingestion.

    Parameters:
    - driver: The Neo4j driver used to open a session.

    Returns:
    - None.
    """
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for statement in statements:
            session.run(statement)


def ingest_seed_knowledge(driver: Any) -> None:
    """
    Insert a small curated weight-loss graph to improve demo-quality relations.

    Parameters:
    - driver: The Neo4j driver used to write the seed graph.

    Returns:
    - None.
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        for source, relation, target in WEIGHT_LOSS_SEED_TRIPLES:
            session.run(
                f"""
                MERGE (source:Entity {{name: $source}})
                MERGE (target:Entity {{name: $target}})
                MERGE (source)-[:{relation}]->(target)
                """,
                source=source,
                target=target,
            )


def ingest_documents_into_graph(driver: Any, documents: List[Dict[str, Any]]) -> None:
    """
    Write document chunks and extracted graph facts into Neo4j.

    Parameters:
    - driver: The Neo4j driver used for writing.
    - documents: The list of chunk records to ingest.

    Returns:
    - None.
    """
    ensure_graph_schema(driver)
    ingest_seed_knowledge(driver)

    with driver.session(database=NEO4J_DATABASE) as session:
        for document in documents:
            metadata = document.get("metadata", {})
            chunk_id = document["id"]
            document_name = metadata.get("document_name", chunk_id)
            document_id = metadata.get("source_path", document_name)
            source_path = metadata.get("source_path", document_name)
            page_number = metadata.get("page_number")
            chunk_index = metadata.get("chunk_index", 0)
            text = document["text"]

            session.run(
                """
                MERGE (d:Document {id: $document_id})
                SET d.name = $document_name,
                    d.source_path = $source_path
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.page_number = $page_number,
                    c.chunk_index = $chunk_index
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                document_id=document_id,
                document_name=document_name,
                source_path=source_path,
                chunk_id=chunk_id,
                text=text,
                page_number=page_number,
                chunk_index=chunk_index,
            )

            entities = extract_entities_from_text(text)
            for entity_name in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $entity_name})
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    entity_name=entity_name,
                    chunk_id=chunk_id,
                )

            triples = extract_triples_from_text(text)
            for source, relation, target in triples:
                session.run(
                    f"""
                    MERGE (source:Entity {{name: $source}})
                    MERGE (target:Entity {{name: $target}})
                    MERGE (source)-[r:{relation}]->(target)
                    WITH source, target
                    MATCH (c:Chunk {{id: $chunk_id}})
                    MERGE (c)-[:SUPPORTS]->(source)
                    MERGE (c)-[:SUPPORTS]->(target)
                    """,
                    source=source,
                    target=target,
                    chunk_id=chunk_id,
                )


def extract_query_terms(question: str) -> List[str]:
    """
    Extract graph search terms from a user question.

    Parameters:
    - question: The user question written in natural language.

    Returns:
    - List[str]: A sorted list of lowercase terms used for graph matching.
    """
    terms = {
        normalize_graph_text(match.group(0)).lower()
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", question)
    }

    if "chromadb" in question.lower():
        terms.add("chromadb")

    for phrase in TECHNICAL_PHRASES:
        if phrase.lower() in question.lower():
            terms.add(phrase.lower())

    for seed_source, _, seed_target in WEIGHT_LOSS_SEED_TRIPLES:
        if seed_source.lower() in question.lower():
            terms.add(seed_source.lower())
        if seed_target.lower() in question.lower():
            terms.add(seed_target.lower())

    return sorted(term for term in terms if term and term not in STOP_TERMS)


def query_graph_facts(driver: Any, question: str, limit: int = 12) -> List[str]:
    """
    Query Neo4j for graph facts that match the user's question.

    Parameters:
    - driver: The Neo4j driver used for querying.
    - question: The natural-language question to match against graph entities.
    - limit: The maximum number of fact strings to return.

    Returns:
    - List[str]: A ranked list of graph fact strings and supporting evidence.
    """
    terms = extract_query_terms(question)
    if not terms:
        return []

    with driver.session(database=NEO4J_DATABASE) as session:
        records = session.run(
            """
            MATCH (e:Entity)
            WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
            OPTIONAL MATCH (e)-[r]->(target:Entity)
            OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(e)
            OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
            RETURN e.name AS entity,
                   type(r) AS relation,
                   target.name AS target,
                   doc.name AS document_name,
                   chunk.page_number AS page_number,
                   chunk.chunk_index AS chunk_index
            LIMIT $limit
            """,
            terms=terms,
            limit=limit,
        )

        curated_facts: List[str] = []
        supporting_facts: List[str] = []
        seen = set()
        for record in records:
            entity = record.get("entity")
            relation = record.get("relation")
            target = record.get("target")
            document_name = record.get("document_name")
            page_number = record.get("page_number")
            chunk_index = record.get("chunk_index")

            if entity and relation and target:
                fact = f"{entity} -[{relation}]-> {target}"
                if fact not in seen:
                    seen.add(fact)
                    if (
                        entity in CURATED_ENTITIES
                        or target in CURATED_ENTITIES
                        or relation in CURATED_RELATIONS
                    ):
                        curated_facts.append(fact)
                    elif not entity[:1].isdigit() and not target[:1].isdigit():
                        supporting_facts.append(fact)

            if entity and document_name is not None:
                location = f"{document_name}"
                if page_number is not None:
                    location += f" page {page_number}"
                if chunk_index is not None:
                    location += f" chunk {chunk_index}"
                fact = f"{entity} appears in {location}"
                if fact not in seen:
                    seen.add(fact)
                    if entity in CURATED_ENTITIES:
                        supporting_facts.append(fact)

        combined = curated_facts + supporting_facts
        return combined[:limit]


def close_neo4j_driver(driver: Optional[Any]) -> None:
    """
    Close the Neo4j driver when the demo finishes.

    Parameters:
    - driver: The driver object to close. It may also be `None`.

    Returns:
    - None.
    """
    if driver is not None:
        driver.close()
