# Vector Distance And Cosine Similarity

## 1. Why Embeddings Matter

In retrieval systems like ChromaDB, text is not compared as raw words. Instead, each text is converted into a numeric vector called an embedding.

Examples:

- query: "safe weight loss"
- chunk A: "balanced diet and physical activity"
- chunk B: "dog training tips"

After embedding, each text becomes a list of numbers. Similar meanings usually produce vectors that are closer together in vector space.

## 2. What Vector Distance Means

Vector distance is a numeric way to measure how far two vectors are from each other.

- small distance means more similar
- large distance means less similar

Think of each vector as a point in space:

- nearby points mean related meaning
- far points mean unrelated meaning

## 3. Euclidean Distance Example

Assume a very small 2D example only for understanding:

- query vector: `q = [2, 3]`
- chunk A vector: `a = [3, 4]`
- chunk B vector: `b = [9, 1]`

Formula:

```text
distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

### Query vs Chunk A

```text
distance(q, a)
= sqrt((3-2)^2 + (4-3)^2)
= sqrt(1^2 + 1^2)
= sqrt(2)
= 1.41
```

### Query vs Chunk B

```text
distance(q, b)
= sqrt((9-2)^2 + (1-3)^2)
= sqrt(7^2 + (-2)^2)
= sqrt(49 + 4)
= sqrt(53)
= 7.28
```

Interpretation:

- chunk A is much closer to the query
- chunk A is more similar
- chunk B is much farther away

## 4. Why Cosine Similarity Is Also Important

Distance measures physical closeness, but sometimes the direction of a vector matters more than absolute length.

Cosine similarity compares the angle between two vectors.

- cosine close to `1` means very similar direction
- cosine close to `0` means weak relation
- cosine close to `-1` means opposite direction

## 5. Cosine Similarity Formula

```text
cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)
```

Where:

- `a . b` is the dot product
- `||a||` and `||b||` are vector magnitudes

## 6. Cosine Example

Use these vectors:

- query: `q = [1, 2]`
- chunk A: `a = [2, 4]`
- chunk B: `b = [2, -1]`

### Query vs Chunk A

Dot product:

```text
q . a = (1*2) + (2*4) = 2 + 8 = 10
```

Magnitudes:

```text
||q|| = sqrt(1^2 + 2^2) = sqrt(5)
||a|| = sqrt(2^2 + 4^2) = sqrt(20)
```

Cosine:

```text
10 / (sqrt(5) * sqrt(20))
= 10 / sqrt(100)
= 10 / 10
= 1
```

That means chunk A points in exactly the same direction as the query.

### Query vs Chunk B

Dot product:

```text
q . b = (1*2) + (2*-1) = 2 - 2 = 0
```

Cosine:

```text
0 / (||q|| * ||b||) = 0
```

That means chunk B is not directionally similar to the query.

## 7. Euclidean Distance vs Cosine Similarity

Euclidean distance asks:

- how far apart are the points?

Cosine similarity asks:

- are the vectors pointing in the same direction?

Example:

- `[1, 1]`
- `[2, 2]`

These are different lengths, but they point in the same direction.

- Euclidean distance is not zero
- cosine similarity is `1`

That is why cosine similarity is often useful for embeddings.

## 8. How This Relates To ChromaDB

In your project:

1. each chunk is embedded using `all-MiniLM-L6-v2`
2. the user query is embedded using the same model
3. ChromaDB compares the query vector with stored chunk vectors
4. the nearest vectors are returned

So the retrieval logic is:

```text
query text -> query embedding
stored chunk text -> stored chunk embeddings
compare vectors
return closest chunks
```

## 9. Why The Same Embedding Model Must Be Reused

If document vectors are created by one embedding model but the query vector is created by a different embedding model, the vectors may not live in the same space.

That causes:

- poor ranking
- wrong neighbors
- possible dimension mismatch

So ingestion and query must use the same embedding model.

## 10. Small Retrieval Example

Suppose ChromaDB has these distances from the query:

- chunk A distance = `0.21`
- chunk B distance = `0.47`
- chunk C distance = `1.12`

Ranking:

1. chunk A
2. chunk B
3. chunk C

That means chunk A is the closest and is retrieved first.

## 11. In Your Code

The embedding model is configured in `rag_chromadb_demo.py` using:

```text
SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```

Then the collection is queried in `retrieve_top_k()`.

So the process is:

- Chroma gives top semantic candidates by vector similarity
- your code then reranks those candidates using `score_retrieved_chunk()`

## 12. Final Summary

Vector distance and cosine similarity are ways to compare embeddings.

- distance focuses on closeness
- cosine focuses on direction

Both are useful mental models for understanding semantic retrieval.

For your project, the key idea is simple:

similar meanings produce nearby vectors, and nearby vectors are retrieved first.
