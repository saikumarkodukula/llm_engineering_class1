# LangChain RAG Teaching Notes

Prepared for classroom use based on the Class 1 materials and the local RAG demo corpus.

## 1. Why This Matters

In retrieval-augmented generation (RAG), the model does not answer only from its pretrained memory. It first:

1. reads source documents
2. splits them into chunks
3. converts those chunks into embeddings
4. retrieves the most relevant chunks for a user question
5. uses those chunks as grounded context for the final answer

This means two design choices strongly affect answer quality:

- how we chunk documents
- how we retrieve the relevant chunks

If chunking is poor, important ideas get split apart.
If retrieval is poor, the model gets the wrong evidence.

## 2. The Classroom Analogy

Think of the document set like a textbook:

- chunking decides how the textbook is cut into study cards
- retrieval decides which study cards are handed to the student during the exam

If the cards are too random or too broad, the student gets confused.
If the wrong cards are selected, even a strong student gives a weak answer.

## 3. Chunking Strategies

### A. Fixed Chunking

**What it is**

Split text into equal-size windows, for example 1000 characters with 150 characters of overlap.

**Use it when**

- you want the simplest baseline
- documents are mostly plain text
- you need a fast and predictable method

**Classroom example**

A long obesity guideline page is cut every 1000 characters:

- Chunk 1: definition of obesity, risk factors, first recommendations
- Chunk 2: end of first recommendation plus beginning of treatment section
- Chunk 3: treatment details and follow-up advice

**Advantages**

- easy to explain
- fast to implement
- consistent chunk size

**Limitations**

- can split ideas in awkward places
- ignores headings and document structure
- not always the cleanest for dense PDFs

**Simple teaching example**

Question: "What supports safe and sustainable weight loss for adults?"

Fixed chunking may retrieve a chunk that contains:

"calorie deficit, physical activity, balanced eating pattern, and long-term behavior change"

That works, but the chunk may also contain unrelated nearby text because the split was mechanical.

### B. Recursive Chunking

**What it is**

Split text by natural separators first, such as headings, paragraphs, or sentences, and only cut more aggressively if the section is still too long.

**Use it when**

- you want a strong default strategy
- documents include paragraphs and sections
- you want cleaner chunks than fixed-size splitting

**Classroom example**

A guideline section titled "Dietary Interventions" may stay together as one chunk if it fits, instead of being cut in the middle of a recommendation.

**Advantages**

- preserves meaning better than fixed chunking
- often gives cleaner retrieval results
- good default for mixed text and PDFs

**Limitations**

- still not truly semantic
- depends on separators being present in the source text

**Why it is often the best classroom default**

It is simple enough to explain and strong enough to show better behavior than naive splitting.

### C. Semantic Chunking

**What it is**

Split text according to meaning, not just size. Adjacent sentences are grouped if their embeddings are semantically similar.

**Use it when**

- the document changes topic within the same page
- you want chunks to align with ideas rather than formatting
- you want students to see a more advanced chunking method

**Classroom example**

Suppose a page contains:

- causes of obesity
- dietary recommendations
- exercise advice
- medication notes

Semantic chunking tries to group the diet sentences together and the exercise sentences together, even if the page layout is messy.

**Advantages**

- better idea-level grouping
- often improves retrieval precision
- useful for concept-heavy documents

**Limitations**

- more computationally expensive
- slightly harder to explain than fixed or recursive chunking
- quality depends on embedding quality

**Teaching point**

This is useful when students need to understand that document boundaries are not always the same as idea boundaries.

### D. Structure-Based Chunking

**What it is**

Chunk according to explicit document structure first, such as headings, sections, bullet groups, or numbered recommendations.

**Use it when**

- documents have clean formatting
- PDFs contain meaningful section headers
- you want recommendation-level retrieval

**Classroom example**

A NICE guideline might contain headings such as:

- Assessment
- Dietary Advice
- Physical Activity
- Pharmacological Treatment

Structure-based chunking tries to keep each section intact before further splitting.

**Advantages**

- aligns well with how humans read documents
- often ideal for reports, policies, and guidelines
- easy to explain in class

**Limitations**

- depends on reliable structure extraction
- messy PDFs can weaken it
- not all documents have clear sections

**Teaching point**

This method is especially intuitive for students because it matches how they naturally organize notes.

### E. LLM-Based Chunking

**What it is**

Ask a language model to split the document into semantically complete chunks.

**Use it when**

- you want the most flexible chunking approach
- structure is weak and semantic boundaries are subtle
- you are demonstrating how LLMs can help upstream, not just at answer time

**Classroom example**

Instead of cutting mechanically, the model may return chunks like:

- causes and diagnosis of obesity
- lifestyle treatment recommendations
- safety limits and warnings
- medication and surgery considerations

**Advantages**

- can produce very natural chunk boundaries
- can preserve complex ideas better
- shows a more advanced LLM engineering pattern

**Limitations**

- depends on a working generation model
- slower and more expensive
- less deterministic
- harder to debug than fixed rules

**Current classroom note**

In your current setup, LLM-based chunking is blocked whenever the Ollama runtime cannot load a model successfully.

## 4. Retrieval Strategies

### A. Similarity Retrieval

**What it is**

Retrieve the chunks whose embeddings are most similar to the question embedding.

**Use it when**

- you want the standard baseline
- the question is direct and clearly phrased
- you need fast retrieval

**Classroom example**

Question: "What warnings are given about very-low-calorie diets?"

Similarity retrieval will usually find chunks containing phrases like:

- "should not be used routinely"
- "clinical support"
- "nutritionally complete plan"

**Advantages**

- simple
- fast
- easy for students to understand

**Limitations**

- may return near-duplicate chunks
- can miss diversity if many top chunks say almost the same thing

### B. MMR Retrieval

**What it is**

MMR stands for maximal marginal relevance. It balances two goals:

- relevance to the question
- diversity among the retrieved chunks

**Use it when**

- the top similarity results are repetitive
- you want broader evidence coverage
- a question could require multiple aspects of an answer

**Classroom example**

Question: "How should obesity be managed in adults?"

Similarity retrieval might return three chunks all about diet.
MMR is more likely to return:

- one chunk about diet
- one chunk about physical activity
- one chunk about behavior change or follow-up

**Advantages**

- reduces redundancy
- gives a more balanced context window
- useful for broad multi-part questions

**Limitations**

- slightly less intuitive than plain similarity
- can sometimes trade away the single strongest match in favor of variety

**Teaching point**

MMR is excellent for showing students that retrieval is not only about closeness, but also about coverage.

### C. HyDE Retrieval

**What it is**

HyDE stands for hypothetical document embeddings. The system first asks an LLM to draft a plausible answer, then uses that generated passage as the retrieval query.

**Use it when**

- user questions are vague or underspecified
- direct similarity search is weak
- you want a more advanced retrieval pattern

**Classroom example**

User question: "How do doctors usually handle this condition?"

That question is vague. HyDE may first generate a hypothetical passage about obesity treatment involving:

- assessment
- calorie reduction
- physical activity
- behavior support
- clinical follow-up

Then retrieval uses that richer hypothetical passage to find better supporting chunks.

**Advantages**

- can improve retrieval for weak or short questions
- shows how generation can improve search
- useful for advanced teaching

**Limitations**

- depends on a working generation model
- adds latency
- can drift if the hypothetical answer is poor

**Current classroom note**

In your environment, HyDE is blocked for the same reason as LLM-based chunking: the Ollama generation runtime is currently unstable.

## 5. Best Way To Explain The Difference In Class

Use this short script:

"Chunking decides how we break knowledge apart. Retrieval decides which pieces we bring back. Better chunking gives cleaner units of meaning. Better retrieval gives better evidence to the model."

Then compare them like this:

- Fixed: fastest baseline
- Recursive: best simple default
- Semantic: groups by meaning
- Structure: groups by headings and sections
- LLM-based: model decides chunk boundaries

And for retrieval:

- Similarity: nearest chunks
- MMR: nearest but more diverse chunks
- HyDE: generate a hypothetical answer first, then retrieve

## 6. Recommended Demo Flow For Students

Because your current Ollama runtime is unstable, the safest teaching sequence is:

1. Start with recursive chunking + similarity retrieval
2. Compare recursive chunking + MMR retrieval
3. Explain semantic and structure chunking conceptually
4. Mention LLM-based chunking and HyDE as advanced patterns
5. Demonstrate those later only after model generation is stable

## 7. Concrete Examples You Can Say In Class

### Example 1: Narrow question

Question: "What warnings are given about very-low-calorie diets?"

Best strategy:

- chunking: recursive or structure
- retrieval: similarity

Why:

The question is specific, so direct nearest-neighbor retrieval is usually enough.

### Example 2: Broad question

Question: "How should adults with obesity be managed?"

Best strategy:

- chunking: recursive, semantic, or structure
- retrieval: MMR

Why:

The answer needs multiple aspects such as diet, exercise, behavior change, and follow-up. MMR helps cover different evidence slices.

### Example 3: Vague question

Question: "What is the usual treatment approach here?"

Best strategy:

- retrieval: HyDE if a strong LLM is available

Why:

The user is not naming the concepts clearly. A hypothetical answer can create a richer retrieval query.

## 8. One-Slide Summary Version

You can teach the entire idea in one slide:

**Chunking**

- Fixed: equal-size windows
- Recursive: split by natural boundaries
- Semantic: split by meaning
- Structure: split by headings and sections
- LLM-based: let a model choose chunks

**Retrieval**

- Similarity: closest chunks
- MMR: relevant and diverse chunks
- HyDE: generate a hypothetical answer, then retrieve

**Best default for class**

- Recursive chunking + similarity or MMR retrieval

## 9. Final Teaching Takeaway

Students should leave with this mental model:

- A strong model alone is not enough
- Good answers depend on good evidence
- Good evidence depends on chunking and retrieval design
- RAG quality is often won or lost before generation even starts
