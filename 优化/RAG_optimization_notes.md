# RAG Optimization Notes

## Overview

This project already implements a relatively complete optimized RAG pipeline, not a basic "embed everything and do one vector search" version.

The current main chain is:

`query rewrite -> dense retrieval + BM25 retrieval -> RRF fusion -> rerank -> context filtering -> evidence-grounded answer`

Below is a structured summary of the optimization points already implemented in the codebase.

## 1. Retrieval-Side Optimizations

### 1.1 Hybrid retrieval: dense + BM25

The project does not rely on a single retrieval method.

It uses:

- Dense vector retrieval based on embeddings
- BM25 lexical retrieval based on token overlap

These two branches are executed inside the same retriever.

Code:

- `HybridRetriever` entry: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:23)
- Main retrieval flow: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:39)
- Dense retrieval call: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:81)
- BM25 retrieval call: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:95)

Why this is a RAG optimization:

- Dense retrieval is good at semantic similarity
- BM25 is good at exact term matching, law/article names, special terms, and low-frequency keywords
- Combining both improves recall robustness

### 1.2 RRF fusion

After dense and BM25 retrieval, the project does not simply concatenate results.

It uses Reciprocal Rank Fusion, RRF, to merge multiple ranked lists.

Code:

- Fusion call: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:109)
- Fusion implementation: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:271)

Why this is a RAG optimization:

- Different retrieval channels have different score scales
- RRF avoids score normalization problems
- It is a stable and practical multi-recall fusion strategy

### 1.3 Rerank after fusion

After retrieval and fusion, the project performs cross-encoder reranking using `bge-reranker`.

Code:

- Reranker class: [app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:20)
- Rerank entry: [app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:32)
- Actual score computation: [app/retriever/reranker.py](C:/Users/25921/Desktop/rag-app/app/retriever/reranker.py:110)

Why this is a RAG optimization:

- Dense retrieval is recall-oriented
- Rerank is precision-oriented
- This improves the final quality of the chunks passed into the LLM

### 1.4 Query rewrite before retrieval

Before retrieval, the project rewrites the user query into a retrieval-friendly query.

It uses conversation history to resolve:

- references
- omitted subjects
- ambiguous expressions
- context-dependent wording

Code:

- Query rewriter: [app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:30)
- Rewrite entry: [app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:36)
- Rewrite provider selection: [app/retriever/query_rewrite.py](C:/Users/25921/Desktop/rag-app/app/retriever/query_rewrite.py:90)

Why this is a RAG optimization:

- Retrieval quality is heavily affected by query clarity
- Multi-turn chat often produces underspecified follow-up questions
- Query rewrite improves recall quality in multi-turn scenarios

### 1.5 Tenant and role scoped retrieval filtering

Retrieval is filtered by `tenant_key` and optionally `role_category`.

Code:

- Filter builder: [app/retriever/hybrid.py](C:/Users/25921/Desktop/rag-app/app/retriever/hybrid.py:298)

Why this is a RAG optimization:

- It reduces retrieval noise
- It prevents wrong-role or wrong-user knowledge contamination
- It improves relevance under multi-user, multi-role deployment

## 2. Knowledge Ingestion Optimizations

### 2.1 Structured semantic chunking with overlap

The project does not split text using a naive fixed-line method.

It uses `RecursiveCharacterTextSplitter` with:

- configurable `chunk_size`
- configurable `chunk_overlap`
- language-aware separators
- token-based length calculation

Code:

- Chunker class: [app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:13)
- Splitter creation: [app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:21)
- Chunking entry: [app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:31)

Why this is a RAG optimization:

- Overlap helps reduce boundary information loss
- Better chunk shape improves both retrieval and answer grounding
- Token-aware splitting is more stable than pure character-based splitting

### 2.2 Heading path preservation

During chunking, the project tries to infer a `heading_path` for each chunk.

Code:

- Heading inference: [app/knowledge/chunker.py](C:/Users/25921/Desktop/rag-app/app/knowledge/chunker.py:81)

Why this is a RAG optimization:

- Preserving section structure improves interpretability
- It gives each chunk stronger contextual identity
- This is helpful for legal, policy, and structured documents

### 2.3 Text cleaning before embedding

Before chunking and embedding, the project cleans parsed text.

It handles:

- repeated header and footer removal
- advertisement line removal
- URL stripping
- special character cleanup
- minimum effective length validation
- sensitive word blocking

Code:

- Cleaner class: [app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:25)
- Repeated header/footer removal: [app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:170)
- Advertisement removal: [app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:192)
- Sensitive word check: [app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:205)
- Minimum length check: [app/knowledge/cleaner.py](C:/Users/25921/Desktop/rag-app/app/knowledge/cleaner.py:216)

Why this is a RAG optimization:

- Dirty chunks degrade both embedding quality and retrieval quality
- Removing repeated noise improves vector quality
- Input cleaning reduces retrieval pollution

### 2.4 Complex PDF parsing with quality evaluation and fallback

The project does not rely on a single PDF parser.

It implements:

- multi-strategy local parsing
- OCR fallback
- table extraction
- local quality scoring
- MinerU API fallback when quality is low

Code:

- Parser class: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:77)
- Cache directory: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:82)
- Local quality evaluation: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:178)
- MinerU fallback call: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:194)
- Quality function: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:333)

Why this is a RAG optimization:

- Better parsing directly improves downstream chunk quality
- Complex PDFs are a major failure source in real RAG systems
- Quality-aware fallback increases robustness without always paying remote parsing cost

### 2.5 PDF parse cache

Parsed PDF results are cached to avoid repeated expensive parsing.

Code:

- Cache hit path: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:103)
- Cache load: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:508)
- Cache save: [app/knowledge/pdf_parser.py](C:/Users/25921/Desktop/rag-app/app/knowledge/pdf_parser.py:593)

Why this is a RAG optimization:

- Reduces repeated parsing cost
- Improves ingest efficiency
- Useful for repeated uploads, retests, and evaluation workflows

## 3. Context Construction Optimizations

### 3.1 Short-term memory + long-term memory

The context builder does not only rely on retrieval results.

It also incorporates:

- recent conversation memory from Redis
- long-term memory summary
- history from MySQL when needed

Code:

- Context builder class: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:28)
- Recent memory loading: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:61)
- Long memory loading: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:62)
- Memory rendering: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:261)

Why this is a RAG optimization:

- Real QA quality depends on both retrieved knowledge and dialogue continuity
- Multi-turn RAG benefits from explicit memory injection

### 3.2 Evidence-grounded answering rules

The prompt explicitly instructs the model to answer based on retrieved evidence.

Code:

- Evidence rules constant: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:19)
- Evidence rules inserted into messages: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:134)

Why this is a RAG optimization:

- Reduces hallucination risk
- Encourages grounded answering
- Makes the system more aligned with retrieval-augmented behavior

### 3.3 Context filtering before prompt injection

The project filters retrieval results before passing them into the model.

It does:

- empty text filtering
- selective score filtering
- hard cap on cited evidence count

Code:

- Result filtering: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:171)
- Max source count: [app/chat/context_builder.py](C:/Users/25921/Desktop/rag-app/app/chat/context_builder.py:18)

Why this is a RAG optimization:

- Prevents prompt overload
- Keeps context concise
- Improves prompt quality and token efficiency

## 4. Configurable Optimization Parameters

These optimizations are not hardcoded only once. Many are parameterized in config.

Important configurable knobs include:

- `RETRIEVAL_BM25_TOP_K`
- `RETRIEVAL_VECTOR_TOP_K`
- `RETRIEVAL_RRF_K`
- `RETRIEVAL_ENABLE_QUERY_REWRITE`
- `RETRIEVAL_ENABLE_RERANK`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MINERU_FALLBACK_QUALITY_THRESHOLD`
- `PDF_PARSE_CACHE_DIR`

Code:

- Config entries: [app/core/config.py](C:/Users/25921/Desktop/rag-app/app/core/config.py:244)
- Retrieval knobs: [app/core/config.py](C:/Users/25921/Desktop/rag-app/app/core/config.py:259)

Why this is a RAG optimization:

- It allows tuning for different tasks, models, and hardware constraints
- It makes the pipeline easier to evaluate and improve experimentally

## 5. Current RAG Optimization Summary

The current project already includes these practical RAG optimizations:

- Hybrid retrieval
- BM25 + dense multi-recall
- RRF fusion
- Cross-encoder rerank
- Query rewrite
- Tenant-scoped retrieval filtering
- Semantic chunking with overlap
- Structural heading preservation
- Document cleaning before embedding
- Complex PDF parsing fallback
- PDF parse caching
- Dialogue memory integration
- Evidence-grounded prompt construction
- Context filtering before generation

## 6. What Is Not Yet Implemented

Based on the current code, these more advanced RAG strategies are not clearly implemented yet:

- Multi-query expansion
- HyDE
- Parent-child retrieval
- Graph retrieval
- Self-reflection / answer verification
- Adaptive top-k
- Retrieval result cache for online queries

## 7. One-Sentence Positioning

This project is already beyond a basic RAG demo and has implemented a practical, multi-stage optimized RAG pipeline centered on hybrid retrieval, reranking, robust document parsing, and evidence-grounded context construction.
