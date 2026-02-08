# RAG Flow Trace

**Query:** How to install Docker Hub on Linux?

## Steps

### Step 1: pre_retrieval
- **Model:** openai/gpt-4o
- **API calls:** 1
- **Detail:** Expanded to 3 search queries

### Step 2: retrieval
- **Model:** openai/text-embedding-3-small
- **API calls:** 3
- **Detail:** Vector search for 3 queries, 14 docs after dedup

### Step 3: check_and_refine
- **Model:** openai/gpt-5.2
- **API calls:** 1
- **Detail:** Grader score 0.42 < 0.5, refined query for retry

### Step 4: retrieval
- **Model:** openai/text-embedding-3-small
- **API calls:** 1
- **Detail:** Vector search for 1 queries, 6 docs after dedup

### Step 5: check_and_refine
- **Model:** -
- **API calls:** 0
- **Detail:** Skipped (no docs or retry limit)

### Step 6: post_retrieval
- **Model:** -
- **API calls:** 0
- **Detail:** Built context from 6 chunks (7010 chars)

### Step 7: generate
- **Model:** openai/gpt-4o
- **API calls:** 1
- **Detail:** Final answer generation

## Model Call Summary

- **openai/gpt-4o:** 2 call(s)
- **openai/gpt-5.2:** 1 call(s)
- **openai/text-embedding-3-small:** 4 call(s)
