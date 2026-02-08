# RAG Flow Trace

**Query:** How can I persist data in Docker containers?

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
- **Detail:** Grader score 0.68 >= 0.5, docs OK

### Step 4: post_retrieval
- **Model:** -
- **API calls:** 0
- **Detail:** Built context from 8 chunks (6513 chars)

### Step 5: generate
- **Model:** openai/gpt-4o
- **API calls:** 1
- **Detail:** Final answer generation

## Model Call Summary

- **openai/gpt-4o:** 2 call(s)
- **openai/gpt-5.2:** 1 call(s)
- **openai/text-embedding-3-small:** 3 call(s)
