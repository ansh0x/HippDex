# HippDex

## Work In-Progess

**Hippocampal Indexing for State-Space Models**

HippDex is a hybrid context-management layer for Mamba and other state-space models (SSMs). It addresses the state bottleneck — the lossy compression of long contexts into fixed hidden states — by implementing a biological-inspired retrieval mechanism that resurfaces forgotten factual data on demand.

The Problem
State-space models like Mamba scale linearly with sequence length, but they pay a price: past tokens are compressed into a fixed-size hidden state. Over long conversations, exact information (names, numbers, specific instructions) gets overwritten or diffused. This is the state bottleneck — formally, lossy compression.
Standard attention-based models avoid this by keeping everything accessible, but at O(n²) cost. HippDex asks: can we get the best of both?
The Idea
The brain doesn't store memories verbatim. It uses a sparse hippocampal index — pointers that reconstruct relevant past information when needed, without loading the entire history.
HippDex applies this to SSMs:

    Semantic state: Mamba's hidden state handles coherence, flow, and contextual understanding
    Factual index: A lightweight RAG layer retrieves exact chunks from conversation history when the model's state shows signs of forgetting
    Resurfacing: Retrieved chunks are appended to the input, updating the hidden state with precise information without full reprocessing

Think of it as context-aware decompression: zip the redundant past, retrieve only what the current state needs.

---

## How It Works

``` plain
User Input → Mamba Hidden State (semantic)
                ↓
        [Forgetting detected?]
                ↓
        Yes → Query RAG Index
                ↓
        Retrieve relevant factual chunks
                ↓
        Append chunks + re-update state
                ↓
        Generate with resurfaced context
```

---
**Key components**:

| Component  | Role |
| -------------- | --------------- |
| StateMonitor | Detects when hidden state diverges from recent factual claims |
| HippocampalIndex | Sparse retrieval index over conversation history |
| Resurfacer | Formats and injects retrieved chunks into the input stream |

---

## Benchmarks

HippDex is intended to improve vanilla Mamba on:

    Needle-in-a-haystack: Retrieve specific fact buried in long context
    Long-context QA: Answer questions requiring distant factual recall
    Conversation consistency: Maintain user-specific details across extended dialogue

---

## Background

HippDex idea was first explored in my blog [Why current LLMs can't reach AGI (and more)](https://dev.to/ansh0x/why-current-llms-cant-reach-agi-and-more-5bc6), under the pretext of a hack. Which HippDex still is, since it's not a architectural fix, but rather a workaround.
