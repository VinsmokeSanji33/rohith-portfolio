---
title: "Achieving 27x Inference Speedup with KV-Cache Offloading"
date: "2025-01-15"
author: "Rohith Behera"
readTime: 8
tags: ["LLM", "Performance", "vLLM", "LMCache"]
---

When working with long-context LLMs (180K+ tokens), the GPU memory bottleneck becomes painfully apparent. Every time you send a query with a large context, the model needs to recompute the entire KV-cache from scratch.

## The Problem

Standard LLM inference has a critical inefficiency: **the KV-cache is computed fresh for every request**, even when the context hasn't changed.

For a 180K-token document:
- **Baseline**: 39 seconds per query
- **GPU VRAM**: Saturated at 100%
- **Repeated queries**: Same 39-second wait

## The Solution: SSD-Backed KV-Cache

We implemented a tiered caching architecture using **vLLM + LMCache**:

```python
from vllm import LLM
from lmcache import KVCache

cache = KVCache(
    backend="ssd",
    path="/nvme/lmcache",
    max_tokens=200_000
)

response = llm.generate(
    context,
    cache=cache
)
```

### Architecture

1. **First request**: Compute KV-cache, store to NVMe SSD
2. **Subsequent requests**: Load from SSD (bypass recomputation)
3. **GPU VRAM freed**: Only store active inference state

## Results

| Metric | Baseline | With LMCache | Improvement |
|--------|----------|--------------|-------------|
| TTFT | 39.0s | 1.43s | **27x faster** |
| GPU Power | ~450W | ~280W | 38% lower |
| Throughput | 1 QPS | 4.3 QPS | 4.3x higher |

## Key Takeaways

1. **NVMe SSDs are fast enough** for KV-cache retrieval
2. **Repeated context queries** are the perfect use case
3. **GPU memory pressure** is dramatically reduced

The 27x speedup unlocks new possibilities for document QA, code analysis, and any application where users query the same context multiple times.

---

*This work was done in collaboration with Solidigm and NVIDIA. [Read more on Solidigm's blog](https://www.solidigm.com/products/technology/mlperf-ai-workloads-with-solidigm-ssds.html).*

