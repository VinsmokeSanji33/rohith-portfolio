---
title: "Paradigms in Large Language Model Inference and Infrastructure Optimization"
date: "2026-02-06"
author: "Rohith Behera"
readTime: 20
tags: ["inference optimization basics"]
---
The landscape of large language model (LLM) deployment has undergone a fundamental shift from simple model hosting to complex infrastructure orchestration. As the parameter counts of transformer-based architectures scale into the hundreds of billions, the primary constraints on performance have migrated from raw computational throughput to the "Memory Wall"—the growing disparity between processor speed and memory bandwidth. Modern inference engines must navigate the intricacies of the transformer's two-stage generation process: the prefill phase, which is compute-bound, and the decode phase, which is memory-bound.1 This dichotomy necessitates a suite of optimizations that span hardware-aware kernels, virtualized memory management, and asynchronous scheduling to maximize tokens-per-second while maintaining strict latency service level objectives (SLOs).
The optimization of LLM inference is not merely an exercise in reducing execution time but a comprehensive restructuring of data movement across the GPU memory hierarchy. High-bandwidth memory (HBM) on modern accelerators offers massive capacity but relatively high latency compared to on-chip SRAM, which serves as a fast but tiny scratchpad for active computations.4 Strategies such as Flash Attention address the IO bottleneck by minimizing HBM access, while Paged Attention addresses the fragmentation of the Key-Value (KV) cache, a secondary memory structure that grows linearly with sequence length and batch size.2 By integrating these techniques with advanced batching and speculative decoding, serving systems can achieve orders-of-magnitude improvements in throughput, fundamentally altering the economics of generative AI at scale.
Architectural Foundations of Transformer Inference
To optimize inference, one must first delineate the distinct computational profiles of the prefill and decode stages. When an LLM receives a prompt, it enters the prefill stage, where it processes the entire input sequence in parallel. This stage is characterized by high arithmetic intensity, as the model performs large-scale matrix-matrix multiplications that effectively saturate the GPU's tensor cores.8 The output of this stage is the first generated token and the initial population of the KV cache, which stores the attention keys and values for all input tokens to avoid redundant calculations in future steps.1
The subsequent decode stage is autoregressive, meaning tokens are produced sequentially. In each iteration, the model takes the previously generated token as input, retrieves the existing KV cache, and computes only the new token’s contribution. This stage involves matrix-vector multiplications, which have low arithmetic intensity and are limited by the speed at which model weights and the KV cache can be streamed from HBM to the processor.3 This sequential nature leaves the GPU largely underutilized unless multiple requests are batched together.
Feature
Prefill Stage
Decode Stage
Computation Type
Matrix-Matrix Multiplication
Matrix-Vector Multiplication
Hardware Constraint
Compute-bound (FLOPS)
Memory-bound (Bandwidth)
Parallelism
High (All prompt tokens at once)
Low (Sequential tokens)
Optimization Priority
Time to First Token (TTFT)
Inter-Token Latency (ITL)
KV Cache Activity
Generation/Population
Retrieval/Updating

8
Flash Attention: IO-Aware Attention Kernels
The standard self-attention mechanism requires the computation of an  attention matrix, where  is the sequence length. This quadratic scaling in memory creates a significant bottleneck, especially for long-context applications. Flash Attention re-engineered this process by focusing on "IO-awareness," ensuring that the majority of the mathematical operations occur within the GPU's fast SRAM rather than the slower HBM.4
The Strategy of IO-Awareness
One-liner: Flash Attention is a hardware-optimized algorithm that speeds up AI thinking by reducing the amount of data moved between the GPU's big storage and its fast workspace.4
Kid-level explanation: Imagine you are making a giant LEGO tower. Usually, you would walk to the big toy box across the room for every single brick, which takes a long time. Flash Attention is like bringing a small basket of bricks right to your seat. You build a whole section of the tower at once without moving, then only get up to put the finished piece on the castle. This makes building much faster because you spend less time walking.6
Conceptual diagram description: The diagram illustrates the GPU memory hierarchy with a large "HBM" pool and a small, high-speed "SRAM" block. In standard attention, large arrows show  matrices being read into SRAM, an  attention matrix being written back to HBM, and then read again for the final multiplication. In the Flash Attention diagram, the  matrices are divided into small "tiles." These tiles are pulled into SRAM one at a time. The diagram shows the softmax and weighted sums being calculated incrementally within the SRAM block, with no intermediate  matrix ever leaving the processor to go to HBM. Only the final output  is shown being written back to HBM at the end.4
Evolutionary Iterations of Flash Attention
Flash Attention has evolved through three versions to maximize utilization on progressing hardware generations. FlashAttention-1 pioneered tiling and recomputation, reducing memory usage from quadratic to linear.4 FlashAttention-2 optimized work partitioning and parallelization across the sequence length, achieving significant speedups on A100 GPUs.4 FlashAttention-3, designed for Hopper (H100) architectures, utilizes asynchrony through the Tensor Memory Accelerator (TMA) to overlap data movement with computation.14

Version
Key Innovation
Typical GPU Utilization
Major Feature
v1
Tiling & Recomputation
25-40% (A100)
Reduced HBM reads/writes.4
v2
Better Parallelism
50-70% (A100)
Optimized work across warps.4
v3
Asynchrony & FP8
75% (H100)
TMA/WGMMA hardware support.14

4
Key Takeaways:
IO-Awareness: The speedup is achieved not by changing the mathematics of attention, but by restructuring the computation to fit hardware memory hierarchies.4
No Accuracy Loss: Flash Attention is an exact algorithm; it produces the same results as standard attention without approximation.4
Scaling: It enables the training and inference of much longer sequences by capping memory requirements to  instead of .5
Hardware Specificity: Performance is maximized when the kernel is tuned for specific GPU memory sizes and instruction sets.5
Paged Attention: Virtualized KV Cache Management
While Flash Attention optimizes the calculation of attention, Paged Attention optimizes the storage of the resulting data. In transformer inference, the KV cache is the most volatile memory consumer. Traditional systems pre-allocate contiguous chunks of memory for the KV cache based on the maximum possible sequence length, which leads to massive waste through internal and external fragmentation.2
Virtual Memory Principles in Inference
One-liner: Paged Attention manages AI memory like a computer's operating system, breaking data into small pieces so it can fill every available gap and share information between users.7
Kid-level explanation: Imagine you have a big sticker book. In a normal book, you have to leave empty pages for every story you might write, which wastes a lot of paper. Paged Attention is like a magic binder where you can add one page at a time exactly where you need it. If two people are writing stories that start with the same sentence, they can both share the very first page instead of having two copies.5
Conceptual diagram description: The diagram shows a "Logical View" where User A's context is a long, continuous strip of tokens. Next to it, a "Physical View" shows the GPU memory as a grid of small squares (blocks). A central "Block Table" acts as a map, with arrows showing that User A's consecutive logical tokens are actually stored in non-adjacent physical squares across the grid. Another request, User B, is shown with a different color. A subset of the physical squares is highlighted in both colors, representing "Shared Prefix" blocks that both users are accessing simultaneously to save space.5
Memory Efficiency and Throughput Gains
By treating memory blocks like pages in an operating system, the vLLM engine eliminates external fragmentation and reduces internal fragmentation to the very last block in a sequence. This efficiency allows serving engines to increase batch sizes significantly, as more requests can fit into the same amount of VRAM.7 Furthermore, Paged Attention facilitates advanced decoding algorithms like Beam Search and parallel sampling by allowing multiple logical sequences to point to the same physical memory blocks via reference counting and copy-on-write mechanisms.5

Allocation Type
Fragmented Waste
Memory Sharing
Scheduling Flexibility
Contiguous
60-80%
No
Low.2
Paged
<4%
Yes (Prefix/Beam)
High.2

2
Key Takeaways:
Elimination of Waste: It effectively solves the problem of pre-allocating memory for "worst-case" sequence lengths.7
Dynamic Mapping: The decoupling of logical and physical memory allows for seamless scaling of contexts during generation.18
Increased Density: Higher memory efficiency directly translates to larger batch sizes and higher serving throughput.7
Advanced Sampling: Enables practical use of complex decoding strategies that would otherwise be memory-prohibitive.18
Comprehensive Batching Techniques for Maximum Throughput
Batching is the practice of grouping multiple user requests into a single computational unit. This is necessary because loading the model weights into the GPU cache is an expensive operation; by applying those same weights to multiple users at once, the cost is amortized across the entire batch.21
Static and Dynamic Batching
One-liner: Static and Dynamic batching group requests together before starting, ensuring the model works on many users at once but making everyone wait for the slowest person to finish.21
Kid-level explanation: Imagine a movie theater that won't start the film until 50 people are in their seats. That's Static Batching. Dynamic Batching is the theater saying, "We'll wait until 8:00 PM or until 50 people show up, whichever happens first." In both cases, if the movie ends early for one person, they still have to sit in the theater until everyone's movie is done before the theater can let new people in.23
Conceptual diagram description: The diagram shows a "Batch Cycle." Four horizontal bars represent User 1 through User 4. All four bars start at the same timestamp. User 3 is very short and finishes quickly. The diagram shows User 3’s slot as a "Dead Zone" or "Idle Space" for the remainder of the cycle. The entire block only ends when User 2 (the longest bar) reaches the finish line. Only after this completion point do four new user bars appear in the next cycle.10
Key Takeaways (Static/Dynamic):
Offline Suitability: Excellent for non-interactive tasks like bulk document processing.21
Latency Bottlenecks: The "first token" for a batch is delayed until the batch is formed.23
Inefficiency: Sequence length variance leads to significant underutilization of GPU resources during the tail end of a batch.23
Continuous Batching (In-Flight Batching)
One-liner: Continuous Batching allows new requests to enter the model's "brain" as soon as an old request finishes, keeping the GPU constantly busy without any waiting.23
Kid-level explanation: Imagine an elevator that doesn't wait to go back to the ground floor. As soon as one person gets off on the 5th floor, someone waiting on the 5th floor can hop right in and go to the 10th floor. The elevator stays full and keeps moving up and down without stopping to "reset".23
Conceptual diagram description: The diagram displays a continuous flow of tokens. A vertical "Iteration" axis shows the batch state at each step. At Iteration 5, User A emits an "End" token and disappears. At Iteration 6, the diagram shows User E (a new request) immediately taking User A’s place in the batch. The other users (B, C, and D) are shown continuing their generation uninterrupted. This "slot-filling" behavior is shown as a seamless, non-stop process.10
Key Takeaways (Continuous):
Iteration-Level Scheduling: Scheduling decisions are made at the granularity of individual tokens rather than entire sequences.23
Throughput Optimization: It minimizes the idle time of GPU compute units by maintaining high occupancy.23
Latency Control: It significantly reduces the queuing time for new requests.23
Framework Standard: It is the foundational scheduling logic for vLLM, SGLang, and TensorRT-LLM.23
Speculative Decoding and Medusa: Accelerating the Decode Phase
Speculative decoding is an advanced technique designed to bypass the sequential nature of the decode stage. Instead of generating one token per forward pass, the system "guesses" multiple tokens and uses the primary LLM to verify them in a single, parallel operation.3
The Mechanics of Speculation
One-liner: Speculative decoding uses a fast, smaller model to predict several future words at once, which the big model then checks in one go to save time.26
Kid-level explanation: Imagine you're writing a sentence and a really fast friend tries to guess the next five words. You then read their whole guess at once. If the first three words are perfect, you keep them and just fix the fourth one. This is much faster than you thinking of every single word one by one!.26
Conceptual diagram description: The diagram depicts a "Draft Model" (small) and a "Target Model" (large). The Draft Model is shown generating a linear sequence of 4 candidate tokens:. These are then shown entering the Target Model together. The diagram uses a green checkmark for Token 1 and Token 2, but a red "X" for Token 3. The final output is shown as the accepted prefix plus a new correction token from the Target Model, effectively generating 3 tokens in the time of one.26
Medusa and Tree Attention
Medusa refines this by eliminating the second model entirely. It adds multiple "Medusa Heads" (extra layers) to the main LLM, each trained to predict a different future offset ().3 Using Tree Attention, it explores multiple branches of possibilities simultaneously. If Head 1 predicts two possible words, and Head 2 predicts two for each of those, a tree of 4 candidates is formed. A specialized sparse attention mask allows the LLM to verify the entire tree in a single pass.27

Speculation Type
Mechanism
Efficiency
Requirement
Draft-Target
Separate small model
Moderate
Managing two models.26
Medusa
Multiple heads on one model
High
Extra training/fine-tuning.3
EAGLE
Feature-level extrapolation
Very High
Advanced head architecture.26

3
Key Takeaways:
Breaking Autoregression: It allows for the generation of multiple tokens per forward pass.28
Arithmetic Intensity: By processing more tokens at once during verification, it moves the decode stage from being memory-bound toward being compute-bound.3
Lossless Results: Output quality is maintained because the target model has the final say in accepting or rejecting tokens.3
Parallel Verification: Custom attention masks enable efficient verification of tree-structured candidates without redundant computation.27
Distributed Parallelism Strategies for Massive Models
When models exceed the memory of a single GPU, they must be partitioned across a cluster. The choice of parallelism strategy impacts both the latency of individual requests and the total throughput of the system.31
Tensor and Pipeline Parallelism
Tensor Parallelism (TP) shards individual layers across multiple GPUs. This requires constant communication (all-reduce) between GPUs at every layer, making it ideal for high-bandwidth intra-node setups like NVLink.1 Pipeline Parallelism (PP) divides the model by layers, with different GPUs handling different stages of the transformer stack. While it has lower communication requirements than TP, it introduces "pipeline bubbles" where GPUs at the end of the chain wait for the start of the chain to finish.31

Parallelism Strategy
Partitioning Level
Communication Overhead
Best For
Tensor (TP)
Intra-layer (Weights)
High (All-Reduce)
Intra-node scaling.31
Pipeline (PP)
Inter-layer (Stages)
Low (P2P)
Inter-node scaling.32
Sequence (SP)
Data-level (Tokens)
Moderate
Long context training/prefill.35
Expert (EP)
Sparse-level (MoE)
Variable (All-to-all)
Mixture of Experts models.37

31
Context and Sequence Parallelism
Sequence Parallelism (SP) focuses on splitting the sequence dimension for non-matmul operations like LayerNorm and Dropout, often used alongside TP to save activation memory.31 Context Parallelism (CP) takes this further by partitioning the entire sequence—including the attention computation—across multiple devices. This is the primary enabler for million-token context windows, where the KV cache and activations for a single user would otherwise overwhelm a single GPU.34
Mixture of Experts (MoE) and Sparse Inference
MoE models represent an architectural shift toward conditional computation. Instead of a single dense feed-forward network (FFN), each layer contains multiple "expert" subnetworks. A gating network or "router" selects only a small subset (e.g., top-2 out of 8) for each token.38
The Efficiency of Sparsity
MoE models provide a way to scale model capacity (total parameters) without proportionally increasing the computational cost per token. For example, a model might have 47B parameters but only use 12B active parameters per forward pass.38 This reduces inference latency compared to a dense model of the same total size while maintaining higher quality than a dense model of the active size.41
Key Takeaways (MoE):
Resource Savings: Only relevant portions of the network are activated, reducing FLOPs per token.38
Specialization: Experts tend to specialize in different linguistic patterns or domains (e.g., code vs. prose) during training.38
Load Balancing: Effective serving requires balancing the "expert load" across GPUs to prevent hotspots where one expert is over-utilized, causing bottlenecks.37
Model Compression and Precision Engineering
To further optimize deployment, models are often compressed via quantization, pruning, or distillation to fit on smaller or cheaper hardware.1
Quantization: GPTQ and AWQ
Quantization reduces the bit-precision of weights and activations (e.g., FP16 to INT8 or INT4).1
GPTQ: Uses second-order information to minimize the output error on a layer-by-layer basis. It is widely compatible and supports aggressive compression down to 2-bit or 3-bit.45
AWQ: Specifically focuses on "salient" weights—those that have the largest impact on activations. By keeping these critical weights accurate, AWQ often achieves better performance for instruction-tuned models at 4-bit precision than GPTQ.45
Pruning and Distillation
Pruning removes redundant weights or neurons that contribute minimally to the model's output.1 Knowledge Distillation trains a smaller "student" model to mimic a larger "teacher" model. The student learns the teacher's probability distributions, allowing a 7B model to capture much of the performance of a 70B model.1

Technique
Analogy
Target
Primary Gain
Quantization
Shorthand notes
Bit-precision
Memory reduction.44
Pruning
Trimming a tree
Network structure
Compute reduction.44
Distillation
Student-Teacher
Model architecture
Size/Latency reduction.44

1
Optimization Metrics and Performance Evaluation
Effective inference optimization requires monitoring key performance indicators (KPIs) to ensure user experience and cost-efficiency.
Time to First Token (TTFT): Measures the latency of the prefill stage. Critical for the perceived responsiveness of an application.8
Inter-Token Latency (ITL): The time between consecutive tokens during the decode stage. This determines the "reading speed" of the model output.8
Tokens Per Second (TPS): The total throughput of the system across all active users.22
Memory Bandwidth Utilization (MBU): A measure of how efficiently the system is streaming data from VRAM. Higher MBU indicates better optimization during the memory-bound decode stage.32

Metric
Phase
User Experience Impact
Target
TTFT
Prefill
Waiting to start
<200ms.8
ITL
Decode
Fluidity of reading
<50ms.8
TPS
Total
Cost per request
Higher is better.22

8
Conclusion on Integrated Inference Infrastructures
The optimization of LLM inference has transitioned from a series of isolated tricks to a holistic engineering discipline. The integration of Flash Attention and Paged Attention provides the memory efficiency required to handle massive contexts and high concurrency.7 Continuous batching maximizes GPU utilization by breaking the rigid cycles of traditional batch processing.23 Meanwhile, speculative decoding and Medusa offer a path forward to overcome the inherent sequential bottlenecks of autoregressive models.3
As models continue to grow, the next frontier lies in disaggregated serving architectures—physically separating the prefill and decode stages onto different hardware optimized for their respective compute and memory profiles.9 This allows for independent scaling and prevents the "interruptive prefill" problem where a single long prompt stalls the generation of tokens for dozens of other users.13 By combining these structural optimizations with precision engineering techniques like AWQ and MoE, the industry is moving toward a future where near-instantaneous, high-reasoning AI is both technically feasible and economically sustainable.
