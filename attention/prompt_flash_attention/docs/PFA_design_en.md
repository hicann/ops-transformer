Declaration: This document uses the [Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). For reprinting, citation, or modification, follow this license.

# PromptFlashAttention Operator Design Introduction

To continuously optimize and improve incremental inference performance, the requirement for the PromptFlashAttention fused operator that supports full-sequence inference is proposed.

## 1 Implementation Principle

It is implemented according to the forward computation flow of flashAttention. The overall computation flow is as follows:

1. A matmul computation of query and the transposed key produces the initial attention_score. Then the position encoding pse is added, and the scaling coefficient scale_value is multiplied. The result then goes through a select operation with atten_mask, which masks the positions where atten_mask is true, producing masked_attention_score. That is, after the select, positions where atten_mask is true become an extremely small negative value, which becomes 0 after the softmax computation, thus achieving the masking effect.

2. To implement the FlashAttention acceleration algorithm, the FlashSoftmax operation is applied to masked_attention_score to replace the softmax operation in the original formula, and then the result is multiplied with value through matmul. Because the FlashSoftmax operation splits masked_attention_score along the Skv direction (the sequence length of the input key and value), a refresh process exists in the implementation, as follows:

   1. Each FlashSoftmax computation operates on only one SkvSplit (SkvSplit is short for the sequence length after splitting along the Skv axis) after splitting. Starting from the second loop iteration, exp is recorded, where i denotes the loop variable after Skv splitting. For exp, i starts from 1. The formula for exp is:
      $$
      exp[i] = e^{max_{i - 1} - max_{i}}
      $$

   2. Starting from i = 1, Mul and Add operations need to be added. That is, the MM[PV] result of the previous iteration is multiplied by the current exp, and the product is added to the MM[PV] result of the current iteration. The sum is saved to GM. This continues until the traversal over Skv is complete.

   3. Because the division by sum in the FlashSoftmax computation is moved to just before outputting attention_out, attention_out in the UB must finally be divided by softmax_sum row by row, and the final complete result is saved to the output memory attention_out(Final).

## 2 Template-Based Design

### 2.1 Template Types

To allow different inputs to reuse the same tiling and pipeline, the fused operator is implemented in a template-based manner. However, using the same set of templates for all inputs cannot achieve optimal performance and functional generalization. Therefore, templates need to be differentiated according to the characteristics of input shapes. The multi-template design approach of the fused operator is mainly as follows:

1. General capability templates divided by product form:
(1) The A2/A3 scenario general capability template supports <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.
(2) For Atlas inference products, the general capability template is enabled.

2. Templates developed for specific business scenarios:
(1) For the MLA scenario, the MLA basic API template is developed.
(2) In the A2/A3 environment, for the standalone interface and the scenario where the graph mode is used and D is not aligned to 32, the special template implementation is supported.
(3) For inputs where a certain axis dimension of the key/value/attentionOut matrix is 0, the empty tensor template is provided.

### 2.2 Data Splitting

  Because the hardware buffer size is limited while the amount of data to compute is huge, computation cannot be completed in one pass. Therefore, tiling splitting is required. Different shapes lead to different splitting axes of the operator, and the splitting axes affect the function and performance of the template. Template splitting is often performed based on the splitting axes. To achieve optimal performance during template splitting, consider the following points:

  a. Use up all cores to prevent some cores from being idle.

  b. Distribute the computation amount assigned to each core relatively evenly to avoid the situation where some cores compute an excessive amount of data while other cores are idle.

  c. Keep the amount of data processed between AIC and AIV consistent with their computing power, so that neither AIC nor AIV remains idle for a long time.

  The PromptFlashAttention operator contains 5 axes in total: B, N2 (N of key and value), G (query_N/kv_N), S1 (S of query), and S2 (S of key and value). The splitting order is intra-core first and then inter-core. Intra-core splitting selects the splitting axis based on the basic block size, and inter-core splitting merges the remaining axes after intra-core splitting and then splits them according to the number of AI Cores.

## 3 tiling design

The purpose of tiling is to find a more efficient NPU execution method. The original data volume is generally very large and cannot be processed with a single instruction call. Therefore, the data volume needs to be distributed to multiple cores for parallel computation. In addition, each core needs to consider how to loop for optimal performance. Different inputs may have different optimal execution methods. Therefore, a tiling strategy is required to decide how to allocate data to each core for computation.

### 3.1 CV Tiling Separation Design

  According to hardware architecture characteristics, an AI Core is divided into two independent cores: AIC and AIV. The AIC and AIV cores have their own independent scalar computation units and can independently load their own code segments and execute separately. The AIC/AIV separated architecture allows AIC and AIV to execute in parallel. The data interaction path between AIC and AIV is L2 and GM (Global Memory, DDR memory), and the number of interactions between them has a significant impact on performance. Meanwhile, due to the computing power difference between AIC and AIV, they need to use different basic block sizes. Following the principle of minimizing the number of AIC/AIV communication rounds and maximizing computing power, the CV tiling separation strategy emerges. It can effectively reduce the number of CV communication rounds, and according to the buffer characteristics of different units, different basic blocks are selected for computation, thereby improving operator performance.

  For the PromptFlashAttention operator, Vector computation involves multiple inputs, outputs, intermediate computation results, double-buffer design, and so on, so the buffer needs to be allocated into multiple parts. In the optimal allocation scheme, the largest part is 32KB. Because the data type used by Vector computation is float32, the basic block of Vector tiling is 8 *1024. To fully utilize the computing power of the Cube, a 1:16 ratio of the data volume per round is adopted between CV. In addition, because the input data type on the Cube side is float16 and the output is float32, the basic block of the Cube is 128* 128. Therefore, through the nRation=8 ratio, a data volume of 128 * 1024 is derived.

### 2.2 tilingkey design

To instantiate a definite template at runtime, a unique identifier through tilingkey is required.

**tilingKey generation rule**: Assemble the tilingkey in decimal digits. The initial value is 0. Each attribute/function corresponds to a different magic number, which is added in sequence to obtain the tilingkey.

The tilingkey design of the PFA operator is as follows:
In the Atlas inference products scenario:
> **1.** The magic number 12288 serves as the identifier of the current scenario.
> **2.** Distinguished by the layout of the input query/key/value matrices: BNSD: +10000.
> **3.** Distinguished by the configured precision mode: input FP16 with the high-precision mode: +600.

In the A2/A3 scenario:
> **1.** When the input query matrix is of the FP16/BF16 type and the D axis has no tail block:
> (1) The magic number 1012 serves as the identifier of the current scenario.
> (2) Divided by input data type: if the input query matrix is FP16 with the high-precision mode configured, or the input query matrix is of the FP16 type, key/value are of the INT8 type, and MSD quantization is enabled: +600; if the input query matrix is of the BF16 type: +100.
> (3) Divided by output data type: output BF16: +10000; output INT8: +20000; others: +0.
> (4) Divided by the input layout: input in BSH/BSND format: +100000.
> (5) Divided by the core splitting method: using SPLIT_NBS_CUBE/SPLIT_ONEN_CUBE: +2000000.
> (6) The input query matrix is of the FP16 type, key/value are of the INT8 type, and MSD quantization is not enabled: +800000000000.
> **2.** When the input query matrix is not of the FP16/BF16 type, or the D axis has a tail block:
> (1) The magic number 10 serves as the identifier of the current scenario.
> (2) Divided by the input layout: BNSD format: +5.
> **3.** Superposition of other functions
> (1) PagedAttention enabled: +10000000U.
> (2) Prefix sharing enabled: +100000000.
> (3) MSD pseudo-quantization enabled: +400200000000.

**For other specialized scenarios, you can define your own bit fields and values in sequence afterward.**

## 3 pipeline design

To achieve ultimate performance, hardware resources must be fully utilized, and pipeline designs for different pipelines are usually required. The purpose of pipeline design is to make one pipeline reach the bound effect, so that a certain hardware unit keeps working and the performance limit is reached.

### 3.1  V-side pipeline

The V-side pipeline design needs to consider the data movement and computation processes inside the Vector. The main optimization method is double buffer.
Take the following pipeline task diagram as an example. The Vec function is split into 2 pipeline tasks: subA and subB, and each task focuses on completing a single function. The data to be processed is split into 2 pieces. ping-pong represents the two data processing tasks, and each task needs to perform the DataCopy movement and the Clc computation in sequence. The arrows between tasks indicate data dependencies. For example, subB can process Clc only after subA finishes DataCopy.
As shown in the figure, without pipeline design, the movement and computation tasks execute serially, causing a stream interruption. That is, after the first DataCopy completes, the movement pipeline remains idle until the data moved in the first round has been computed and moved out. Only then does the movement pipeline continue to work and perform the second DataCopy (the Vector computation and move-out pipelines have the same problem). Usually in this case, the performance is extremely poor.

After applying double buffer processing between the ping-pong pipelines for the pipeline tasks in the preceding figure, the running diagram of the pipeline tasks is as follows. From the running diagram, for the same piece of data, the DataCopy movement and the Clc computation have a dependency and need to be processed serially. For different data slices, multiple tasks can be processed in parallel at the same time point, thereby achieving task parallelism and improving performance.

The pseudocode for implementation is as follows:

```python
def Vec():
    subA(ping) # Compute the ping block
    subA(pong) # Compute the pong block
    subB(ping) # Compute the ping block
    subB(pong) # Compute the pong block
```

The memory resources occupied by the ping and pong computation data are independent of each other.
​The PromptFlashAttention-like fused operators involve many V-side computation processes, and the situations are complex. A simple double buffer usually cannot cover all cases, so different computation pipeline arrangements appear. Different computation pipelines are applicable to different classes of shape characteristics to achieve the best pipeline design for that class of characteristics.

### 3.2 CV pipeline

Fused operators usually contain both Vector computation and Cube computation. For the PromptFlashAttention operator, V-side computation depends on C-side computation results. If only the V-side pipeline is considered and the C side is ignored, the C side and the V side are likely to present a serial pipeline effect, failing to achieve parallel computation and optimal fused operator performance. Therefore, the CV pipeline design is introduced. In addition, the CV pipeline behaves differently for different operators. The Cube dual-issue mechanism of PromptFlashAttention can achieve pipeline optimization in two scenarios:

- Total time consumed on the C side > Total time consumed on the V side

  Under the pipeline characteristics of this scenario, there are few Vector computation nodes and the computation is fast. In the case of <term>Atlas A2 products</term> with C:V=1:2, the Cube movement duration is sufficient to cover the Vector computation duration, so only the Cube MTE2 time consumption needs attention, ultimately achieving MTE2 bound. Under the Cube dual-issue mechanism, two Cube computations are issued in advance, and the Cube1 and Cube2 computations can be connected, maximizing Cube utilization and achieving Cube bound.

  Fused operators usually contain both Vector computation and Cube computation. For the FA operator, V-side computation depends on C-side computation results. If only the V-side pipeline is considered and the C side is ignored, the C side and the V side are likely to present a serial pipeline effect, failing to achieve parallel computation and optimal fused operator performance. Therefore, the CV pipeline design is introduced. In addition, the CV pipeline behaves differently for different operators. The Cube dual-issue mechanism of FA can achieve pipeline optimization:

  Under the pipeline characteristics of this scenario, there are few Vector computation nodes and the computation is fast. In the case of <term>Atlas A2 training products</term> with C:V=1:2, the Cube movement duration is sufficient to cover the Vector computation duration, so only the Cube MTE2 time consumption needs attention, ultimately achieving MTE2 bound. Under the Cube dual-issue mechanism, two Cube computations are issued in advance, and the Cube1 and Cube2 computations can be connected, maximizing Cube utilization and achieving Cube bound.

  ![FA pipeline.jpg](../../../docs/zh/figures/fa_pipeline.png)

- Total time consumed on the C side < Total time consumed on the V side

  Under the pipeline characteristics of this scenario, there are many Vector computation nodes and Vector computation is the bottleneck. The C-side movement is insufficient to cover the V-side pipeline. Therefore, CV pipeline arrangement is required to achieve the CV parallel effect as much as possible. The most common optimization method is early C-side issue.

  The C side issues two Cube computations consecutively. This ensures that when the V side finishes computing the previous round, it can immediately start the computation of the current round without waiting for the data of Cube1. In this way, the V side keeps working, achieving Vector bound.
