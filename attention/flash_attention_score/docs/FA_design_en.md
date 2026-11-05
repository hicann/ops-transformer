## 1 Computation Process

Figure 1 Training computation flowchart

![Flowchart](./fig/FlashAttentionScore.png)

It is implemented according to the forward computation flow of FlashAttention. The overall computation flow is as follows:

1. After query and the transposed key undergo matmul computation, the initial attention_score is obtained. Then it is added to the positional encoding pse and multiplied by the scale factor scale_value. The result is processed by the select operation with atten_mask, masking the positions where atten_mask is true, to obtain masked_attention_score. That is, after select, the positions where atten_mask is true have a tiny negative value, which becomes 0 after the softmax computation, achieving the masking effect.

2. To implement the FlashAttention acceleration algorithm, the FlashSoftmax operation is used to compute masked_attention_score, replacing the softmax operation in the original formula, and then the result is multiplied with value by matmul. Since the FlashSoftmax operation splits masked_attention_score along the Skv direction (the sequence length of the input key and value), a refresh process exists in the implementation, as follows:
    1. Each FlashSoftmax computation operates on only one SkvSplit after splitting (SkvSplit is short for the sequence length after splitting along the Skv axis), and exp is recorded starting from the second loop iteration, where i is the loop variable after Skv splitting. For exp, i starts from 1. The formula for exp is as follows:
       $$
       exp[i] = e^{max_{i - 1} - max_{i}}
       $$
    2. When i = 0, the computed MM[PV] result is directly saved to the ub of ub_attention_out[0].
    3. Starting from i = 1, Mul and Add operations need to be added. That is, the previous MM[PV] result is multiplied by the current exp, and the product is added to the current MM[PV] result. The sum is saved to the ub of ub_attention_out[1]. This process repeats until the traversal of Skv is complete.
    4. Since the division by sum in the FlashSoftmax computation is postponed to before the output of attention_out, the ub_attention_out in the ub needs to be divided by softmax_sum row by row at the end, and the final complete result is saved to the output memory attention_out(Final).

## 2 Entry Point of Each Computation Stage

Taking the S1 template as an example

1. After obtaining the number of task blocks processed by each core, call the function: `Process`
2. Cube1 stage (Cube1 in the preceding figure), computation entry: `IterateBmm1`
3. Vector1 stage (Vector1 in the preceding figure), computation entry: `ProcessVec1`
4. Cube2 stage (Cube2 in the preceding figure), computation entry: `IterateBmm2`
5. Vector2 stage (Vector2 in the preceding figure), computation entry: `ProcessVec2`

## 3 Tiling Design

1. Atlas A2 training products
  The purpose of tiling is to find a more efficient NPU execution method. The original data volume is generally very large, and all computation cannot be completed with a single instruction call. Therefore, the data volume needs to be distributed across multiple cores for parallel computation. Each core also needs to consider how to achieve the best performance in cyclic computation. Different inputs may have different optimal execution methods, so the tiling policy needs to decide how to allocate data to the cores for computation.

  Based on the hardware architecture, the AI Core is divided into two independent cores: AIC and AIV. The AIC and AIV cores have their own independent Scalar computing units, can independently load their own code segments, and execute separately. The separated AIC/AIV architecture allows AIC and AIV to execute in parallel. The paths for data exchange between AIC and AIV are L2 and GM (Global Memory, high-bandwidth memory), and the number of interactions between them greatly affects performance. In addition, due to the computing power difference between AIC and AIV, the two cores need to use different basic block sizes. Guided by the principle of minimizing the number of communications between AIC and AIV and maximizing the computing power, the CVtiling separation strategy was developed. This strategy effectively reduces the number of CV communications and selects different basic blocks for computation based on the buffer characteristics of different units, thereby improving operator performance.

  For the FA operator, Vector computation involves multiple inputs, outputs, intermediate computation results, and double-buffer designs, so the buffer needs to be divided into multiple parts. In the optimal allocation scheme, the largest part is 32KB. Since the data type used by Vector computation is float32, the basic tiling block of Vector is 8 *1024. To fully leverage the computing power of the Cube, the data volume of one round of computation is allocated at a ratio of 1:16 between C and V. In addition, since the input data type on the Cube side is float16 and the output is float32, the basic block of the Cube is 128* 128. Therefore, a data volume of 128 * 1024 is allocated through nRatio=8. The pseudocode is as follows:

  ```c++
  // C-Tiling: (S1_c_i,D)x(D,S2_c_i) => (S1_c_i, S2_c_i):(128,1024)
  // V-Tiling: (S1_v_i, S2_v_i) => (8,1024)

  // Matmul computation on the C side
  Bmm((S1_c_i,D)x(D,S2_c_i)) => 128*1024  // Output result 128*1024, stored on the workspace
  // Vector computation on the V side
  for S1_c_i/S1_v_i=128/8:
    copy_gm_to_ub(S1_v_i*S2_v_i)  // Copy the bmm result data from the bmm workspace
    Vector(S1_v_i,S2_v_i)         // Perform Vector computation
    copy_ub_to_gm(S1_v_i*S2_v_i)  // Vector computation completes and the final output data is obtained; copy it to GM

  // Since the data computed on the Cube side is larger than that on the Vector side, Vector tiling needs to be performed again within the UB, which produces the S1-dimension ratio: S1_c_i/S1_v_i
  ```

  In the preceding example, the ratio is enabled only in the S1 dimension, and the C/V computation lengths in the S2 dimension are the same. Of course, the ratio can also be enabled in both the S1 and S2 dimensions. The benefit is that the Cube can launch a large block of data at a time, avoiding the communication overhead caused by continuously launching small data blocks, and the buffer of the Cube unit can be used to the greatest extent.

2. Ascend 950PR&950DT products
  Ascend 950PR&950DT products support both the separated AIC/AIV architecture and the fused AIC/AIV architecture. The interaction paths between AIC and AIV include L2, GM, and UB (Unified Buffer). UB is a more efficient interaction path between AIC and AIV than GM. AIC can directly output to the UB, and AIV uses the data on the UB to perform vec computation, which greatly reduces the data movement time. Since the UB capacity is limited, when D_v>128, the output of mm2 still needs to be saved to GM first; the output of mm1 is always saved in the UB.

## 4 Pipeline Design

To achieve ultimate performance, hardware resources must be fully utilized, and pipeline designs for different pipelines are usually required. The purpose of pipeline design is to make a certain pipeline reach the bound state as much as possible, so that a certain hardware unit keeps working and the performance upper limit is reached.

### 4.1 V-Side Pipeline

The V-side pipeline design needs to consider the data movement and computation processes of Vector. The main optimization method is double buffer.
Take the following pipeline task diagram as an example. Vec uses ping-pong to represent two data processing tasks, and each task needs to perform DataCopy movement and Clc computation in sequence. Data dependencies exist between tasks; for example, Clc can be processed only after DataCopy is completed.
After double buffer is applied between the ping-pong pipelines of the pipeline tasks in the figure, the run diagram shows that for the same piece of data, DataCopy movement and Clc computation are processed serially, while for different data slices, multiple tasks can be processed in parallel at the same time point, thereby achieving task parallelism and improving performance.

![Design Diagram 2](./fig/fa_design_diagram_2.png)

The memory resources occupied by the ping and pong computation data are independent of each other.
​The V-side computation process of FA-class fused operators involves many steps and is complex. A simple double buffer usually cannot cover all cases, so different computation pipeline arrangements appear. Different computation pipelines are applicable to different types of shape features, so as to achieve the best pipeline design for each type of feature.

### 4.2 CV Pipeline

1. Atlas A2 training products
  Fused operators usually include both Vector computation and Cube computation. For the FA operator, the V-side computation depends on the C-side computation result. If only the V-side pipeline is considered and the C side is ignored, the C side and the V side are likely to run as serial pipelines, which cannot achieve parallel computation or optimal fused operator performance. Therefore, the CV pipeline design was introduced. In addition, the CV pipeline behaves differently for different operators. The Cube dual-launch mechanism of FA (also known as inter-CV preload pipeline) enables pipeline optimization:

    Under the pipeline characteristics of this scenario, there are few Vector computation nodes and the computation speed is fast. In the case of C:V=1:2 on <term>Atlas A2 training products</term>, the Cube data movement duration is sufficient to cover the Vector computation duration, so only the MTE2 time consumption of the Cube needs to be considered, finally achieving the MTE2 bound. Under the Cube dual-launch mechanism, two Cube computations are launched in advance, and Cube1 and Cube2 computations can be connected, maximizing Cube utilization and achieving the Cube bound.

  ![FA Pipeline.jpg](../../../docs/zh/figures/fa_pipeline.png)

2. Ascend 950PR&950DT products
  The CV pipeline design idea of Ascend 950PR&950DT products is basically the same as that of A2. The difference is that the number of Cube preload operations is 3: the mm2 computation starts only after three mm1 computations are completed. The purpose is to optimize the CV pipeline in the startup phase and make it tighter, so as to achieve the best overall performance.

## 5 Multi-Template Design

To allow different inputs to reuse the same tiling and pipeline, fused operators are implemented using templates. However, when all different inputs use the same set of templates, optimal performance and functional generalization cannot be achieved. Therefore, different templates need to be implemented based on the characteristics of the input shape.
The multi-template design idea of the FA (FlashAttentionScore, FA for short) fused operator is mainly as follows:

- **Template splitting based on intra-core and inter-core splitting**

  Since the hardware buffer size is limited and the data volume to be computed is huge, the computation cannot be completed at once, so tiling splitting is required. Different shapes lead to different splitting axes of the operator, and the splitting axes of the operator affect the functions and performance of the template. Simple elewise operators often fuse all axes into one axis for splitting, with simple logic, so their templates are relatively simple. However, fused operators combine multiple types of scenarios such as elewise, broadcast, reduce, and matmul, and their functions are complex. To meet high performance requirements, templates usually need to be split based on the splitting axes. To achieve optimal performance when splitting templates, the following points need to be considered:

  a. Use all cores to prevent some cores from being idle.

  b. The computation volume assigned to each core is relatively even, avoiding a situation where some cores compute excessively large amounts of data while the remaining cores stay idle.

  c. The data volume processed between AIC and AIV must match their corresponding computing power, preventing long idle periods of AIC or AIV.

  The FA operator contains five axes in total: B, N2 (the N of key and value), G (query_N/kv_N), S1 (the S of query), and S2 (the S of key and value). The splitting order is intra-core first and then inter-core. Intra-core splitting selects the splitting axes based on the basic block size, and inter-core splitting merges the remaining axes after intra-core splitting and then splits them based on the number of AI Cores. The splitting axes change with different shape sizes.

    - Atlas A2 training products: From the Vector perspective, the FA operator is divided into the following template types. Templates are prioritized by sequence number; the smaller the number, the higher the priority, and the earlier the template is matched.

    <table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
      <col style="width: 170px">
      <col style="width: 170px">
      <col style="width: 310px">
      <col style="width: 212px">
    </colgroup>
    <thead>
      <tr>
        <th>Template</th>
        <th>Splitting Axis</th>
        <th>Entry Condition</th>
        <th>Applicable Scope</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>TNDSameAB template</td>
        <td>Split S1 in UB</td>
        <td>(B >= 4 and accumS1 >= 8192 and accumS2 >= 8192 and maxS1 >= 512 and maxS2 >= 512) or
          (maxS2 >= 5120 and maxS1 >= 5120). </td>
        <td>TND scenario</td>
      </tr>
      <tr>
        <td>TND template</td>
        <td>Split S1S2 in UB</td>
        <td>TND scenario, but the TNDSameAB template conditions are not met. </td>
        <td>TND scenario</td>
      </tr>
      <tr>
        <td>SameAB template</td>
        <td>Split S1 in UB</td>
        <td>Non-FP32, S2 >= 512 and D % 16 != 0 or D = 96, or S2 > 1024 and 128 < D < 196. </td>
        <td>General scenario</td>
      </tr>
      <tr>
        <td>S1S2 template</td>
        <td>Split S1S2 in UB</td>
        <td>S2 > 1024 and the preceding conditions are not met. </td>
        <td>General scenario</td>
      </tr>
      <tr>
        <td>S1 template</td>
        <td>Split S1D in UB</td>
        <td>(N2 * G * ((alignedS1 + alignedS2) * alignedD + alignedS2) * dtypeSize) >= 256 * 1024 or (N2 * G * (alignedS1 + alignedD) * alignedS2 * dtypeSize) >= 256 * 1024 or N2 * G * alignedS1 * alignedS2 * dtypeSize > 65536 * 2.</td>
        <td>General scenario</td>
      </tr>
      <tr>
        <td>B template</td>
        <td>Split B across cores</td>
        <td>The preceding conditions are not met.</td>
        <td>General scenario</td>
      </tr>
    </tbody>
    </table>

    Each type of template has its own unique UB and Block splitting axes and can handle scenarios with inputs of a specific shape feature. The template is designed for that type of shape feature.
    - Ascend 950PR&950DT products

    Since the basic block for intra-core splitting on Ascend 950PR&950DT products is 128 * 128, optimal performance can be achieved for all shapes. Therefore, only one set of templates is designed, supporting all shapes. This set of templates also does not distinguish whether the layout is TND.

- **Template specialization based on special scenarios and specific optimizations**

  Each fused operator has a basic template, supplemented by multiple specialized templates:

    - The basic template covers functions and the performance of most inputs with shape features of this type.
    - Specialized templates cover the ultimate performance of specific scenarios. They are mainly optimized using special methods applicable to specific scenarios and are not suitable for generalization.

  For example, the empty_input template specialized for the special empty tensor scenario, and the deterministic computation template designed based on the role-based Cube core management (RCM) optimization solution.
- **Template specialization based on different computation pipelines**

  To fully leverage hardware advantages, fused operators usually require pipeline design to improve fused operator performance. Different pipeline designs greatly affect the code architecture. To improve the maintainability, testability, and readability of the code, templates need to be specialized based on different computation pipelines to achieve the ultimate performance in specific scenarios.

### 5.1 Detailed Multi-Template Design

- **Basic concepts:**

  **CV basic block:** Indicates the data volume computed by the Cube or Vector in a single computation. It describes the data volume of a complete interaction between the Cube and Vector, and is usually equivalent to the **inter-core basic block**. On the chip, communication between the Cube and Vector has a certain overhead, so the CV basic block is set to a large value. A suitable data volume is generally 512*1024 (unit: Bytes). Since the intra-core buffers of the Cube and Vector are limited, the inter-core basic block may need to be completed through **multiple intra-core computations**.

  **Intra-core basic block:**

  If the CV basic block is too large, the Cube and Vector cores further split the CV basic block into basic blocks that fit the intra-core L0A, L0B, L0C, UB, and other buffer sizes. This is called the **intra-core basic block**. On the Cube side, the intra-core basic block is generally 32KB (unit: Bytes), which allows the DoubleBuffer capability of L0A and L0B to be fully utilized, and the computing power and bandwidth can also be used as fully as possible. On the Vector side, the basic block size is generally 32KB (unit: Bytes).

  **xxx.i:** Indicates the size of an axis in the CV basic block after splitting. A basic block is generally 2-dimensional, and xxx.i indicates the size of one of its dimensions. xxx can be B, N2, G, S1, or S2;
  For example, if S1 = 512, S2 = 1024, S1.i = 64, and S2.i = 128, [S1, S2] is split into basic blocks of size [64, 128]. The basic block size of the S1 axis is 64, and the basic block size of the S2 axis is 128.

  **xxx.o**: Indicates the number of segments into which an axis is divided after basic block splitting. xxx can be B, N2, G, S1, or S2;

  For example, if S1 = 512, S2 = 1024, S1.i = 64, and S2.i = 128, [S1, S2] is split into basic blocks of size [64, 128], S1.o = 512 / 64 = 8, and S2.o = 1024 / 128 = 8, so a total of 8 * 8 basic blocks are split.
  - Atlas A2 training products: The FA operator is divided into the following template types based on the splitting axes. For the following FA templates, the S2 axis is currently not split across multiple cores:
    > 1. Template that splits the B, N2, G, and S1 axes across cores and splits the S1 and S2 axes within a core. This template is the most general template and supports all inputs (except TND):
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score/op_host/flash_attention_score_tiling_general.cpp
    >
    >    tiling code class: FlashAttentionScoreTilingS1s2Bn2gs1
    >
    >    kernel code: ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_s1s2_bn2gs1.h
    >
    >    Condition: S2 > 1024;
    >    Rationale: Since the shape of the FlashSoftmax computation on the Vector side is [S1, S2], and the FlashSoftmax operation requires the S2 split size to be as large as possible, the basic block of the S2 axis is usually set to 1024 in FlashAttention. In addition, since FlashSoftmax has a refresh process that continuously updates the Softmax result when the S2 axis is split, more Softmax update processes appear in the computation flow when S2 is greater than 1024, which differs from other templates.
    >    CV basic block selection:
    >    S1.i: The default value is 64. When splitting by 64, if B *N2* G * S1.o exceeds the number of Vector cores, S1.i is set to 128. The purpose is to fully use all cores first when S1 is small, since full use of multiple cores delivers higher performance.
    >    S2.i: 1024
    >
    >
    >
    > 2. Template that splits the B, N2, G, and S1 axes across cores and splits the S1 axis within a core
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score/op_host/flash_attention_score_tiling_general.cpp
    >
    >    tiling code class: FlashAttentionScoreTilingS1Bn2gs1
    >
    >    kernel code: ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_s1_bn2gs1.h
    >
    >    Condition: (128 < S2 <= 1024) ||  (N2 *G* (align(S1,16) + align(S2,16)) *align(D,16)* sizeof(INPUT_T)>= 512KB) || (N2 *G* (align(S1,16) + align(D,16)) *align(S2,16)* sizeof(INPUT_T))
    >
    >    Rationale: When S2 < 1024, S2 is not split, so the FlashSoftmax result does not need to be updated, making the flow simpler. The generalization template described in point 1 is not used. Since the B-splitting template does not bring performance improvement when S2 > 128, this template is used by default. In addition, when S2 <= 128, if the input of matmul1 or matmul2 without the Batch axis has filled the entire L1, the B-splitting template is also not used. The B-splitting template means splitting the batch axis, and the inter-core basic block size is B.i *N2* G *S1* D (the D of query), or B.i *N2* G *S2* D (the D of key/value), or B.i *N2* G *S1* S2 (the Softmax result, that is, P). If B splitting meets the splitting requirements, B.i is at least 2, so the product of these axes inside B.i must be smaller than the L1 size. The condition check above is based on this: if the input of either matmul in Q *K or P* V is greater than the L1 size, using the B-splitting template brings no benefit.
    >
    >    CV basic block selection:
    >
    >    S1.i: It is dynamically adjusted based on the size of S2 to make the data volume of S1 * S2 as large as possible, in order to reduce the number of communications and the communication overhead.
    >
    >    S2 is not split, and the basic block size of S2 = S2
    >
    >
    >
    > 3. Template that splits the B axis across cores. On the intra-core Cube side, S1 and S2 are not split; B.i, N2, and G are used as loop axes to open loops for processing batch matmul. The Vector core comprehensively splits B.i *N2* G * S1 intra-core to find the most suitable intra-core basic block.
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score/op_host/flash_attention_score_tiling_general.cpp
    >
    >    tiling code class: FlashAttentionScoreTilingB
    >
    >    kernel code: ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_bn2gs1s2_b.h
    >
    >    Condition: Other shapes that cannot enter the preceding two templates
    >    Rationale: When S1, S2, and D are all small, the CV basic block is small. B.i, N2, and G are also included in the CV basic block, so the data volume of one CV interaction is larger, improving execution performance.

  - Ascend 950PR&950DT products
    > 1. Template that splits the B, N2, G, and S1 axes across cores and splits the S1 and S2 axes within a core. This template is the most general template and supports all inputs (except TND):
    >
    >    tiling code file: attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp
    >
    >    tiling code class: FlashAttentionScoreTilingS1S2Const, FlashAttentionScoreTilingVarLenConst
    >
    >    kernel code: attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h
    >
    >    CV basic block selection:
    >    S1.i: 128, or 64 in a few scenarios
    >    S2.i: 128, or 256 in a few scenarios

## 6 Programming Perspective

### 6.1 AscendC High-Level APIs

Currently, AscendC high-level APIs provide two programming modes. In the first mode, Vector is the main core and Cube is the secondary core. Vector0 and Vector1 independently launch Matmul tasks, and the two are independent of each other.

In the second mode, Cube is the main core and Vector is the secondary core. V0 uniformly launches the Matmul task, and the result of this task is processed by V0 and V1 together; generally, V0 and V1 each process half. Currently, if a template name ends with "_sab", the template is a template with Cube as the main core. For example:

```c++
ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_s1s2_bn2gs1_sab.h

ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
```

For FlashAttention, using Cube as the main core allows the Matmul tasks of V0 and V1 to reuse the left matrix, and part of the output results can be accumulated in L0C, reducing the bandwidth dependency. Performance is better in most scenarios.

### 6.2 AscendC Low-Level APIs

Currently, some templates still do not use AscendC high-level APIs. For example:

```c++
ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h
```

This template uses Cube as the main core and Vector as the secondary core more thoroughly. All Matmul tasks are fully launched from the Cube side, and the Vector side is notified through synchronization.

On Ascend 950PR&950DT products, FA is implemented using AscendC low-level APIs.
