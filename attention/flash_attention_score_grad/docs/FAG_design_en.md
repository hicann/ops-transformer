## 1 Computation Process

Figure 1 Training computation flow diagram

![Flow diagram](./fig/FlashAttentionScoreGrad.png)

This operator is implemented following the FlashAttention backward computation flow. The overall computation flow is as follows:

1. Recompute p: $p = SimpledSoftmax(Mask(Matmul(query, key^T) + pse\_shift) * scale)$. This step recomputes the softmax result p from the FA forward process, and the computation result is saved in the UB.
2. Compute dp: $dp = Dropout(Matmul(dy, value^T))$. This computation includes a matmul computation and a dropout computation. In the matmul computation, the left matrix is dy and the right matrix is the transposed value.
3. Compute ds: $ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention\_in))$. In this computation, the inputs of the FlashSoftmaxGrad computation are dy and the forward output attention\_in. The result of FlashSoftmaxGrad is subtracted from dp, and the final result is multiplied by p to obtain ds.
4. Compute dq: $dq = Matmul(ds, key) * scale$. This computation performs matmul on ds and key, and multiplies the result by scale to obtain dq.
5. Compute dk: $dk = Matmul(ds^T, query) * scale$. This computation performs matmul on the transposed ds and query, and multiplies the result by scale to obtain dk.
6. Compute dv: $dv = Matmul(DropOut(p)^T, dy)$. In this computation, the result p is processed by dropout, transposed, and then multiplied by dy through matmul.

## 2 Entry Point of Each Computation Stage

This section uses the sameAB template as an example.

1. After obtaining the number of task blocks processed by each core, call the function: `Process`
2. Cube1 stage, step 1 in the flow, computation entry: `ComputeMM1`
3. Vector1 stage, steps 2-3 in the flow, computation entry: `ComputeVec`
4. Cube2 stage, steps 4-6 in the flow, computation entry: `ComputeMMDqkv`

## 3 Tiling Design

Tiling aims to find a more efficient way to execute on the NPU. The original data volume is generally very large, so all computations cannot be completed in a single instruction call. Therefore, the data needs to be distributed to multiple cores for parallel computation. In addition, each core needs to consider how to loop over the computations for optimal performance. Different inputs may have different optimal execution approaches, so a tiling strategy is required to decide how to distribute data to the cores for computation. Based on hardware architecture characteristics, an AI Core is divided into two independent cores: AIC and AIV. The AIC and AIV cores each have their own independent scalar computation unit and can load and execute their own code segments independently.

### 3.1 <term>Atlas A2 training products</term>

The separated AIC/AIV architecture of the <term>Atlas A2 training products</term> chip allows the AIC and AIV to execute in parallel. Data is exchanged between the AIC and AIV through L2 and GM (Global Memory, the high-bandwidth memory). The number of interactions between the two greatly affects performance. In addition, due to the difference in computing power between the AIC and AIV, the two need to use different basic block sizes. Guided by the principles of minimizing the number of AIC-AIV communications and maximizing computing power, the CV tiling separation strategy was developed. This strategy effectively reduces the number of CV communications and, based on the buffer characteristics of different units, selects different basic blocks for computation, thereby improving operator performance.

 For the FAG operator, the Vector computation involves multiple inputs, outputs, intermediate computation results, and the double-buffer design. Therefore, the buffer needs to be split into multiple portions, and the largest portion in the optimal allocation scheme is 32KB. Because the Vector computation uses the float32 data type, the basic tiling block of the Vector is 8 *1024. To fully utilize the computing power of the Cube, the data volume of one round of computation is allocated at a ratio of 1:16 between C and V. In addition, because the input data type on the Cube side is float16 and the output is float32, the basic block of the Cube is 128* 128. Therefore, a data volume of 128 * 1024 is derived through the ratio nRatio=8. The pseudocode is as follows:

```c++
// C-Tiling: (S1_c_i,D)x(D,S2_c_i) => (S1_c_i, S2_c_i):(128,1024)
// V-Tiling: (S1_v_i, S2_v_i) => (8,1024)

// matmul computation on the C side
Bmm((S1_c_i,D)x(D,S2_c_i)) => 128*1024  // Output result 128*1024, placed on the workspace
// Vector computation on the V side
for S1_c_i/S1_v_i=128/8:
  copy_gm_to_ub(S1_v_i*S2_v_i)  // Copy in the bmm result data from the bmm workspace
  Vector(S1_v_i,S2_v_i)         // Perform the Vector computation
  copy_ub_to_gm(S1_v_i*S2_v_i)  // Vector computation ends, final output data is obtained, copy to GM

// Because the data volume computed on the Cube side is larger than that on the Vector side, Vector Tiling needs to be performed again inside the UB, which produces the S1-dimension ratio: S1_c_i/S1_v_i
```

In the preceding example, the ratio is enabled only in the S1 direction, and the C/V computation lengths are the same in the S2 direction. The ratio can also be enabled in both the S1 and S2 directions. The advantage is that the Cube can dispatch a large block of data at a time, avoiding the communication overhead caused by continuously dispatching small blocks, and the buffer of the Cube unit can be used to the maximum extent.

### 3.2 **<term>Ascend 950PR&950DT products</term>**

The Ascend 950PR&950DT products also adopt the separated AIC/AIV architecture and retain the AIC-AIV parallel execution feature. In addition, high-speed data exchange paths between the AIC and AIV, L0C->UB and UB->L1, are added. These paths reduce the cost of CV interactions and increase the pipelining parallelism. Compared with the complex tiling block strategy of the <term>Atlas A2 training products</term>, good performance can be achieved on the 950 with only one basic block.

 For the FAG operator, the Vector computation involves multiple inputs, outputs, intermediate computation results, and the double-buffer design. Therefore, the buffer needs to be split into multiple portions, and the largest portion in the optimal allocation scheme is 32KB. Because the Vector computation uses the float32 data type, the basic tiling block of the Vector is 64 *128. Because the number of Cube cores to Vector cores is 1:2, to fully utilize the computing power of the Cube cores, the Cube side uses a basic block of 128* 128. That is, after each Cube core computes a 128 * 128 data block, the block is evenly split between the two Vector cores for processing. The pseudocode is as follows:

```c++
// C-Tiling: (S1_i,D)x(D,S2_i) => (S1_i, S2_i):(128,128)
// V-Tiling: (S1_i / 2, S2_i) => (64,128)

// matmul computation on the C side
Bmm((S1_i,D)x(D,S2_i)) => 128*128  // value@dy,q@k output result 128*128, placed on L0C
fixp_l0c_to_ub(S1_i / 2,S2_i); => v0: 64*128, v1: 64*128 // Through the CV path L0C->UB, evenly written to the UBs of the two V cores
// Vector computation on the V side
Vector(S1_i / 2,S2_i)       // Perform the Vector computation
copy_ub_to_l1(S1_i/2*S2_i)  // Vector computation ends, output data is obtained, and through the CV path UB->L1, data from the two V cores is aggregated and copied to L1
Bmm((S1_i,S2_i)*(S2_i,D)); // dq, final result output to the workspace
Bmm((S2_i,S1_i)*(S1_i,D));  // dkv, final result output to the workspace
```

In the preceding example, the two copy instructions fixp_l0c_to_ub and copy_ub_to_l1 copy the data computed on the Cube side to the Vector side and the data computed on the Vector side to the Cube side, respectively. Both copies use intra-core high-speed channels.

## 4 Pipeline Design

To pursue ultimate performance, hardware resources must be fully utilized, which usually requires pipeline design for different pipelines. The purpose of pipeline design is to make one pipeline reach a bound state, keeping a hardware unit constantly working to reach the performance limit.

### 4.1 V-Side Pipeline

The V-side pipeline design needs to consider the data movement and computation processes of the Vector. The main optimization method is double buffer.
Taking the following pipeline task diagram as an example, the Vec function is split into two pipeline tasks: subA and subB, each focusing on a single function. The data to be processed is split into two slices, and ping-pong denotes the two data processing tasks. Each task needs to sequentially perform the DataCopy movement and Clc computation operations (Clc denotes the Vector computation). Arrows between tasks indicate data dependencies. For example, subB can process Clc only after subA completes DataCopy.
As shown in the figure, without pipeline design, the movement and computation tasks are executed serially, causing a stream interruption. That is, the movement pipeline stays idle after the first DataCopy completes, and it continues to work and starts the second DataCopy only after the first batch of moved-in data is computed and moved out (the Vector computation and move-out pipelines have the same problem). In this case, the performance is usually extremely poor.

![Design diagram 1](./fig/fag_design_diagram_1.png)

After double buffer is applied between the ping-pong pipelines of the pipeline tasks in the preceding figure, the pipeline tasks run as shown in the following diagram. As shown in the run diagram, for the same data slice, the DataCopy movement and Clc computation have a dependency and need to be processed serially. For different data slices, multiple tasks can be processed in parallel at the same time point, thereby achieving task parallelism and improving performance.

![Design diagram 2](./fig/fag_design_diagram_2.png)

The memory resources occupied by the two computation data blocks, ping and pong, are independent of each other.
The V-side computation of the FAG fused operator involves many steps and complex situations. A simple double buffer usually cannot cover all cases, so different computation pipeline arrangements exist. Different computation pipelines are applicable to different types of shape features, so as to achieve the best pipeline design for each type of shape feature.

### 4.2 CV Pipeline

Fused operators usually include both Vector computation and Cube computation. For the FA operator, the V-side computation depends on the C-side computation results. If only the V-side pipeline is considered and the C side is ignored, the C side and V side are likely to run as serial pipelines. This cannot achieve parallel computation and cannot make the fused operator perform at its best. Therefore, the CV pipeline design is introduced. In addition, the CV pipeline behaves differently for different operators. The Cube dual-dispatch mechanism of FA/FAG, also known as the inter-CV preload pipeline, enables pipeline optimization in two scenarios:

- Total C-side time > Total V-side time

  In this scenario, the Vector computation has few nodes and runs fast. On the <term>Atlas A2 training products</term> with C:V=1:2, the Cube movement duration is long enough to cover the Vector computation duration, so only the MTE2 time of the Cube needs attention, achieving MTE2 bound. Under the Cube dual-dispatch mechanism, two Cube computations are dispatched in advance, so the Cube1 and Cube2 computations can be chained, maximizing Cube utilization and achieving Cube bound.

![Design diagram 3](./fig/fag_design_diagram_3.png)

- Total C-side time < Total V-side time

  In this scenario, the Vector computation has many nodes and is the bottleneck. The C-side movement is not enough to cover the V-side pipeline, so CV pipeline arrangement is needed to achieve CV parallelism as much as possible. The most common optimization method is early dispatch on the C side.

![Design diagram 4](./fig/fag_design_diagram_4.png)

The C side dispatches two Cube computations consecutively, so when the V side finishes computing the previous round, it can immediately start the computation of the current round without waiting for the data of Cube1. This keeps the V side constantly working and achieves Vector bound.

## 5 Multi-Template Design

Templates are used to implement fused operators so that different inputs can reuse the same tiling and pipeline. However, using the same template for all different inputs cannot achieve optimal performance and functional generalization. Therefore, different templates need to be distinguished based on the characteristics of the input shape.
The multi-template design of the FAG (FlashAttentionScoreGrad, abbreviated as FAG) fused operator mainly includes:

- **Splitting templates based on intra-core and inter-core partitioning**

  Because the hardware buffer size is limited while the data volume to be computed is huge, the computation cannot be completed at one time, so tiling partitioning is required. Different shapes lead to different partitioning axes of the operator, and the partitioning axes affect the function and performance of the template. Simple elewise operators usually fuse all axes into one axis for partitioning, which has simple logic and therefore relatively simple templates. In contrast, fused operators combine elewise, broadcast, reduce, matmul, and other scenarios with complex functions. To meet high performance requirements, templates often need to be split based on partitioning axes. To achieve optimal performance when splitting templates, consider the following points:

  - a. Use all cores to prevent some cores from being idle

  - b. Distribute the computation volume relatively evenly across cores, preventing some cores from computing too much data while other cores stand by.

  - c. Match the data volume processed between the AIC and AIV with their respective computing power, preventing long idle periods of the AIC or AIV.

  The FAG operator contains five axes in total: B, N2 (the N of key and value), G (query_N/kv_N), S1 (the S of query), and S2 (the S of key and value). The partitioning order is intra-core first and then inter-core. For intra-core partitioning, the partitioning axes are selected based on the basic block size. For inter-core partitioning, the remaining axes after intra-core partitioning are merged and then partitioned based on the number of AI Cores. Because shapes differ in size, the partitioning axes change. From the Vector perspective, the FAG operator is divided into the following template types. Templates are prioritized by sequence number: the smaller the sequence number, the higher the priority and the earlier the match.

- **Depending on the applicable scope, the FAG operator is divided into the following template types:**

**<term>Atlas A2 training products</term>**

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
    <th>Partitioning Axis</th>
    <th>Entry Condition</th>
    <th>Applicable Scope</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>B template</td>
    <td>Multi-core partitioning on B</td>
    <td>Non-FP32, bestBasicBlockNum = S1 >= 4 ? 64 * 128 / 4 * 3 : 64 * 128;
N1 * G * alignedS1 * alignedS2 <= bestBasicBlockNum. </td>
    <td>Normal scenario</td>
  </tr>
  <tr>
    <td>N2 template</td>
    <td>Multi-core partitioning on N2</td>
    <td>Non-FP32, S1 * S2 < 64 * 128. </td>
    <td>Normal scenario</td>
  </tr>
  <tr>
    <td>SameAB template</td>
    <td>UB partitioning on S1S2, multi-core partitioning on S2</td>
    <td>In the deterministic computation scenario, if non-FP32 or S1 >= 1024 or S2 >= 1024, the SameAB template is used; otherwise, template 1.2 is used.<br>
        In the TND scenario, if average S1 >= 1024 and average S2 >= 1024, the SameAB template is used; otherwise, template 1.1 is used.<br>
        In the normal scenario, S1 >= 1024 or S2 >= 1024.
    </td>
    <td>Deterministic computation scenario<br>TND scenario<br>Normal scenario</td>
  </tr>
  <tr>
    <td>S1S2 template</td>
    <td>UB partitioning on S1S2, multi-core partitioning on N2</td>
    <td>In the deterministic computation scenario, the SameAB template conditions are not met.<br>
        In the normal scenario, the conditions of other templates are not met.
    </td>
    <td>Deterministic computation scenario<br>Normal scenario</td>
  </tr>
  <tr>
    <td>TND template</td>
    <td>UB partitioning on S1S2, multi-core partitioning on S2</td>
    <td>The SameAB template conditions are not met.</td>
    <td>TND scenario</td>
  </tr>
</tbody>
</table>

**Ascend 950PR&950DT products**

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
    <th>Partitioning Axis</th>
    <th>Entry Condition</th>
    <th>Applicable Scope</th>
  </tr>
</thead>
<tbody>
 <tr>
    <td>BN2</td>
    <td>Multi-core partitioning on BN2</td>
    <td>S1<=128 and S2 <= 128 and G=1, non-FP32.</td>
    <td>Normal scenario</td>
  </tr>
  <tr>
    <td>BN2S2 template</td>
    <td>Multi-core partitioning on BN2S2</td>
    <td>Non-FP32, TND, B * N2 * S2 > cubeCoreNum and  G = 1. </td>
    <td>Normal scenario</td>
  </tr>
  <tr>
    <td>Deterministic computation template</td>
    <td>Multi-core partitioning on BN2GS1S2</td>
    <td>Deterministic computation is enabled.</td>
    <td>Deterministic computation scenario</td>
  </tr>
  <tr>
    <td>BN2GS1S2 template</td>
    <td>Multi-core partitioning on BN2GS1S2</td>
    <td>None of the preceding scenarios is met.</td>
    <td>Normal scenario</td>
  </tr>
</tbody>
</table>

Each template type has its own UB and Block partitioning axes and can process scenarios with inputs of specific shape features. The template is designed for that type of shape feature.

- **Specializing templates based on special scenarios and specific optimizations**

  Each fused operator has a base template supplemented by multiple specialized templates:

    - The base template covers the functionality and the performance for most inputs of this type of shape feature.
    - The specialized templates deliver ultimate performance for specific scenarios. They mainly apply optimizations based on special methods that suit specific scenarios and are not suitable for generalization.

  For example, the empty_input template is specialized for the special scenario of empty tensors, and the deterministic computation template is designed based on the role-based Cube core management (RCM) optimization solution.

- **Specializing templates based on different computation pipelines**

  To fully exploit hardware advantages, fused operators usually require pipeline design to improve performance. Different pipeline designs have a great impact on the code architecture. To improve the maintainability, testability, and readability of the code, templates need to be specialized based on different computation pipelines to achieve ultimate performance in specific scenarios.

### 5.1 Detailed Multi-Template Design

- **Basic concepts:**

  **CV basic block:** Represents the data volume computed by the Cube or Vector in a single computation. It describes the data volume of one complete Cube-Vector interaction and is usually equivalent to the **inter-core basic block**. On the chip, communication between the Cube and Vector has a certain overhead, so the CV basic block is set relatively large. A suitable data volume is generally 512*1024 (unit: Bytes). In addition, because the intra-core buffers of the Cube and Vector are limited, the inter-core basic block may need to be completed through **multiple intra-core computations**.

  **Intra-core basic block:**

  If the CV basic block is too large, the Cube and Vector further split the CV basic block inside the cores into basic blocks suitable for the sizes of intra-core L0A, L0B, L0C, UB, and so on. This is called the **intra-core basic block**. On the Cube side, the intra-core basic block is generally 32KB (unit: Bytes), which fully unfolds the double buffer capability of L0A and L0B and makes full use of the computing power and bandwidth. On the Vector side, the basic block size is generally 32KB (unit: Bytes).

  **xxx.i:** Represents the size of an axis in the CV basic block after partitioning. A basic block is generally two-dimensional, and xxx.i represents the size of one dimension. xxx can be B, N2, G, S1, or S2.
  For example, S1 = 512, S2 = 1024, S1.i = 64, S2.i = 128 means that [S1, S2] is partitioned into basic blocks of size [64, 128]; the basic block size of the S1 axis is 64, and the basic block size of the S2 axis is 128.

  **xxx.o**: Represents the number of parts of an axis after basic block partitioning. xxx can be B, N2, G, S1, or S2;

  For example, S1 = 512, S2 = 1024, S1.i = 64, S2.i = 128 means that [S1, S2] is partitioned into basic blocks of size [64, 128]; S1.o = 512 / 64 = 8, S2.o = 1024 / 128 = 8, and a total of 8 * 8 basic blocks are partitioned.

  - **<term>Atlas A2 training products</term>**
  The templates of the FAG operator are divided as follows. For the following templates, the larger the sequence number, the higher the priority of the template. Template 1 is the generalized template (supports all shapes):

    > 1. Template with inter-core partitioning on the B, N2, G, and S1 axes and intra-core partitioning on the S1 and S2 axes:
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp
    >
    >    tiling code class: FlashAttentionScoreGradTilingS1s2Bn2gs1s2
    >
    >    kernel code: ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2.h
    >
    >    Condition: supports all shapes. If other templates do not support a shape, this template is used.
    >    Rationale: Following the most general approach, this template supports all shapes. However, because the intra-core computation can process only S1 * S2 at a time, frequent CV interaction overhead occurs and the performance is poor when both S1 and S2 are small. When both S1 and S2 are small, the flow routes to the following templates.
    >
    >
    >
    > 2. Template with inter-core partitioning on the B and N2 axes and intra-core partitioning on the G, S1, and S2 axes. This template optimizes performance for scenarios where both S1 and S2 are small and (B * N2 is relatively large or G = 1) through pure core partitioning changes.
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2.cpp
    >
    >    tiling code class: FlashAttentionScoreGradTilingS1s2Bn2
    >
    >    kernel code: ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2.h
    >
    >    Condition: (S1 < 1024) && (S2 < 1024)  && (B *N2* 2 > CoreNum || G == 1)
    >
    >    Rationale: When both S1 and S2 are small and both B and N2 are large, B and N2 are used for core partitioning. N2.i is not partitioned intra-core, and the Cube and Vector computations are performed N2.i times in a loop.
    >
    >
    >
    > 3. Template with inter-core partitioning on the B and N2.o axes and intra-core partitioning on the N2.i, G, S1, and S2 axes. This template optimizes performance for scenarios where G *S1* S2 are all small by partitioning the N2 axis intra-core and also partitioning the N2.i axis intra-core to accelerate the Vector computation. Compared with template 2, template 3 is more complex. Template 3 also partitions the N2.i axis in intra-core computations, making the computation volume of each round larger.
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_ngs1s2_bn.cpp
    >
    >    tiling code class: FlashAttentionScoreGradUngs1s2BbnTiling
    >
    >    kernel code: FlashAttentionScoreGradUngs1s2Bbn
    >
    >    Condition: G *S1* S2 <= 32KB && G = 1 && S2 < 1536
    >
    >    Rationale: When G *S1* S2 is smaller than 32KB, part of the N2 axis can be partitioned intra-core to enlarge the CV basic block. Meanwhile, within the Vector core, N2.i is also partitioned so that the computation volume of each Vector round is larger, improving Vector utilization.
    >
    >
    >
    > 4. Template with inter-core partitioning on the B axis and intra-core computation on the B.i, N2, G, S1, and S2 axes. This template optimizes performance when N2 *G* S1 * S2 is relatively small.
    >
    >    tiling code file: ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_bngs1s2_b.cpp
    >
    >    tiling code class: FlashAttentionScoreGradUbngs1s2BbTiling
    >
    >    kernel code: FlashAttentionScoreGradUngs1s2Bbn
    >
    >    Condition: N2 *G* S1 *S2 <= 64* 128
    >
    >    Rationale: If B.i alone is to be placed into the CV basic block, the inner axes N2 *G* S1 * S2 need to be small enough. Generally, when this value is smaller than 64KB, the data volume of Bmm1 and Bmm2 usually does not exceed half of L1, making partitioning on the B axis meaningful. Otherwise, a single Matmul fills L1, and the data move-in between multiple Matmuls cannot be parallelized with the computation.

  - **<term>Ascend 950PR&950DT products</term>**
  The templates of the FAG operator are divided as follows. For the following templates, the larger the sequence number, the higher the priority of the template. Template 1 is the generalized template (supports all shapes). Although multiple templates exist, only two template files exist in the implementation: one for deterministic computation and the other for non-deterministic computation. The non-deterministic computation template includes three partitioning templates: BN2, BN2S2, and BN2GS1S2. In the code, their implementation logic is isolated through template parameters:

    > 1. Template with inter-core partitioning on the B, N2, G, S1, and S2 axes:
    >
    >       tiling code file: ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling code class: FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel code: ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       Condition: supports all shapes. If other templates do not support a shape, this template is used.
    >       Rationale: Following the most general approach, this template supports all shapes. However, the performance is not optimal in some scenarios, in which case the flow routes to the following specialized templates based on the scenario.
    >
    > 2. Template with inter-core partitioning on the B and N2 axes:
    >
    >       tiling code file: ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling code class: FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel code: ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       Condition: S1<=128 and S2 <= 128 and G=1, non-FP32.
    >       Rationale: When S1<=128 and S2 <= 128 and G=1, the S1S2 axes are not partitioned across cores, meaning no multi-core accumulation exists. This eliminates the preceding GM zeroing and the following cast and musl computations, resulting in better performance.
    >
    > 3. Template with inter-core partitioning on the B, N2, and S2 axes:
    >
    >       tiling code file: ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling code class: FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel code: ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       Condition: Non-FP32, TND, B *N2* S2 > cubeCoreNum and  G = 1.
    >       Rationale: In the scenario of B *N2* S2 > cubeCoreNum and  G = 1, the dk and dv computations do not require multi-core accumulation. This eliminates the preceding GM zeroing for dk and dv and the following cast and musl computations, resulting in better performance.
    >
    > 4. Deterministic computation template
    >
    >       tiling code file: ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling code class: FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel code: ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel_deter.h
    >
    >       Condition: Deterministic computation is enabled.
    >       Rationale: The deterministic computation scenario has strict requirements on core partitioning. A specific core partitioning method avoids multi-core accumulation at the same address, achieving deterministic computation.

##

## 6 Programming Perspective

### 6.1 AscendC High-Level API

**<term>Atlas A2 training products</term>**

The current AscendC high-level API provides two programming modes. In the first mode, the Vector is the master core and the Cube is the slave core. Vector0 and Vector1 independently dispatch Matmul tasks, and the two are independent of each other.

In the second mode, the Cube is the master core and the Vector is the slave core. V0 uniformly dispatches the Matmul task, and the result of this task is processed jointly by V0 and V1, usually with each processing half. Currently, if a template name ends with "_sab", the template uses the Cube as the master core. For example:

```c++
ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_s1s2_bn2gs1_sab.h

ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
```

For FlashAttention, using the Cube as the master core allows the Matmul tasks of V0 and V1 to reuse the left matrix, and part of the output results can be accumulated in L0C. This reduces the bandwidth dependency, so the performance is better in most scenarios.

**<term>Ascend 950PR&950DT products</term>**

To achieve ultimate performance, all scenarios except some deterministic computation scenarios are implemented using the AscendC low-level API. Both the high-level and low-level implementations use the Cube as the master core.

### 6.2 AscendC Low-Level API

**<term>Atlas A2 training products</term>**

Some templates do not use the AscendC high-level API. For example:

```c++
ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h
```

This template more thoroughly uses the Cube as the master core and the Vector as the slave core. All Matmul tasks are fully dispatched from the Cube side, which notifies the Vector side through synchronization.

**<term>Ascend 950PR&950DT products</term>**

Except for some deterministic computation scenarios, all other scenarios are implemented using the low-level API, achieving ultimate memory reuse.
