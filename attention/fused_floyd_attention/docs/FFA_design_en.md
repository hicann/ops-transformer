## 1 Computation Process

The implementation follows the forward computation flow of FusedFloydAttention. The overall computation flow is as follows:

1. a_0, computed from <query, key_1> through a specific matrix multiplication, is added to a_1, computed from <query, key_2> through a specific matrix multiplication, and the sum is then multiplied by the scaling factor scale_value. The result is then selected using atten_mask, which masks the positions that are true in atten_mask, producing masked_attention_score. That is, after the select operation, the positions that are true in atten_mask become an extremely small negative value, and become 0 after the softmax computation, thereby achieving the masking effect.

2. Drawing on the FlashAttention acceleration algorithm, FusedFloydAttention uses the FlashSoftmax operation to process masked_attention_score, replacing the softmax operation in the original formula, and then performs a matmul operation between the result and value. Because the FlashSoftmax operation splits masked_attention_score along the Skv direction (the sequence length of the input key and value), the implementation includes a refresh process, as follows:
    1. Each FlashSoftmax computation operates on only one split SkvSplit (SkvSplit is short for the sequence length after splitting along the Skv axis) and records exp starting from the second loop iteration, where i denotes the loop variable after the Skv split. For exp, i starts from 1. The formula for computing exp is as follows:
       $$
       exp[i] = e^{max_{i - 1} - max_{i}}
       $$
    2. When i = 0, the computed MM[PV] result is directly saved to the ub_attention_out[0] UB.
    3. Starting from i = 1, Mul and Add operations are added. That is, the previous MM[PV] result is multiplied by the current exp, and the product is added to the current MM[PV] result. The sum is saved to the ub_attention_out[1] UB. This process continues until the traversal of Skv is complete.
    4. Because the division by sum in the FlashSoftmax computation is moved to just before outputting attention_out, ub_attention_out in the UB must be divided by softmax_sum row by row at the end, and the final complete result is saved to the output memory attention_out(Final).

## 2 Tiling Design

Tiling aims to find a more efficient NPU execution method. The original data volume is usually very large, and all computations cannot be completed with a single instruction call. Therefore, the data volume needs to be distributed to multiple cores for parallel computation, and each core also needs to consider how to loop for optimal performance. Different inputs may have different optimal execution methods. Therefore, the tiling strategy determines how to allocate data to each core for computation.

 Based on the hardware architecture, the AI Core is divided into two independent cores: AIC and AIV. The AIC and AIV cores have their own independent Scalar computing units, can independently load their own code segments, and execute separately. The separated AIC and AIV architecture allows AIC and AIV to execute in parallel. The data exchange path between AIC and AIV is L2 and GM (Global Memory, a high-bandwidth memory). The number of interactions between them significantly affects performance. In addition, due to the difference in computing power between AIC and AIV, the two cores need to use different basic block sizes. Guided by the principles of minimizing the number of AIC-AIV communications and maximizing the computing power, the CV tiling separation strategy was developed. This strategy effectively reduces the number of CV communications and, based on the buffer characteristics of different units, selects different basic blocks for computation, thereby improving operator performance.

 For the FFA operator, the Vector computation involves multiple inputs, outputs, intermediate computation results, double-buffer design, and more. Therefore, the buffer needs to be divided into multiple parts, and the largest part in the optimal allocation scheme is 32KB. Because the data type used by the Vector computation is float32, the tiling basic block of the Vector is 8 *1024. To fully utilize the computing power of the Cube, the data volume of one round of computation between Cube and Vector is configured at a ratio of 4:1. In addition, because the input data type on the Cube side is float16 and the output is float32, the basic block of the Cube is 32* 32. Therefore, a data volume of 32 * 1024 is configured through nRatio=32. The pseudocode is as follows:

```c++
// C-Tiling: (S1_c_i,D)x(D,S2_c_i) => (S1_c_i, S2_c_i):(32,1024)
// V-Tiling: (S1_v_i, S2_v_i) => (8,1024)

// matmul computation on the Cube side
Bmm((S1_c_i,D)x(D,S2_c_i)) => 32*1024  // the output result is 32*1024 and is placed on the workspace
// Vector computation on the Vector side
for S1_c_i/S1_v_i=32/8:
 copy_gm_to_ub(S1_v_i*S2_v_i)  // copy the bmm result data from the bmm workspace
 Vector(S1_v_i,S2_v_i)         // perform the Vector computation
 copy_ub_to_gm(S1_v_i*S2_v_i)  // the Vector computation is complete, the final output data is obtained and copied to the GM

// Because the data computed on the Cube side is larger than that on the Vector side, Vector Tiling needs to be performed again inside the UB, which produces the ratio in the S1 direction: S1_c_i/S1_v_i
```

In the preceding example, the ratio is enabled only in the S1 direction, and the lengths of the C/V computations in the S2 direction are the same. The ratio can also be enabled in both the S1 and S2 directions. The benefit is that the Cube can launch a large block of data at a time, avoiding the communication overhead caused by continuously launching small data blocks, and can also make maximum use of the buffer of the Cube unit.

## 3 Pipeline Design

To achieve the ultimate performance, hardware resources must be fully utilized, and pipeline designs for different pipelines are usually required. The goal of pipeline design is to make one pipeline reach the bound state, so that a hardware unit keeps working and the performance limit is reached.

### 3.1  Vector-Side Pipeline

The Vector-side pipeline design needs to consider the data movement and computation processes inside the Vector, and the main optimization measure is double buffer. The functions of Vec are split into two pipeline tasks: Stage1 and Stage2, and each task focuses on a single function. The data to be processed is split into two slices, and ping-pong denotes the two data processing tasks. Each task sequentially performs data movement DataCopy and computation Clc operations. Arrows between tasks represent data dependencies. For example, Stage2 can process Clc only after Stage1 finishes DataCopy. Without pipeline design, data movement and computation tasks are executed serially, resulting in flow interruption. That is, the movement pipeline remains idle after the first DataCopy is complete, and the movement pipeline continues to work and performs the second DataCopy only after the first batch of moved-in data is computed and moved out (the Vector computation and move-out pipelines have the same problem). In this case, performance is usually extremely poor.

After double buffer is applied between the ping-pong pipelines, for the same data slice, DataCopy and Clc have a dependency and need to be processed serially. For different data slices, multiple tasks can be processed in parallel at the same time point, thereby achieving task parallelism and improving performance.

The memory resources occupied by the two pieces of computation data, ping and pong, are independent of each other. The Vector-side computation of the fused operator has many steps and is complex. A simple double buffer usually cannot cover all cases, so different computation pipeline arrangements occur.

### 3.2 CV Pipeline

For a fused operator, the computation on the Vector side depends on the computation result on the Cube side. If only the Vector-side pipeline is considered and the Cube side is ignored, the Cube side and Vector side are likely to form a serial pipeline, which fails to achieve parallel computation and prevents the fused operator from reaching optimal performance. This motivates the CV pipeline design. In addition, the CV pipeline behaves differently for different operators. The Cube dual-launch mechanism can implement pipeline optimization in two scenarios:

- Total time consumed on the Cube side > total time consumed on the Vector side

  In the pipeline characteristics of this scenario, the Vector has few computation nodes and fast computation. On Atlas A2 training products with C:V=1:2, the Cube movement duration is sufficient to cover the Vector computation duration. Therefore, only the MTE2 time consumed by the Cube needs attention, finally achieving MTE2 bound. Under the Cube dual-launch mechanism, two Cube computations are launched in advance, and the computations of Cube1 and Cube2 can be connected, maximizing Cube utilization and achieving Cube bound.

- Total time consumed on the Cube side < total time consumed on the Vector side

  The Cube side launches two Cube computations consecutively. This ensures that when the Vector side finishes computing the previous round, it can immediately start computing the current round without waiting for the data of Cube1. In this way, the Vector side keeps working, achieving Vector bound.

## 4 Optimization Results

- Single-operator inference time

| Input shape (B, H, N, M, K, D) | Chunk (s) | AscendC (s) | Speedup |
| :--- | :--- | :--- | :--- |
| (1, 32, 256, 256, 256, 32)    | 0.0282     | **0.0209** | **1.35x** |
| (1, 32, 384, 384, 384, 32)    | 0.0955     | **0.0604** | **1.58x** |
| (1, 32, 512, 512, 512, 32)    | 0.1941     | **0.1383** | **1.40x** |
| (1, 32, 768, 768, 768, 32)    | 0.7640     | **0.4563** | **1.67x** |
| (1, 16, 256, 256, 256, 64)    | 0.0158     | **0.0142** | **1.11x** |
| (1, 16, 384, 384, 384, 64)    | 0.0561     | **0.0316** | **1.78x** |
| (1, 16, 512, 512, 512, 64)    | 0.1205     | **0.1151** | **1.05x** |
| (1, 16, 768, 768, 768, 64)    | 0.4249     | **0.3302** | **1.29x** |

- Memory usage

| Input shape (B, H, N, M, K, D) | Chunk (GB) | AscendC (GB) | Memory Saving Rate (Mem Saving) |
| :--- | :--- | :--- | :--- |
| (1, 32, 256, 256, 256, 32)    | 6.38        | 2.65          | **58.46%** |
| (1, 32, 384, 384, 384, 32)    | 19.97       | 5.51          | **72.41%** |
| (1, 32, 512, 512, 512, 32)    | 45.50       | 9.50          | **79.12%** |
| (1, 16, 256, 256, 256, 64)    | 3.88        | 2.40          | **38.14%** |
| (1, 16, 384, 384, 384, 64)    | 11.53       | 5.18          | **55.07%** |
| (1, 16, 512, 512, 512, 64)    | 25.50       | 9.13          | **64.20%** |

## 5 Appendix — IterateBatch Configuration in the FusedFloydAttention Operator

This design involves two types of batched matrix multiplication (data format ND)

The first type is NMD @ NKD, where N is the batch axis and D is the K axis;

The second type is NMD @ KMD, where M is the batch axis and D is the K axis

In the first scheme, the batch axis is the outermost axis, and M and K may have slices. In this way, the ND matrix of each single batch is continuous, but gaps may exist between the ND matrices of different batches.

For example, when M=256, assume that the slices are N[0:16]M[128:256]D[:]. The data of each batch is 128*D, and the start addresses of batch data are spaced 256* D apart;

For this case, we use the NORMAL data layout of IterateBatch. For the address gaps between batch data, parameters such as matrixStrideA and matrixStrideB need to be configured. These two parameters can specify the gaps between the ND matrices of adjacent batches of input matrices A and B. In the preceding example, matrixStrideA is set to 256*D.

In the second scheme, the batch axis is in the middle. In this case, the ND matrix of each single batch is discontinuous and jumpy. One method is to control data movement in the implementation using low-level APIs. Alternatively, use the high-level API IterateBatch in AscendC.

When using IterateBatch, for the case where the batch axis is in the middle, the **BSNGD** data layout format can be configured. Each batch computes one piece of SD data. For example, for NMD @ KMD, the B matrix KMD can be understood as the [B=1,S=K,N=1,G=M, D=D] layout, so that the high-level API IterateBatch can be used.

When using IterateBatch, the batch data for each call needs to be specified, but it cannot be specified arbitrarily due to hardware constraints. The main requirement is that the specified computation batch data can be completely moved into the L1 cache (512KB for 910B), that is, the computation data volume cannot exceed the NPU L1 Cache. The related computation process is as follows.

- **Computing the data volume (Bytes) of one batch of matrix multiplication**

  oneBatchBytes = datatype Bytes *D* (K1 + K2)

  K1 and K2 take the slice lengths of the axes of the actually computed matrices. For example, M[128:256] in the preceding example corresponds to 128

- **The actual maximum number of batches is as follows**

  maxBatchNums = Floor(L1Size / oneBatchBytes)  # round down

- **The corresponding number of loops is as follows**

  Loops = Ceil(TotalBatches, maxBatchNums)    # round up
