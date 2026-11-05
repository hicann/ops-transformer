Declaration: This document uses the [Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Reproduction, citation, modification, and other operations must follow this license.

# 1 FFN Fused Operator Design Introduction

<img src="../../../docs/en/figures/ffn_compute_flow.png" alt="FFN computation flow" style="zoom:75%;" />

The core computation process of the FFN operator is shown in the preceding figure. The computation mainly consists of matmul1 -> activation function -> matmul2, and the main logic (flow control) runs on the VectorCore.

1. The initialization process includes tiling initialization (tiling is copied from the DDR memory into the AiCore), instantiation and initialization of the matmul objects and custom objects, and so on;
2. Determine how matmul1 and matmul2 are split across cores to fully utilize the multi-core parallel capability of the AiCore;
3. Execute matmul1 and matmul2 IterateAll on the vector to initiate the corresponding matmul computations on the cube;
4. Inter-core synchronization. When matmul1 -> activation function is computed across cores, each core outputs only a small portion. Inter-core synchronization waits until all cores complete the activation function computation, so that matmul2 starts computing with complete data;
5. The workspace (DDR memory) stores the data to be transferred between the vector core and the cube core. For example, the result computed by the activation function on the vector core is saved in the workspace, and the matmul2 computation reads data from this workspace as input;
6. Loop through different experts to perform the preceding computation.

In addition to the preceding computation flow, the operator implementation must also consider the following:

1. Support different parameters and data types, such as with or without bias, different activation function types, and different scenarios such as non-quantization, quantization, and pseudo-quantization. Different scenarios have different computation flows and performance optimization methods. Therefore, the implementation is divided into different templates, each with its own template parameters;
2. The AiCore memory size on hardware is limited. Computing an operator generally requires splitting the data and arranging data movement and computation in a pipelined parallel manner. This process has a significant impact on the operator and is the main adjustment object during performance optimization. The tiling function on the host computes the parameters for this splitting and pipelining.

# 2 Scenario Division

## 2.1 Non-Quantization

Non-quantization means that x/weight1/weight2/y are all floating-point types, such as float16/bfloat16. Non-quantization scenarios share the same computation process but differ in some computation details:

1. In the float16 scenario with innerPrecise=1, this is the high-performance scenario. The computation output of matmul1 on the CubeCore and the input and output of the activation function on the vector are all of the float16 type. The computation flow is: matmul1(float16) -> activation function (float16) -> matmul2(float16);
2. In the float16 scenario with innerPrecise=0, this is the high-precision scenario. The computation output of matmul1 on the CubeCore and the input and output of the activation function on the vector are all of the float32 type. Then the Cast high-level API is called on the vector to convert the result to float16 and output it to the workspace (the input of matmul2 is float16). The computation flow is: matmul1(float32) -> activation function (float32) -> Cast(float16) -> matmul2(float16);
3. In the bfloat16 scenario, the computation output of matmul1 on the CubeCore and the input and output of the activation function on the vector are all of the float32 type. Then the Cast high-level API is called on the vector to convert the result to bfloat16 and output it to the workspace (the input of matmul2 is bfloat16). The computation flow is: matmul1(float32) -> activation function (float32) -> Cast(bfloat16) -> matmul2(bfloat16);

Among the three scenarios, the latter two share the same computation process. Therefore, non-quantization is divided into two templates: the high-performance template (1) and the high-precision template (2/3).

## 2.2 Quantization

Quantization scenarios share one template, and template parameters distinguish the computation flow of different scenarios. The computation flow is: matmul1(int32) -> dequantization (float16/float32) -> activation function (float16/float32) -> quantization (float16/bfloat16) -> matmul2(int32) -> dequantization (float16/bfloat16);

1. The data type of output y is float16, and the data type of dequant scale is uint64 or int64. In this case, matmul1/matmul2(int32) -> dequantization (float16) is completed on the CubeCore as fused on-the-fly computation;
2. The data type of output y is float16, and the data type of dequant scale is float32. The dequantization high-level API does not support the float32 type. Therefore, the dequant scale must first be converted to the uint64 type on the vector, and then the matmul computation is performed. Dequantization is likewise completed as fused on-the-fly computation;
3. The data type of output y is bfloat16, and the data type of dequant scale is bfloat16.

* dequant scale float32 to uint64 conversion

  The float32 data is placed in the low 32 bits of uint64 through gather. The process is as follows: <img src="../../../docs/en/figures/ffn_dequant_to_uint64_compute_process.png" alt="FFN dequant to uint64 computation process" style="zoom:67%;" />

  1. Generate the index required by gather, that is, srcOffset;
     * First use CreateVecIndex to generate the data sequence 0,1,2,...,2*n-1, where n is the number of dequant scale data elements, with the data type int32, and then Cast it to uint32;
     * Use ShiftRight to logically shift the sequence right by one bit, forming the sequence: 0,0,1,1,2,2,...,n-1,n-1;
     * Multiply the sequence by sizeof(float32)=4. The gather index uses bytes as the unit when representing address offsets;
  2. The mask is defined bit by bit (see the Ascend C development documentation) as mask={0x5555555555555555 & (2 ^ n -1), 0} (one vector cycle can process 8*32B, that is, 64 float32 values). The actual code does not use 2^n-1; instead, invalid data in the tail block is skipped during data movement.
  3. Use Duplicate to fill the destination address with 0;
  4. Use Gather to generate the expected uint64 data.

  In the preceding process, src (dequant scale) and dst are treated as uint32 (not Cast), and the result is finally output to the workspace, where the matmul is configured with the uint64 type.

* The vector computation process of the quantization scenario is summarized in the following figure:

  ![FFN quantization scenario vector computation process](../../../docs/en/figures/ffn_quant_scenario_vector_compute_process.png)

## 2.3 Pseudo-Quantization

Pseudo-quantization scenarios share one template, and template parameters distinguish the computation flow of different scenarios. The computation flow is: CastWeight1(float16/bfloat16) -> matmul1(float16/float32) -> activation function (float16/float32) -> Cast(float16) -> CastWeight2(float16/bfloat16) -> matmul2(float16);

CastWeight1/2 is completed on the vector by calling the high-level API. CastWeight2 has no dependency on other data, so it can run in parallel with matmul1. Similarly, matmul2 can run in parallel with the CastWeight1 of the next expert (the flow is described later).

# 3 Tiling Design

To clearly present the core idea of tiling computation, this section uses specific platform parameters and a typical input shape as an example:

* Use the following hardware information as an example: 20 AICores, L0A/L0B buffer size of 64kb, and L0C buffer size of 128kb;
* The data types of x and weight are float16, x[128, 5120], weight1[8, 5120, 2560], and weight2[8, 2560, 5120], with expert_token=[16, 16, 16, 16, 16, 16, 16, 16]. In this case, 8 experts are processed cyclically. Each cycle needs to compute matmul1: A[16, 5120] \* B[5120, 2560], the activation function: [16, 2560], and matmul2: A[16, 2560] \* B[2560, 5120].

The computation of the FFN operator mainly consists of matmul1 + activation function + matmul2. Therefore, the tiling parameters are also divided into these three parts. The matmul computation is completed by calling the high-level API, and the main body of the corresponding tiling parameters is also generated by the corresponding API. FFN adjusts some parameters based on the actual operator optimization results.

## 3.1 matmul tiling

matmul tiling consists of two parts: splitting across cores and splitting within a single core. These two parts are coupled. Generally, splitting within a single core is considered first, followed by splitting across cores. For higher performance, the general splitting principle of the AICore is to fill the L0 (L0A/L0B/L0C) buffers as much as possible to reduce the number of instructions and the instruction start/end overheads (special shapes are analyzed separately).

* Splitting within a single core

  The L0 buffer splitting is controlled by the parameters baseM/baseK/baseN in matmul tiling. Changing the tiling generally does not change the actual computation amount, but changes the number of data movements. One direction of performance optimization is to reduce the number of movements. From the perspective of filling the L0 buffer as much as possible and reducing the number of movements, the following relationships can be derived:

  1. The computation data movement ratio = $\frac{baseM*baseK*baseN}{baseM*baseK+baseN*baseK}=1/(\frac{1}{baseM}+\frac{1}{baseN})$; that is, increasing baseM/baseN improves the computation-to-movement efficiency ratio. With the computation amount unchanged, this reduces the amount of data moved to the L0 buffer. In essence, the data in the L0 buffer is reused as much as possible.
  2. L0C buffer constraint: $baseM*baseN*sizeof(float)<128kb$;
  3. L0A buffer constraint (double buffer enabled): $baseM*baseK*sizeof(half)<32kb$;
  4. L0A buffer constraint (double buffer enabled): $baseK*baseN*sizeof(half)<32kb$;
  5. baseM/basek/baseN should be aligned to 16 as much as possible;
  6. When moving data to the AiCore, contiguous data should be a multiple of 128 as much as possible, and the closer to 512 the better (depending on the product version);

  According to items 1/2/5/6, the optimal baseM/baseK/baseN combination is 128/64/256. Actual tests show that these tiling parameters achieve better performance for most shapes.

* Splitting across cores

  Splitting within a single core mainly considers computation and movement efficiency, while splitting across cores aims to utilize the AiCore computing power in parallel as much as possible. matmul splitting can be performed on three axes: m, k, and n. However, splitting on the k axis requires an additional accumulation, or uses the atomic accumulation capability of DDR (supports float; if the output is float16, a type conversion is required when moving data to the vector). This solution was tried in the early implementation of the FFN operator, but the performance did not meet expectations, so this direction was not explored further (this method may be further optimized). The current FFN implementation splits on the m/n axes or on the expert dimension. Given that the single-core baseM/baseK/baseN combination is 128/64/256:

  1. For the shape of matmul1: A[16, 5120] \* B[5120, 2560], n=2560, and the computation is split across 10 cores, with single-core singleN=256;
  2. For the shape of matmul2: A[16, 2560] \* B[2560, 5120], n=5120, and the computation is split across 20 cores, with single-core singleN=256;

  With the optimal single-core strategy, matmul1 can only be split across 10 cores, wasting half of the AiCore capability. matmul2 has a data dependency on matmul1, so the idle cores cannot be used to compute matmul2. There are two solutions:

  1. Use the matmul1 of the next expert for parallel computation, called the expert parallel splitting solution in this document;
  2. Readjust singleN of matmul1 to 128 or smaller to enable multi-core parallel computation.

  The current code combines these two solutions and selects one based on the actual shape information.

* Expert parallel splitting

  For the preceding shape, the expert parallel splitting strategy is as follows:

  <img src="../../../docs/en/figures/ffn_expert_parallel_diagram.png" alt="FFN expert parallel diagram" style="zoom:67%;" />

  1. When the matmul corresponding to an expert cannot fill all AiCore cores, the idle cores are used to compute the second group of experts. At this time, the intermediate results of two experts exist in the workspace at the same time.
  2. The n of matmul2 is large enough to fill all AICore cores, so it is computed serially (parallelism is not adopted because splitting on m and n more easily causes uneven computing power allocation and affects performance, as described in the mixed parallelism section below);

* Mixed parallelism

  A problem with expert parallelism is that when the m axes of the two parallel matmuls differ greatly (the n/k axes are always equal), half of the cores compute faster and can only idle at SyncAll, resulting in waste of computing power. An intuitive solution is to select experts with similar m values for splitting. Combining the shape information and the single-core splitting information baseM:

  1. Initialize an empty array tokenCache. The maximum length of the array is the maximum number of experts allowed to run in parallel. Here, 2 is used as an example;
  2. Traverse the tokens to obtain the m axis size of the matmul computation of the current expert;
  3. When m > baseM, the m axis can be split. Combined with n axis splitting, all AiCore cores can be filled. In this case, expert parallelism is not required, and the computation of this expert is completed directly;
  4. When m <= baseM, store the information of the current expert in the array tokenCache;
  5. When tokenCache has cached 2 experts, start expert parallel computation and clear tokenCache. Otherwise, return to step 2;
  6. When all tokens are traversed and tokenCache is not empty, traverse tokenCache, do not perform expert parallelism, and adjust singeN=128 for split computation;

  With this method, the m values for expert parallelism are all smaller than or equal to baseM=128, the difference is small, and the waste of computing power is small.

## 3.2 vector tiling

The FFN computation has no reduction, so the tiling can be more flexible. It is generally performed based on the matmul tiling. The main factors to consider:

1. An efficient computation process. However, the vector computation process of FFN is generally simple and deterministic, so there is not much optimization space for this factor;
2. The UB buffer size, mainly considering memory reuse and the amount of data computed each time.

The vector tiling parameters baseM and baseN (which are not the same variables as the matmul baseM and baseN; they are distinguished as cubeTiling.baseM and cubeTiling.baseN) determine the vector computation amount each time. The vector computation is generally a two-level loop.

```c++
for (int offsetM = 0; offsetM < cubeTiling.baseM; offsetM += baseM) {
    for (int offsetN = 0; offsetN < cubeTiling.baseN; offsetN += baseN) {
        int curBaseM = offsetM + baseM > cubeTiling.baseM ? cubeTiling.baseM - offsetM : baseM;
        int curBaseN = offsetN + baseN > cubeTiling.baseN ? cubeTiling.baseN - offsetN : baseN;
        VectorCompute(workspace, offsetM, curBaseM, offsetN, curBaseN);
    }
}
```

The preceding computation process needs to determine the vector computation baseM and baseN, and also determine the allocation and reuse of the UB buffer. The computation processes of non-quantization and pseudo-quantization are simple and do not reuse the UB buffer. Quantization has many scenarios and complex computations, so the UB buffer needs to be reused to increase the amount of data computed each time. The reuse is as follows (for the vector computation process of the quantization template, see the preceding text):

![FFN quantization scenario UB buffer allocation](../../../docs/en/figures/ffn_quant_scenario_ub_buffer_allocation.png)

Define the number of copies of a buffer as the allocated byte size of each copy divided by the number of data elements processed (baseM*baseN, recorded as ubCalcSize). The UB buffer allocation in different scenarios is summarized as follows:

| Scenario | Input | Output | double buffer | temp buffer | High-level API temp | baseM*baseN example | Total copies * ubCalcSize |
| -------------------------- | ---- | ---- | ------------- | ----------- | ------------ | --------------- | ------------------ |
| High precision | 4    | 2    | 1             | 4           | 8            | 32*256=8kb      | 18*8kb=144kb       |
| High performance | 2    | 2    | 1             | 0           | 4            | 64*256=16kb     | 8*16kb=128kb       |
| Quantization, float16 output | 2    | 1    | 1             | 2           | 4            | 64*256=16kb     | 9*16kb=144kb       |
| Quantization, bfloat16 output | 4    | 2    | 1             | 6           | 8            | 64*256=16kb     | 20*8kb=160kb      |
| Pseudo-quantization, float16 activation function | 2    | 2    | 2             | 0           | 4            | 32*256=8kb      | 13*8kb=104kb       |
| Pseudo-quantization, float16 cast weight | 1    | 2    | 2             | 0           | 0            | Same as above            | scale counts as one copy in the total copies of the preceding row |
| Pseudo-quantization, bfloat16 activation function | 4    | 2    | 2             | 4           | 6            | 32*256=8kb      | 23*8kb=184kb       |
| Pseudo-quantization, bfloat16 cast weight | 1    | 2    | 2             | 0           | 6            | Same as above            | scale counts as one copy in the total copies of the preceding row |

Note:

1. double buffer involves only the input and output, which is reflected in the memory initialization parameters of the input and output TQueues;
2. The temp buffer is the UB temporary space used in the FFN operator processing. For example, in activation function (float32)->Cast(bfloat16), the output of the activation function is in the UB temporary space;
3. High-level API temp refers to the temporary space reserved for the high-level API. In addition to the input and output, the computation process of the high-level API also occupies other UB buffers, generally reserving twice the input size;
4. To simplify the computation logic of different scenarios and keep performance stable, ubCalcSize is aligned to 8kb;
5. Taking a UB size of 192kb as an example, the last two columns give the maximum value of ubCalcSize and the total size of the UB buffer occupied;
6. In the pseudo-quantization scenario, the memory allocation additionally considers that scale occupies one copy.

# 4 Pipeline Parallelism

The AiCore has multiple parallel units, and the corresponding APIs include asynchronous and synchronous APIs to fully utilize the parallel capability of the hardware.

In actual FFN application scenarios, the MTE2 on the Cube (data movement from DDR to the CubeCore) is the main bottleneck in the computation flow. This process is optimized by tiling parameters (as described above). In addition, the time of the activation function on the vector accounts for less than 5% in total. Therefore, in the non-quantization and quantization scenarios, no CV (CubeCore and VectorCore) parallel optimization is specifically performed. The complex pipeline parallel design is mainly used in the pseudo-quantization scenario, where the weight must be dequantized before the matmul computation. This process runs on the VectorCore and is extremely time-consuming.

## 4.2 Pseudo-Quantization Pipeline Parallelism

Pseudo-quantization is mainly the parallel computation of CastWeight on the VectorCore and matmul on the CubeCore.

<img src="../../../docs/en/figures/ffn_pseudo_quant_scenario_pipeline_concurrency.png" alt="FFN pseudo-quantization scenario pipeline concurrency" style="zoom:67%;" />

1. In the preceding example flow, CastWeight1 runs in parallel with matmul2 (except the first time), and CastWeight2 runs in parallel with matmul1 (except the last time);
2. Three SyncAll operations. The SyncAll before matmul1 waits for all VectorCores to complete CastWeight1 (the dequantization process of weight). The SyncAll before matmul2 waits for all VectorCores to complete the activation function and CastWeight2 processes;
3. In the preceding example flow, matmul1 runs with two experts in parallel, and matmul2 runs serially. In the example, both CastWeight2 operations run in parallel with matmul1, instead of the second CastWeight2 running in parallel with the first matmul2. This is because the single-core computation amount of matmul1 is twice that of matmul2 (taking the preceding shape as an example, baseN is the same, and the k axis of matmul1 is twice the k axis of matmul2), so the computation time is long enough, and actual tests show that parallelism with matmul1 achieves better results.
