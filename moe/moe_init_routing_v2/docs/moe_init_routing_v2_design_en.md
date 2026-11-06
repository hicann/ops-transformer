Declaration: This document uses the [Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Reproduction, citation, modification, and other operations must comply with this license.

# 1 MoeInitRoutingV2 Operator Design Introduction

Figure 1 Computation flowchart:

![MOE diagram](../../../docs/en/figures/MoeInitRoutingV2.png)

The main logic of the MoeInitRoutingV2 operator runs on the VectorCore, and its core computation process is shown in the preceding figure, mainly including the following steps:

1. Initialization: including tiling initialization (copying the tiling data from DDR memory to AiCore memory), custom object instantiation, initialization, and so on.
2. Sort expertIdx to obtain expandedExpertIdx and sortedRowIdx, where expandedExpertIdx corresponds to the sorted values and sortedRowIdx corresponds to the sorted indexes.
3. Perform the histogram operation on expandedExpertIdx to obtain expertTokenCountOrCumSum or expertTokenBeforeCapacity.
4. Sort sortedRowIdx, and use the sorted index as the output expandedRowIdx.
5. Perform the Scatter operation with x as the input and expandedRowIdx as the index to obtain the output expandedX.

# 2 Template Design

To enable different inputs to reuse the same tiling and pipeline, the fused operator is implemented in the form of templates. However, using the same set of templates for all different inputs cannot achieve optimal performance and functional generalization. Therefore, different templates need to be implemented based on the characteristics of the input shapes.

## 2.1 General Template

In the general scenario, the MoeInitRoutingV2 operator is divided into four steps according to the computation flow: Sort, ComputeTokenCount, SrcToDst, and Scatter. Since each step depends on the result of the previous step and needs to occupy the complete UB space, **multi-core synchronization** is required before the computation of each step to ensure that the previous step has completed computation.

### 2.1.1 Sort

This step is used to sort expertIdx. Based on the maximum number of sorts that a single core can process, the Sort phase is divided into the single-core template and the multi-core template:

a. Single-core template: in this case, the data volume does not exceed the single-core processing upper limit, and only core 0 is used to sort expertIdx while other cores are idle.

b. Multi-core template: used when the data volume exceeds the single-core processing upper limit. First, the data to be sorted is split based on the number of cores, then sorting is performed within each core, and finally one core is used to perform merge sorting on all single-core sorting results.

### 2.1.2 ComputeTokenCount

This step is used to count the number of tokens processed by each expert or the cumsum result.

### 2.1.3 SrcToDst

This step is used to convert sortedRowIdx into expandedRowIdx. Depending on dropPadMode, it is divided into the dropless template and the drop&pad template:

a. Dropless template: the conversion is performed directly.

b. Drop&pad template: during the conversion, the number of tokens processed by each expert is counted, the number of tokens processed by each expert is limited to expertCapacity, and redundant tokens are dropped. Finally, for the dropped rows, the corresponding expandedRowIdx is -1.

### 2.1.4 Scatter

Perform the Scatter operation with x as the input and expandedRowIdx as the index to output expandedX.

## 2.2 Performance Template

In scenarios where the input shape of the operator is small, the multi-core synchronization time between operations accounts for a high proportion and becomes the performance bottleneck. Therefore, a performance template is added for this specialized scenario.
The performance template requires the data volume to be small enough that each core can process the computation of the full expandedRowIdx. In this template, only x in the input is split across cores, that is, each core first independently performs the redundant expandedRowIdx computation, and then independently processes the corresponding scatter task based on its own computed expandedRowIdx. In this way, multi-core synchronization between steps is no longer needed, improving operator performance.

# 3 TilingKey Design

TilingKey is of the uint64 type. Generally, each template parameter corresponds to a decimal digit in TilingKey.

```C++
uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_HIGH_PERFORMANCE;                // 20000
  }
  if (dropPadMode == 0) {
    if (totalLength <= sortLoopMaxElement) {     // Sorting uses only one core
      return TILING_KEY_DROPLESS_SORT_ONE_CORE;        // 10001
    } else {
      return TILING_KEY_DROPLESS_SORT_MULTI_CORE;      // 10002
    }
  } else {
    if (totalLength <= sortLoopMaxElement) {
      return TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;   // 10011
    } else {
      return TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE; // 10012
    }
  }
}
```

Description of the fields in TilingKey:

| Decimal digit | Variable | Description |
| --- | ---- | ---- |
| 0 | Function/performance | Indicates the function template or performance template. 1 indicates the function template, and 2 indicates the performance template. |
| 1~2 | 0 | Reserved bits |
| 3 | dropPadMode | 0 indicates the dropless mode, and 1 indicates the drop&pad mode |
| 4 | sort | 1 indicates that expertIdx in the first step is sorted by a single core, and 2 indicates multi-core sorting |
