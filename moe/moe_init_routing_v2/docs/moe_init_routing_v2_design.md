声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# 1 MoeInitRoutingV2算子设计介绍

图1 计算流程图：

![MOE图](../../../docs/zh/figures/MoeInitRoutingV2.png)

MoeInitRoutingV2算子的主体逻辑运行在VectorCore上，其核心计算过程如上图所示，主要包括以下步骤：

1. 初始化：包括tiling初始化（将tiling数据从DDR内存拷贝到AiCore内存）、自定义对象实例化、初始化等；
2. 对expertIdx进行排序，得到expandedExpertIdx和sortedRowIdx，其中expandedExpertIdx对应排序后的value，sortedRowIdx对应排序后的index；
3. 对expandedExpertIdx做直方图运算，得到expertTokenCountOrCumSum或expertTokenBeforeCapacity；
4. 对sortedRowIdx进行排序，并以排序后的index作为输出expandedRowIdx；
5. 以x为输入，expandedRowIdx为index做Scatter运算，得到输出expandedX。

# 2 模板设计

为了使不同的输入可以复用相同的tiling和流水，采用了模板的方式来实现融合算子，但是不同的输入全部使用同一套模板时又无法达到性能最优和功能泛化，因此需要根据输入shape的特征区分不同的模板来实现。

## 2.1 通用模板

在通用场景下，MoeInitRoutingV2算子会按照计算流程，划分为Sort、ComputeTokenCount、SrcToDst、Scatter四个步骤。由于每一步都依赖于上一步的结果，并且需要占用完整的UB空间，因此每个步骤计算前都需要进行**多核同步**以保证上一步骤已计算结束。

### 2.1.1 Sort

这一步骤用于对expertIdx进行排序。根据单核可以处理的最大的排序的数量，Sort阶段被划分为单核模板和多核模板：

a. 单核模板：此时数据量未超过单核处理上限，仅使用第0核对expertIdx排序，其他核不工作。

b. 多核模板：在数据量超过单核处理上限时使用，首先根据核数切分待排序的数，然后在每个核内进行排序，最后再用一个核对所有的单核排序结果做归并排序。

### 2.1.2 ComputeTokenCount

这一步骤用于统计每个专家处理的token的数量或cumsum的结果。

### 2.1.3 SrcToDst

这一步骤用于将sortedRowIdx转化为expandedRowIdx。根据dropPadMode的不同，分为dropless模板和drop&Pad模板：

a. dropless模板：直接进行转换。

b. drop&pad模板：在转换的同时，统计每个专家处理的token数量，并限制每个专家处理的token数量在expertCapacity之内，drop掉多余的Token。最终，被drop的行，其expandedRowIdx对应为-1。

### 2.1.4 Scatter

以x为输入，expandedRowIdx为index，做Scatter操作，输出得到expandedX。

## 2.2 性能模板

在算子输入shape较小的场景，操作间的多核同步时间占比较高，成为性能瓶颈。因此，针对这种特化场景，添加性能模板。
性能模板要求数据量小到每个核能够处理全量expandedRowIdx的计算。在本模板中，只对输入中的x进行分核处理，即每个核先进行独立的冗余expandedRowIdx计算，再根据各自计算出的expandedRowIdx独立处理对应的scatter任务，这样就不再需要各步骤之间的多核同步，提升算子性能。

## 3 TilingKey设计

TilingKey为uint64类型，通常每个模板参数对应TilingKey中的一个十进制位。

```C++
uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_HIGH_PERFORMANCE;                // 20000
  }
  if (dropPadMode == 0) {
    if (totalLength <= sortLoopMaxElement) {     // 排序只用到一个核排序
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

TilingKey中字段说明：

| 十进制位 | 变量 | 说明 |
| --- | ---- | ---- |
| 0 | 功能/性能 | 表示功能模板或性能模板，1表示功能模板，2表示性能模板 |
| 1~2 | 0 | 预留位 |
| 3 | dropPadMode | 0表示dropless模式，1表示drop&pad模式 |
| 4 | sort | 1表示第一步expertIdx使用单核排序，2表示多核排序 |
