# QuantGroupedMatmulInplaceAddTransposeFusionPass

## 融合模式

该融合 Pass 提供融合 Transpose、Reshape 节点的两种模式，参考下图。图中的 Transpose 也可为 TransposeD。

融合模式一：TT/TC量化场景下，消除 x1 前的转置类节点；若有 Bitcast 则保留，否则将源节点直接连接到算子输入。

![TT/TC量化的 x1 转置融合](../../../docs/zh/figures/QuantGroupedMatmulInplaceAddTransposeFusionPass_1.png)

融合模式二：MX 量化场景下，配套消除 x1 和 scale1 前的转置类节点；各侧若有 Bitcast 则保留，否则将源节点直接连接到对应算子输入。

![MX 量化的 x1 与 scale1 配套转置融合](../../../docs/zh/figures/QuantGroupedMatmulInplaceAddTransposeFusionPass_2.png)

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
