# RotaryMul2RotaryPositionEmbeddingFusionPass

## 融合模式

将图中的RotaryMul算子替换为RotaryPositionEmbedding算子（mode=0），如下图所示。

![](../../../docs/zh/figures/RotaryMul2RotaryPositionEmbeddingFusionPass_1.png)

输入映射关系：RotaryMul的x/r1/r2分别映射为RotaryPositionEmbedding的x/cos/sin，可选输入rotate不接入。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->

## 使用约束

- 仅匹配输入数为3的RotaryMul算子（x、r1、r2均为必选输入）。
- 融合后替换子图会基于原输入的shape/dtype/format重新推导输出shape，推导失败时放弃融合。
