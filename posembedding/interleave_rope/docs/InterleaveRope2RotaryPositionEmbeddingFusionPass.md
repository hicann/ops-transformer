# InterleaveRope2RotaryPositionEmbeddingFusionPass

## 融合模式

将图中的InterleaveRope算子替换为RotaryPositionEmbedding算子（mode=3），如下图所示。

![](../../../docs/zh/figures/InterleaveRope2RotaryPositionEmbeddingFusionPass_1.png)

InterleaveRope与RotaryPositionEmbedding的输入一一对应（x/cos/sin），可选输入rotate不接入。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->

## 使用约束

- 仅匹配输入数为3的InterleaveRope算子（x、cos、sin均为必选输入）。
- 融合后替换子图会基于原输入的shape/dtype/format重新推导输出shape，推导失败时放弃融合。
