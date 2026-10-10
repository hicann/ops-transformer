# RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass

## 融合模式

将图中的RotaryMulGrad算子替换为RotaryPositionEmbeddingGrad算子（mode=0），如下图所示。

![](../../../docs/zh/figures/RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass_1.png)

输入映射关系：RotaryMulGrad的dy/r1/r2/x分别映射为RotaryPositionEmbeddingGrad的dy/cos/sin/x；输出dx/dr1/dr2分别对应dx/dcos/dsin。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->

## 使用约束

- 仅匹配输入数为4的RotaryMulGrad算子（x、r1、r2、dy均为必选输入）。
- RotaryMulGrad的可选属性need_backward（默认true）决定x输入是否接入融合节点：need_backward为true时接入x；
  为false时x不接入，RotaryPositionEmbeddingGrad的可选输入x为空。
- 融合后替换子图会基于原输入的shape/dtype/format重新推导输出shape，推导失败时放弃融合。
