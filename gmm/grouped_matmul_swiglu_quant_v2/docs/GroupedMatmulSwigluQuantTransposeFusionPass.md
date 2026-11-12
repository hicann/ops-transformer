# GroupedMatmulSwigluQuantTransposeFusionPass

## 融合模式

下图中的 `[Bitcast]` 表示可选节点，存在时保留，不由本 Pass 消除；不存在时直接连接对应源节点。图中的 Transpose 也可为 TransposeD。

所有模式均不改动 x、x_scale 等其他输入。融合成功后，将算子的 `transpose_weight` 从 `false` 设置为 `true`，由算子承担 weight 的转置语义。

### 非 MX 量化

非 MX 场景只融合 weight 侧的转置链，不改动 weight_scale 侧的连接和节点。

融合模式一：weight 经过单个 Transpose 或转置等价的 Reshape，消除该转置类节点，保留可选 Bitcast。

![非 MX：weight 单个转置类节点融合](../../../docs/zh/figures/GroupedMatmulSwigluQuantTransposeFusionPass_1.png)

融合模式二：weight 经过 Reshape → Transpose → Reshape，整条链等价于 weight 的轴交换时，消除这三个节点，保留可选 Bitcast。

![非 MX：weight 两段 Reshape 转置链融合](../../../docs/zh/figures/GroupedMatmulSwigluQuantTransposeFusionPass_2.png)

### MX 量化

#### 可融合转置链

MX 场景要求 weight 和 weight_scale 两侧均匹配到可融合转置链，才配套融合。可融合转置链有以下两类。

第一类：单个 Transpose/TransposeD，或转置等价的单个 Reshape。

“转置等价的单个 Reshape”指待交换的两个维度中至少有一个为 1，仅改变 shape、不重排数据即可得到与转置相同的结果。例如，weight 从 `[2, 128, 1]` Reshape 为 `[2, 1, 128]`，等价于交换最后两个维度。

![可融合转置链：单个转置类节点](../../../docs/zh/figures/GroupedMatmulSwigluQuantTransposeFusionPass_3.png)

第二类：Reshape → Transpose/TransposeD → Reshape。整条链须满足指定轴交换的 shape 和排列检查，不能将任意三节点链当作可融合结构。

![可融合转置链：两段 Reshape 与中间 Transpose](../../../docs/zh/figures/GroupedMatmulSwigluQuantTransposeFusionPass_4.png)

例如，weight 的 `[1, 128, 256] → [128, 256] → [256, 128] → [1, 256, 128]`，以及 MX scale 的 `[1, 128, 4, 2] → [128, 4, 2] → [4, 128, 2] → [1, 4, 128, 2]`，分别表示先去掉单位 group 轴、完成转置、再恢复 group 轴。

两侧可以分别选择不同结构：均为第一类、均为第二类，或一侧第一类、另一侧第二类。下文用“可融合转置链”统一表示这两类结构，不再逐一展开组合。

#### MXFP8

下图中，weight 侧没有 Bitcast，weight_scale 使用字节载体，通过 Bitcast 解释为 E8M0。

融合时，配套消除两侧的可融合转置链。融合后 weight 直接连接算子，weight_scale 仍经 Bitcast 连接算子。

![MXFP8：配套消除转置链，保留 scale 的 Bitcast](../../../docs/zh/figures/GroupedMatmulSwigluQuantTransposeFusionPass_5.png)

## 使用约束

- 算子的 `transpose_weight` 属性须为 `false`，融合后设为 `true`；已为 `true` 时不再融合。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
