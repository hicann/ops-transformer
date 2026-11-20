# GroupedMatmulFinalizeRoutingTransposeFusionPass

## 融合模式

下图中的 `[Bitcast]` 表示可选节点，存在时保留，不由本 Pass 消除；不存在时直接连接对应源节点。图中的 Transpose 也可为 TransposeD。

所有模式均不改动 x、x_scale 等其他输入。融合成功后，将算子的 `transpose_w` 从 `false` 设置为 `true`，由算子承担 weight 的转置语义。

### 非 MX 量化

非 MX 场景要求 weight 和 weight_scale 两侧配套匹配，其中 weight_scale 仅支持 Reshape → Transpose → Reshape 结构。以下是支持的特殊图结构，不表示非 MX 量化调用必然生成这条 scale 链。

融合模式一：weight 经过单个 Transpose 或转置等价的 Reshape，weight_scale 经过 Reshape → Transpose → Reshape。配套消除两侧转置链，保留已有的 Bitcast。

![非 MX：weight 单节点与 scale 两段 Reshape 链配套融合](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_1.png)

融合模式二：weight 和 weight_scale 均经过 Reshape → Transpose → Reshape。配套消除两侧的三节点转置链，保留已有的 Bitcast。

![非 MX：两侧两段 Reshape 转置链配套融合](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_2.png)

### MX 量化

#### 可融合转置链

MX 场景要求 weight 和 weight_scale 两侧均匹配到可融合转置链，才配套融合。可融合转置链有以下两类。

第一类：单个 Transpose/TransposeD，或转置等价的单个 Reshape。

“转置等价的单个 Reshape”指待交换的两个维度中至少有一个为 1，仅改变 shape、不重排数据即可得到与转置相同的结果。例如，weight 从 `[2, 128, 1]` Reshape 为 `[2, 1, 128]`，等价于交换最后两个维度。

![可融合转置链：单个转置类节点](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_3.png)

第二类：Reshape → Transpose/TransposeD → Reshape。整条链须满足指定轴交换的 shape 和排列检查，不能将任意三节点链当作可融合结构。

![可融合转置链：两段 Reshape 与中间 Transpose](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_4.png)

例如，weight 的 `[1, 128, 256] → [128, 256] → [256, 128] → [1, 256, 128]`，以及 MX scale 的 `[1, 128, 4, 2] → [128, 4, 2] → [4, 128, 2] → [1, 4, 128, 2]`，分别表示先去掉单位 group 轴、完成转置、再恢复 group 轴。

两侧可以分别选择不同结构：均为第一类、均为第二类，或一侧第一类、另一侧第二类。下文用“可融合转置链”统一表示这两类结构，不再逐一展开组合。

#### MXFP8

下图中，weight 侧没有 Bitcast，weight_scale 使用字节载体，通过 Bitcast 解释为 E8M0。

融合时，配套消除两侧的可融合转置链。融合后 weight 直接连接算子，weight_scale 仍经 Bitcast 连接算子。

![MXFP8：配套消除转置链，保留 scale 的 Bitcast](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_5.png)

#### MXFP4

下图表示 weight 使用字节载体、经 Bitcast 和尾部 Reshape 恢复逻辑 FP4 shape，weight_scale 经 Bitcast 解释为 E8M0 的调用路径。

融合时，配套消除 weight、weight_scale 的可融合转置链；保留 weight 的 Bitcast、尾部打包 Reshape，以及 weight_scale 的 Bitcast，并更新打包链的 shape、输入描述及相关 Shape 依赖。

![MXFP4：配套消除转置链，保留两侧 Bitcast 和 weight 打包 Reshape](../../../docs/zh/figures/GroupedMatmulFinalizeRoutingTransposeFusionPass_6.png)

weight 的尾部打包 Reshape 不属于“可融合转置链”，不能随转置链一起删除。

## 使用约束

- 算子的 `transpose_w` 属性须为 `false`，融合后设为 `true`；已为 `true` 时不再融合。

### MXFP4、MXFP8 单位轴转置的处理行为

以下 K、N 均指逻辑维度。对于 MXFP4 和 MXFP8，weight 的 K 轴与 weight_scale 的量化块轴不是同一个轴；`ceil(K/64)=1` 表示只有一个量化块，并不要求 `K=1`。是否触发本 GE Pass，取决于前端优化后的两侧转置链能否配套匹配。

#### MXFP4

使用包含 MXFP4 单位轴适配（TorchAir PR #4001）的版本时，ND 字节载体转置路径满足其视图及布局检查后，以下场景由 TorchAir 预先处理：

- 动态图：`N=1` 或 Scale 块数 `ceil(K/64)=1`，且 Scale 转置降为等价 Reshape 或两个单位轴间的转置已被省略。
- 静态图：`N=1` 且 `ceil(K/64)=1`，Weight 转置降为等价 Reshape，Scale 两个单位轴间的转置已被省略。

TorchAir 处理后已设置 `transpose_w=true`，该节点不再触发本 GE Pass；未被接管的路径仍须满足前述 GE 融合模式。上述条件不放宽算子自身的 shape 约束。

#### MXFP8

MXFP8 不由上述 TorchAir MXFP4 专用逻辑接管。weight 交换逻辑 N、K 轴，weight_scale 交换 N 轴与量化块轴，分别判断单位轴：

- 动态图：weight 的 `K=1` 或 `N=1`，或 weight_scale 的 `N=1` 或 `ceil(K/64)=1` 时，相应的单位轴转置可能改写为 Reshape 或被省略。若优化后的两侧转置链无法配套匹配，则不触发本 GE Pass。
- 静态图：`K=N=1` 且两侧的恒等转置均已被省略时，不触发本 GE Pass。若仍保留可配套匹配的转置链（包括等价 Reshape），则仍可触发融合。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
