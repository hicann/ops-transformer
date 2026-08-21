# GroupedMatmulTransFusionPass

## 融合模式

### 融合模式一：
将Transpose或TransposeD从图中删除，并将x/weight转置信息添加在算子属性上。如下图所示。

该融合模式支持的产品如下：
<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->
**场景一**：

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_1.png)

**场景二**：

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_2.png)

<!-- npu="950,A3,910b" id1 -->
### 融合模式二：
将x/weight前的Reshape+Transpose/TransposeD+Reshape从图中删除，并将x/weight转置信息打在算子属性上。如下图所示。

该融合模式支持的产品如下：

<!-- npu="910b" id2 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id3 -->

<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->

**场景一**：

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_3.png)

**场景二**：

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_4.png)

<!-- end id1 -->
<!-- npu="950" id5 -->
### 融合模式三：
Ascend 950PR/Ascend 950DT的伪量化场景下，将weight/antiquantScale前的Reshape+Transpose/TransposeD+Reshape从图中删除，并将weight转置信息打在算子属性上。如下图所示。

该融合模式支持的产品如下：
<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_5.png)

### 融合模式四：
Ascend 950PR/Ascend 950DT的MX/GB量化场景下，融合模式是：将Transpose或TransposeD从图中删除，并将x和weight的转置信息打在算子属性上。如下图所示：

该融合模式支持的产品如下：
<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->

![](../../../docs/zh/figures/GroupedMatmulTransFusionPass_6.png)

>[!NOTE]说明
>Ascend 950PR/Ascend 950DT的MX/GB量化场景下，scale跟随weight的转置信息，pertokenScale跟随x的转置信息。
<!-- end id5 -->

## 使用约束

- 只支持输入x为单tensor、weight为单tensor、y为单tensor的场景（单tensor表示tensorList输入中只有一个tensor）。
- Transpose/TransposeD支持第二轴和第三轴转置。
