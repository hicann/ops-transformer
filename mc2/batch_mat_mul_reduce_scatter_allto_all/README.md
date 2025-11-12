# BatchMatMulReduceScatterAlltoAll

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

算子功能：BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。

计算公式：大体计算流程为：BatchMatMul计算-->转置（yShardType等于0时需要）-->ReduceScatter集合通信-->Add-->AllToAll集合通信。计算逻辑如下，其中y为输出
$$
temp1 = BatchMatMul(x, weight)
$$
$$
temp2 = ReduceScatter(temp1)
$$
$$
temp3 = Add(temp2, bias)
$$
$$
y = AllToAll(temp3)
$$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 800px">
 <col style="width: 200px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>BatchMatMul计算的左矩阵，必须为3维。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>输入</td>
   <td>BatchMatMul计算的右矩阵，数据类型与x保持一致，必须为3维。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>输入</td>
   <td>Add计算的bias，需在ReduceScatter通信后执行Add操作。x为FLOAT16时，biasOptional需为FLOAT16；x为BFLOAT16时，biasOptional需为FLOAT32。支持两维或三维，支持传入空指针。</td>
   <td>FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>专家并行的通信域名，字符串长度需大于0且小于128。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>Tensor并行的通信域名，字符串长度需大于0且小于128。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>EP通信域size，支持2、4、8、16、32。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>TP通信域size，支持2、4、8、16、32。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>yShardType</td>
   <td>输入</td>
   <td>整型，0表示在H维度（BatchMatMul计算结果的第2维，结果共3维，维度索引依次为0、1、2）按tp进行ReduceScatter；1表示在C维度（BatchMatMul计算结果的第1维）按tp进行ReduceScatter。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>out</td>
   <td>输出</td>
   <td>Device侧的aclTensor，为batch_matmul计算+reduce_scatter计算+all_to_all通信的结果，数据类型与输入x保持一致，必须为3维。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>


## 约束说明

因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize，E表示专家数，C表示Capacity即一个专家里的token数，H表示每个token的长度，M表示tensor的高）
- 按H轴进行ReduceScatter场景，即yShardType为0场景：
  - x: $(E/ep, ep*C, M/tp)$
  - weight：$(E/ep, M/tp, H)$
  - biasOptional：非空指针情况下，三维时为$(E/ep, 1, H/tp)$，两维时为$(E/ep, H/tp)$
  - y：$(E, C, H/tp)$

- 按C轴进行ReduceScatter场景，即yShardType为1场景：
  - x: $(E/ep, ep*tp*C/tp, M/tp)$
  - weight：$(E/ep, M/tp, H)$
  - biasOptional：非空指针情况下，三维时为$(E/ep, 1, H)$，两维时为$(E/ep, H)$
  - y：$(E, C/tp, H)$

- 数据关系说明：
  - 比如x.size(0)等于E/tp，y.size(0)等于E，则表示，y.size(0) = ep*x.size(0)，y.size(0)是ep的整数倍；其他关系类似。
  - E的取值范围为[2, 512]，且E是ep的整数倍。
  - H的取值范围为：[1, 65535]，当yShardType为0时，H是tp的整数倍。
  - M/tp的取值范围为：[1, 65535]。
  - E/ep的取值范围为：[1, 32]。
  - ep、tp均仅支持2、4、8、16、32。
  - groupEp和groupTp名称不能相同。
  - C大于0，上限为算子device内存上限，当yShardType为1时，C是tp的整数倍。
  - 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
  - 不支持跨超节点，只支持超节点内。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_batch_mat_mul_reduce_scatter_allto_all.cpp](./examples/test_aclnn_batch_mat_mul_reduce_scatter_allto_all.cpp) | 通过[aclnnBatchMatMulReduceScatterAlltoAll](./docs/aclnnBatchMatMulReduceScatterAlltoAll.md)接口方式调用batch_mat_mul_reduce_scatter_allto_all算子。 |