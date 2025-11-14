# AlltoAllAllGatherBatchMatMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

算子功能：完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。

计算公式：
计算逻辑如下，其中y1、y2、y3为输出
$$
x1 = AllToAll(x)
$$
$$
y2 = AllGather(x1)
$$
$$
y3 = BatchMatMul(y2, weight, bias)
$$
$$
y1 = 激活函数(y3)
$$



## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
   <td>通信后结果作为BatchMatMul计算的左矩阵。该输入进行AllToAll、AllGather集合通信，必须为3维。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>输入</td>
   <td>BatchMatMul计算的右矩阵。数据类型与x保持一致，必须为3维。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>输入</td>
   <td>BatchMatMul计算的bias。x为FLOAT16时，biasOptional需为FLOAT16；x为BFLOAT16时，biasOptional需为FLOAT32，支持两维或三维。支持传入空指针。</td>
   <td>FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>ep通信域名称，专家并行的通信域。字符串长度需大于0，小于128。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>tp通信域名称，Tensor并行的通信域。字符串长度需大于0，小于128。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>ep通信域size，支持2/4/8/16/32。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>tp通信域size，支持2/4/8/16/32。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>xShardType</td>
   <td>输入</td>
   <td>0表示在H维度（即x的第2维，x为3维，分别为第0维、第1维、第2维）按tp域进行allgather，1表示在C维度（即x的第1维）上按tp域进行allgather。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>actType</td>
   <td>输入</td>
   <td>激活函数类型，支持0/1/2/3/4的输入，0表示无激活函数，对应关系为[0：None，1：GELU，2：Silu，3：Relu，4：FastGELU]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>y1Out</td>
   <td>输出</td>
   <td>最终计算结果，如果有激活函数则为激活函数的输出，否则为BatchMatMul的输出。支持3维，数据类型与输入x保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y2OutOptional</td>
   <td>输出</td>
   <td>可选输出，AllGather的输出，反向可能需要。支持3维，数据类型与输入x保持一致。空指针表示不需要该输出。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y3OutOptional</td>
   <td>输出</td>
   <td>可选输出，有激活函数时，BatchMatMul的输出。支持3维，数据类型与输入x保持一致。空指针表示不需要该输出。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>


## 约束说明

因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize）

按H轴进行AllGather场景，即xShardType为0场景：
  - x: (E, C, H/tp)
  - weight：(E/ep, H, M/tp)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, M/tp)，两维时为(E/ep, M/tp)
  - y1Out：(E/ep, ep*C, M/tp)
  - y2OutOptional：(E/ep, ep*C, H)
  - y3OutOptional：(E/ep, ep*C, M/tp)

按C轴进行AllGather场景，即xShardType为1场景：
  - x: (E, C/tp, H)
  - weight：(E/ep, H, M/tp)
  - biasOptional：非空指针情况下，三维时为(E/ep, 1, M/tp)，两维时为(E/ep, M/tp)
  - y1Out：(E/ep, ep*tp\*C/tp, M/tp)
  - y2OutOptional：(E/ep, ep*tp\*C/tp, H)
  - y3OutOptional：(E/ep, ep*tp\*C/tp, M/tp)

数据关系说明：
  - 比如x.size(0)等于E，weight.size(0)等于E/ep，则表示，x.size(0) = ep*weight.size(0)，x.size(0)是ep的整数倍；其他关系类似。
  - E的取值范围为[2, 512]，且E是ep的整数倍。
  - H的取值范围为：[1, 65535]，当xShardType为0时，H是tp的整数倍。
  - M/tp的取值范围为：[1, 65535]。
  - E/ep的取值范围为：[1, 32]。
  - ep、tp均仅支持2、4、8、16、32。
  - groupEp和groupTp名称不能相同。
  - C必须大于0，上限为算子device内存上限，当xShardType为1时，C是tp的整数倍。
  - 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
  - 不支持跨超节点，只支持超节点内。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_allto_all_all_gather_batch_mat_mul.cpp](./examples/test_aclnn_allto_all_all_gather_batch_mat_mul.cpp) | 通过[aclnnAlltoAllAllGatherBatchMatMul](./docs/aclnnAlltoAllAllGatherBatchMatMul.md)接口方式调用allto_all_all_gather_batch_mat_mul算子。 |