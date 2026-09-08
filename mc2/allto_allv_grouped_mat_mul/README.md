# AlltoAllvGroupedMatMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，**先通信后计算**。

- **计算公式**：
    - 路由专家：

    $$
    ataOut = AlltoAllv(gmmX) \\
    permuteOut = Permute(ataOut) \\
    gmmY = permuteOut \times gmmWeight
    $$

    - 共享专家：

    $$
    mmY = mmX \times mmWeight
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>gmmX</td>
        <td>输入</td>
        <td>该输入进行AlltoAllv通信与Permute操作后结果作为GroupedMatMul计算的左矩阵。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul计算的右矩阵。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的左矩阵。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的右矩阵。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>专家并行的通信域名，字符串长度要求(0, 128)。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>ep通信域的大小。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>输入</td>
        <td>表示发送给其他卡的token数。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>输入</td>
        <td>表示接收其他卡的token数。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul的右矩阵是否需要转置。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>输入</td>
        <td>共享专家MatMul的右矩阵是否需要转置。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>输入</td>
        <td>permuteOutOptional是否需要输出。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>输出</td>
        <td>路由专家计算的输出。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>输出</td>
        <td>共享专家计算的输出。</td>
        <td>与mmXOptional保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>输出</td>
        <td>permute之后的输出。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    </tbody></table>

## 约束说明

- 通信引擎约束：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持AICPU通信。
  - <term>Ascend 950DT</term>：支持CCU通信和AICPU通信，CCU仅支持单机UB域内互联，AI_CPU可支持跨机UB域内互联。

- 确定性计算：
  - aclnnAlltoAllvGroupedMatMul默认确定性实现。

- 参数说明里shape使用的变量：
  - BSK：本卡发送的token数，是sendCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数。除下述AIV附加约束外，e<=32，e * epWorldSize最大支持256。
  - N1：表示路由专家的head_num，取值范围(0, 65536)。
  - N2：表示共享专家的head_num，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡收到的token数，是recvCounts参数累加之和。
  - ep通信域内所有卡的A参数的累加和等于所有卡上的BSK参数的累加和。

- AIV模式的当前保留实现：
  - Atlas A2的epWorldSize支持2、4、8；Atlas A3支持2、4、8、16、32、64、128。gmmX、gmmWeight、gmmY必须为相同的FLOAT16或BFLOAT16类型且为ND格式。
  - AIV模式下BSK/A取值范围为[1, 5000000]，H1/N1取值范围为[1, 65535]；本卡专家数e不超过512，`epWorldSize * e`不超过1024。
  - sendCounts和recvCounts接口属性仍为INT64直接计数数组，布局为`[rank][localExpert]`。Host侧校验每项非负且不超过对应本卡输入/输出M，并校验sendCounts之和等于BSK、recvCounts之和等于A。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  : 单卡通信量在2MB以下可能存在性能劣化。

## 调用说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_grouped_mat_mul.cpp)。 | 通过[aclnnAlltoAllvGroupedMatMul](./docs/aclnnAlltoAllvGroupedMatMul.md)接口方式调用allto_allv_grouped_mat_mul算子。 |

- <term>Ascend 950DT</term>:

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/arch35/test_aclnn_allto_allv_grouped_mat_mul.cpp)。 | 通过[aclnnAlltoAllvGroupedMatMul](./docs/aclnnAlltoAllvGroupedMatMul.md)接口方式调用allto_allv_grouped_mat_mul算子。 |
