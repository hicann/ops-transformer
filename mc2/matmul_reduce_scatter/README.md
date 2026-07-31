# MatmulReduceScatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如Bus Error等。

## 功能说明

- 接口功能：完成`Matmul`计算和`ReduceScatter`通信的融合，先计算后通信。融合算子内部实现计算和通信流水并行。

- 计算公式：

    $$
    output=ReduceScatter(x1@x2+bias)
    $$

    - x1指Matmul计算的左矩阵。
    - x2指Matmul计算的右矩阵。
    - bias指Matmul计算的偏置。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMatmulReduceScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulReduceScatter”接口执行计算。

```cpp
aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* bias,
    const char*      group,
    const char*      reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor* output,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulReduceScatter(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMatmulReduceScatterGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1</td>
        <td>输入</td>
        <td>Matmul计算的左矩阵输入，即计算公式中的x1。</td>
        <td>
            <ul>
                <li>shape为（m，k）。</li>
                <li>支持空Tensor。</li>
            </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>输入</td>
        <td>Matmul计算的右矩阵输入，即计算公式中的x2。</td>
        <td>
            <ul>
                <li>不转置场景的shape为（k，n），转置场景的shape为（n，k）。</li>
                <li>支持空Tensor。</li>
                <li>与x1的数据类型保持一致。</li>
                <li>当前版本仅支持二维输入，支持转置/不转置场景。</li>
                <li>仅支持两根轴转置情况下的<a href="../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>，其他场景的非连续的Tensor不支持。</li>
            </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>输入</td>
        <td>Matmul计算的偏置，即计算公式中的bias。</td>
        <td>
            <ul>
                <li>shape为（n）。</li>
                <li>支持传入空指针场景。</li>
                <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：暂不支持输入为非0的场景。</li>
                <li><term>Ascend 950PR/Ascend 950DT</term>：支持输入为非0的场景。</li>
            </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>输入</td>
        <td>reduce操作类型。</td>
        <td>当前版本仅支持“sum”。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>输入</td>
        <td>通信数据切分数，即总数据量/单次通信量。</td>
        <td>当前版本仅支持输入0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>输入</td>
        <td>流模式的枚举。</td>
        <td>streamMode：任务失败后的流处理模式。取值范围如下：
            1表示ACL_STOP_ON_FAILURE，某个任务失败后，停止执行后续的任务。此模式为当前版本唯一支持的值。
    </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>输出</td>
        <td>AllGather通信与MatMul计算的结果，即计算公式中的output。</td>
        <td>
            <ul>
                <li>shape为(m/rank_size, n)。</li>
                <li>支持空Tensor。</li>
                <li>与x1的数据类型保持一致。</li>
            </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x1、x2或output是空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>x1、x2、bias或output的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>streamMode不在合法范围内。</td>
    </tr>
    <tr>
        <td>x1、x2或output的shape不符合约束要求。</td>
    </tr>
    </tbody></table>

## aclnnMatmulReduceScatter

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMatmulReduceScatterGetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 通信引擎约束：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持AICPU通信。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：仅支持AICPU通信。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持AICPU通信。

- 参数说明中shape涉及的变量说明：
  - m为卡数rank_size的整数倍。
  - k的取值范围为[256, 65535)。
  - x1/x2支持的空tensor场景，m和n可以为0，k不可为0，且需满足以下条件：
    - m为0，k不为0，n不为0；
    - m不为0，k不为0，n为0；
    - m为0，k不为0，n为0。
  - rank_size为卡数。
- x2支持转置/不转置场景，x1只支持不转置场景。
- x1、x2计算输入的数据类型要和output计算输出的数据类型一致。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 支持2、4、8卡，并且仅支持hccs链路all mesh组网。
  - 一个模型中的通算融合MC2算子，仅支持相同通信域。
  - aclnnMatmulReduceScatter默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
  - 支持2、4、8、16、32卡，并且仅支持hccs链路double ring组网。
  - aclnnMatmulReduceScatter默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16、32、64卡，并且仅支持hccs链路all mesh组网。
  - ReduceScatter集合通信数据总量不能超过16*256MB，集合通信数据总量计算方式为：m* n * sizeof(output_dtype)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。
  - aclnnMatmulReduceScatter默认为确定性实现。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_matmul_reduce_scatter.cpp](./examples/test_aclnn_matmul_reduce_scatter.cpp) | 通过[aclnnMatmulReduceScatter](./docs/aclnnMatmulReduceScatter.md)接口方式调用matmul_reduce_scatter算子。 |
