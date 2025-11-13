# aclnnMoeDistributeBufferReset

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：对EP通信域做数据区与状态区的清理。若当前机器为未被隔离机器，则对其进行通信域的重置操作，对有效的die进行数据区和状态区的清0，确保后续使用时通信域不会存在已被隔离机器的数据或状态信息。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnMoeDistributeBufferResetGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeBufferReset”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeBufferResetGetWorkspaceSize(
    const aclTensor *elasticInfo,
    const char      *groupEp,
    int32_t          epWorldSize,
    int32_t          needSync,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnMoeDistributeBufferReset(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMoeDistributeBufferResetGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 250px">
    <col style="width: 200px">
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 100px">
    <col style="width: 100px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>elasticInfo</td>
    <td>输入</td>
    <td>Device侧的aclTensor，有效rank掩码表，标识有效rank的tensor，其中0标识本卡与对应rank链路不通，1为联通</td>
    <td>shape为(epWorldSize,)</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>ep通信域名称，专家并行的通信域。</td>
    <td>字符串长度范围为(0, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>通信域大小。</td>
    <td>取值支持[16, 128]内16整数倍的数值</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>needSync</td>
    <td>输入</td>
    <td>是否需要全卡同步。</td>
    <td>取值支持0或1，0表示不需要，1表示需要。</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
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
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 400px">
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
    <td>传入的elasticInfo或groupEp是空指针。</td>
    </tr>
    <tr>
    <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3" align="left">161002</td>
    <td align="left">传入的elasticInfo的数据类型不在支持的范围内。</td>
    </tr>
    <tr><td align="left">传入的elasticInfo的数据格式不在支持的范围内。</td></tr>
    <tr><td align="left">传入的elasticInfo的shape不匹配。</td></tr>
    </tbody></table>

## aclnnMoeDistributeBufferReset

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 800px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeBufferResetGetWorkspaceSize获取。</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输入</td>
    <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>输入</td>
    <td>指定执行任务的Stream。</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无

