# aclnnElasticReceivableTest

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

算子功能：对一个通信域内的所有卡发送数据并写状态位，用于检测通信链路是否正常。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnElasticReceivableTestGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnElasticReceivableTest”接口执行计算。

```cpp
aclnnStatus aclnnElasticReceivableTestGetWorkspaceSize(
    const aclTensor *dstRank,
    const char      *group,
    int64_t          worldSize,
    int64_t          rankNum,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnElasticReceivableTest(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnElasticReceivableTestGetWorkspaceSize

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
    <td>dstRank</td>
    <td>输入</td>
    <td>表示同一个通信域内目的server内的通信卡。</td>
    <td>shape为(rankNum,)，表示在同一个server内的卡号</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>ep通信域名称，专家并行的通信域。</td>
    <td>字符串长度范围为(0, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>输入</td>
    <td>通信域大小。</td>
    <td>取值支持[16, 128]内16整数倍的数值</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>rankNum</td>
    <td>输入</td>
    <td>本端需要发送的目的server内的卡数。</td>
    <td>当前只支持16卡</td>
    <td>INT64</td>
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
    <td>传入的dstRank或group是空指针。</td>
    </tr>
    <tr>
    <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3" align="left">161002</td>
    <td align="left">传入的dstRank的数据类型不在支持的范围内。</td>
    </tr>
    <tr><td align="left">传入的dstRank的数据格式不在支持的范围内。</td></tr>
    <tr><td align="left">传入的dstRank的shape不匹配。</td></tr>
    </tbody></table>

## aclnnElasticReceivableTest

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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnElasticReceivableTestGetWorkspaceSize获取。</td>
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

