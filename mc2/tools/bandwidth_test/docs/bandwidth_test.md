# aclnnBandwidthTest

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |

## 功能说明

- 接口功能：实现MOE（Mixture of Experts）场景下的`AlltoAll Dispatch`通信带宽测试。该算子用于测试NPU设备间的通信带宽性能，每个rank根据`dst_rank_id`将token发送到对应的目标rank，实现数据的分布式分发。

    执行流程（mode=0 完整流程）：
    1. SendToMoeExpert：根据`dst_rank_id`将token写入目标rank的通信窗口
    2. SetStatus：统计发送到各rank的token数量，写入状态区通知对端
    3. WaitDispatch：等待所有rank完成数据发送并写入状态
    4. LocalWindowCopy：从本地窗口读取接收到的数据，拷贝到输出tensor

    执行流程（mode=1 仅发送）：
    仅执行SendToMoeExpert，用于单向带宽测试。

- 计算公式：

    ```
    y = AlltoAllV(x, dst_rank_id)
    receiveCnt = 统计从各rank接收的token数量
    ```

    其中：
    - 每个token根据`dst_rank_id`指定的目标rank进行分发
    - 数据通过HCCL通信窗口在rank间传输
    - `receiveCnt`记录从每个rank接收到的token数量

## 函数原型

每个算子分为两段式接口，必须先调用 "aclnnBandwidthTestGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBandwidthTest"接口执行计算。

```cpp
aclnnStatus aclnnBandwidthTestGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstRankId,
    const char*      group,
    int64_t          worldSize,
    int64_t          maxBs,
    int64_t          mode,
    const char*      commAlg,
    int64_t          aivNum,
    aclTensor*       y,
    aclTensor*       receiveCnt,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBandwidthTest(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnBandwidthTestGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1200px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 280px">
    <col style="width: 150px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 120px">
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
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code>（BS=batch size，H=hidden size）</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dstRankId</td>
    <td>输入</td>
    <td>每个token的目标rank索引。</td>
    <td>要求为1D Tensor。每个元素表示对应token要发送到哪个rank。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS,)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>通信域名称。</td>
    <td>字符串长度范围为<code>[1, 128)</code>。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>输入</td>
    <td>通信域大小（rank总数）。</td>
    <td>取值范围<code>[2, 768]</code>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxBs</td>
    <td>输入</td>
    <td>最大batch size大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mode</td>
    <td>输入</td>
    <td>执行模式。</td>
    <td>取值范围：<ul><li>0：精度模式（完整流程：发送+同步+接收）</li><li>1：纯发模式（仅发送，用于单向带宽测试）</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>通信算法。</td>
    <td>可选参数，默认为空字符串。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>aivNum</td>
    <td>输入</td>
    <td>AIV核数量。</td>
    <td>可选参数，默认为-1（自动获取）。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出的token数据。</td>
    <td>要求为2D Tensor。数据类型与输入x一致。</td>
    <td>FLOAT16</td>
    <td>ND</td>
    <td><code>(totalReceiveCnt, H)</code>（totalReceiveCnt为从所有rank接收的token总数）</td>
    <td>√</td>
    </tr>
    <tr>
    <td>receiveCnt</td>
    <td>输出</td>
    <td>从各rank接收的token数量。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(worldSize,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回Device侧需申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回包含算子计算流程的op执行器。</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 500px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>输入和输出的shape不在支持的范围内。</td>
    </tr>
    <tr>
        <td>参数的取值不在支持的范围内（如worldSize、maxBs、mode等）。</td>
    </tr>
    </tbody>
    </table>

## aclnnBandwidthTest

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 550px">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnBandwidthTestGetWorkspaceSize</code>获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **确定性计算**：
  - aclnnBandwidthTest默认确定性实现。

- **驱动约束**：
  - 算子通信域各节点的驱动版本应当相同。

- **Shape变量约束**：

    | 变量         | 定义与取值范围                                                                 |
    | :----------- | :----------------------------------------------------------------------------- |
    | worldSize    | 表示通信域中rank总数，取值范围<code>[2, 768]</code>。                         |

- **数据类型约束**：
    - 输入`x`和输出`y`的数据类型必须一致，支持FLOAT16、BFLOAT16、FLOAT32。
    - 输入`dstRankId`的数据类型必须为INT32。
    - 输出`receiveCnt`的数据类型必须为INT32。

- **参数约束**：
    - `dstRankId`中每个元素的取值范围必须为<code>[0, worldSize)</code>。
    - `mode`取值必须为0或1。
    - `group`字符串长度必须小于128。

- **环境变量约束**：
    - 调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB。
    - 设置大小要求：<code>≥ 2 *(maxBs* worldSize *H* sizeof(dtype) + 2MB)</code>。

- **通信域使用约束**：
    - 一个通信域内的节点需在一个超节点内，不支持跨超节点。

- **其他约束**：
    - 公式中的"×"表示乘法。
