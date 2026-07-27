# aclnnMoeDistributeCombineSetup

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_setup)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **接口功能**：进行AlltoAllV通信，将数据写入对端GM。
- **计算公式**：

    $$
    ataOut = AllToAllV(expandX)\\
    $$

    按MoeDistributeDispatchSetup和MoeDistributeDispatchTeardown算子收集数据的路径原路返还，本算子只做通信状态和通信数据的发送，数据发送后即刻退出，无需等待通信完成，通信状态确认和数据后处理由aclnnMoeDistributeCombineTeardown完成。

- **注意**：该接口必须与aclnnMoeDistributeDispatchSetup、aclnnMoeDistributeDispatchTeardown及aclnnMoeDistributeCombineTeardown配套使用。

## 函数原型

该算子分为两段式接口，必须先调用“`aclnnMoeDistributeCombineSetupGetWorkspaceSize`”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“`aclnnMoeDistributeCombineSetup`”接口执行计算。为用户提供“aclnnMoeDistributeCombineSetupTeardownCalcOutputSize”接口计算“aclnnMoeDistributeCombineSetup”部分输出的size大小。

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBs,
    int64_t          commQuantMode,
    int64_t          commType,
    const char*      commAlg,
    aclTensor*       quantExpandXOut,
    aclTensor*       commCmdInfoOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetup(
    void             *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream      stream)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor *expandX,
    const aclTensor *expertIds,
    const aclTensor *assistInfoForCombine,
    const char      *groupEp,
    int64_t         epWorldSize,
    int64_t         epRankId,
    int64_t         moeExpertNum,
    int64_t         expertShardType,
    int64_t         sharedExpertNum,
    int64_t         sharedExpertRankNum,
    int64_t         globalBs,
    int64_t         commQuantMode,
    int64_t         commType,
    const char      *commAlg,
    uint64_t        &tokenMsgSize,
    uint64_t        &commCmdInfoOutSize)
```

## aclnnMoeDistributeCombineSetupGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1434px"><colgroup>
    <col style="width: 156px">
    <col style="width: 71px">
    <col style="width: 338px">
    <col style="width: 450px">
    <col style="width: 87px">
    <col style="width: 67px">
    <col style="width: 170px">
    <col style="width: 95px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX</td>
        <td>输入</td>
        <td>自刷新参数，根据expertIds进行扩展过的token特征</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>每个token的topK个专家索引</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeDispatchTeardown中的assistInfoForCombine输出</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>输入</td>
        <td>EP通信域名称</td>
        <td>字符串长度范围为[1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>EP通信域size</td>
        <td>取值支持[2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>输入</td>
        <td>EP域本卡Id</td>
        <td>取值范围[0, epWorldSize)。<br>同一个EP通信域中各卡的epRankId不重复。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量</td>
        <td>取值范围(0, 512]。<br>满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>输入</td>
        <td>共享专家卡分布类型</td>
        <td>当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>输入</td>
        <td>共享专家数量</td>
        <td>当前取值范围[0, 4]。0表示无共享专家。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>输入</td>
        <td>共享专家卡数量</td>
        <td>当前取值范围[0, epWorldSize / 2]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>输入</td>
        <td>EP域全局的batch size大小</td>
        <td>当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize或globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>输入</td>
        <td>通信量化类型</td>
        <td>当前仅支持传入0，表示不进行量化。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>输入</td>
        <td>通信方案选择</td>
        <td>当前仅支持2，表示URMA通路。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法选择</td>
        <td>预留字段，当前版本不支持，传空指针或空字符串即可。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantExpandXOut</td>
        <td>输出</td>
        <td>量化处理后的token</td>
        <td>不支持空Tensor。</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(A, tokenMsgSize)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>commCmdInfoOut</td>
        <td>输出</td>
        <td>通信的cmd信息</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>((A + epWorldSize) * 16, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

  - Ascend 950DT：
    - 不支持共享专家场景。
    - epWorldSize当前取值仅支持2、8。
    - moeExpertNum表示MoE专家数量，当前仅能传入32。
    - expertShardType当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum表示共享专家数量，当前不支持共享专家，仅能传入0。
    - sharedExpertRankNum表示共享专家卡数，当前不支持共享专家，仅能传入0。
    - commQuantMode当前仅支持传入0，表示不进行量化。
    - commType取值范围[0, 2]，当前仅支持2，表示URMA通路。
    - commAlg当前版本不支持，传空指针即可。

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR </td>
        <td>161001</td>
        <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID </td>
        <td>161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>输入和输出的shape不在支持的范围内。</td>
    </tr>
    <tr>
        <td>参数的取值不在支持的范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeCombineSetup

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 670px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineSetupGetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的stream流。</td>
    </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnMoeDistributeCombineSetupTeardownCalcOutputSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1434px"><colgroup>
    <col style="width: 156px">
    <col style="width: 71px">
    <col style="width: 338px">
    <col style="width: 450px">
    <col style="width: 87px">
    <col style="width: 67px">
    <col style="width: 170px">
    <col style="width: 95px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX</td>
        <td>输入</td>
        <td>自刷新参数，根据expertIds进行扩展过的token特征</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>每个token的topK个专家索引</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeDispatchTeardown中的assistInfoForCombine输出</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>输入</td>
        <td>EP通信域名称</td>
        <td>字符串长度范围为[1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>EP通信域size</td>
        <td>取值支持[2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>输入</td>
        <td>EP域本卡Id</td>
        <td>取值范围[0, epWorldSize)。<br>同一个EP通信域中各卡的epRankId不重复。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量</td>
        <td>取值范围(0, 512]。<br>满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>输入</td>
        <td>共享专家卡分布类型</td>
        <td>当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>输入</td>
        <td>共享专家数量</td>
        <td>当前取值范围[0, 4]。0表示无共享专家。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>输入</td>
        <td>共享专家卡数量</td>
        <td>当前取值范围[0, epWorldSize / 2]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>输入</td>
        <td>EP域全局的batch size大小</td>
        <td>当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize或globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>输入</td>
        <td>通信量化类型</td>
        <td>当前仅支持传入0，表示不进行量化。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>输入</td>
        <td>通信方案选择</td>
        <td>当前仅支持2，表示URMA通路。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法选择</td>
        <td>预留字段，当前版本不支持，传空指针或空字符串即可。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>tokenMsgSize</td>
        <td>输出</td>
        <td>aclnnMoeDistributeCombineSetup接口quantExpandX第二维tokenMsgSize大小，表示每个token在数据通信时的维度信息。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>Align512(Align32(H) + Align8(H) / 8 * sizeof(float))，其中AlignN(x) = ((x + N - 1) / N) * N。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commCmdInfoOutSize</td>
        <td>输出</td>
        <td>aclnnMoeDistributeCombineSetup接口commCmdInfoOut的大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>(A + epWorldSize) * 16</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeDistributeCombineSetup默认确定性实现。
- aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口必须配套使用。
- 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值所有卡需保持一致，`groupEp`、`epWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值在网络中不同层中也需保持一致，且和aclnnMoeDistributeDispatchSetup接口、aclnnMoeDistributeDispatchTeardown接口、aclnnMoeDistributeCombineTeardown接口对应参数也保持一致。
- 参数说明里shape格式说明：
  - A：表示本卡需要分发的最大token数量，取值范围如下：
    - 对于共享专家，当globalBs为0时，要满足A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum；当globalBs非0时，要满足A = globalBs\* sharedExpertNum / sharedExpertRankNum。
    - 对于MoE专家，当globalBs为0时，要满足A >= BS \* epWorldSize \* min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs \* min(localExpertNum, K)。
  - H：表示hidden size隐藏层大小。取值为[1024, 8192]。当前仅支持4096、7168。
  - BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0 < BS ≤ 512。当前仅支持8、16、256。
  - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。当前仅支持6、8。
  - localExpertNum：表示本卡专家数量。
    - 对于共享专家卡，localExpertNum = 1
    - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)。moeExpertNum当前仅支持32。
  - tokenMsgSize：表示每个token在数据通信时的维度信息，计算公式为Align512(Align32(H)+Align8(H)/8\*sizeof(float))，其中AlignN(x)=((x+N-1)/N*N)。
  - 当前不支持共享专家。sharedExpertNum和sharedExpertRankNum当前仅支持0。
- HCCL_BUFFSIZE：
  调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
  - Ascend 950DT：
    - 要求 >= 2且满足>= 4 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32。
- 通信域使用约束：
  - 一个模型中的aclnnMoeDistributeDispatchSetup、aclnnMoeDistributeDispatchTeardown、aclnnMoeDistributeCombineSetup、aclnnMoeDistributeCombineTeardown仅支持相同EP通信域，且该通信域中不允许有其他算子。
- 通信方式约束：
    - <term>Ascend 950DT</term>：仅支持URMA通信。

## 调用示例

- 文件准备：

    1. 按照下方指导创建rank_table_m2.json文件并修改。

    2. 将项目拷贝到两台服务器中，并根据机器的device ip配置rank_table_m2.json文件内容。注意两机rank_table_m2.json文件保持一致。

    3. 安装cann包，并根据[算子调用](../../../docs/zh/invocation/quick_op_invocation.md)编译运行。

- 关于rankTable:

    1. 开发者可以通过ranktable文件配置参与集合通信的NPU资源信息，详细配置请参考[《集合通信用户指南》](https://hiascend.com/document/redirect/CannCommunityHcclUg)中“通信功能开发>集群信息配置>ranktable文件配置资源信息”。

    2. 使用`cat /etc/hccn.conf`或者`for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done`查询机器的device ip。然后参考集合通信文档填写json文件。

    > 注意：两机16卡场景中，两机器的device_id都是0~7，其中一台机器的rank_id为0~7，另一台机器的rank_id为8~15。单机16卡场景中，device_id和rank_id都是0~15。

- 环境变量配置：

    ```bash
    # 运行前需设置三个环境变量
    ## FIRST_RANK_ID说明：以两机16卡为例，其中一机器设置为0，另一机器设置为8
    ## 如export FIRST_RANK_ID=0
    export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
    export FIRST_RANK_ID=<设备的起始rank_id>
    ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量，以两机16卡为例，将两台机器设置为16
    export ENV_DEV_NUM=16

- 机器数量设置：

    两机16卡场景中，需将参数MACHINE_NUM设置为2，即

    ```Cpp
    const uint32_t MACHINE_NUM = 2;
    ```

    单机16卡场景则无需修改。

- Ascend 950DT：

    示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <cstring>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_combine_setup.h"
    #include "aclnnop/aclnn_moe_distribute_combine_teardown.h"

    #define CHECK_RET(cond, return_expr)                                                                                   \
        do {                                                                                                               \
            if (!(cond)) {                                                                                                 \
                return_expr;                                                                                               \
            }                                                                                                              \
        } while (0)

    #define LOG_PRINT(message, ...)                                                                                        \
        do {                                                                                                               \
            printf(message, ##__VA_ARGS__);                                                                                \
        } while (0)

    constexpr int DEV_NUM = 2;

    template <typename Func>
    class Guard {
    public:
        explicit Guard(Func &func) : func_(func)
        {
        }
        ~Guard()
        {
            func_();
        }

    private:
        Func &func_;
    };

    int64_t AlignN(int64_t x, int64_t n)
    {
        return (x + n - 1) / n * n;
    }

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream combinesetupstream;
        aclrtStream combineteardownstream;
        aclrtContext context;
    };

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    void DestroyTensor(aclTensor *tensor)
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }

    void FreeDeviceAddr(void *deviceAddr)
    {
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    int LaunchOneProcess(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return ret);

        auto destroyFunc = [&args]() {
            std::cout << "== begin to destroy " << std::endl;
            HcclCommDestroy(args.hcclEpComm);
            aclrtDestroyStream(args.combinesetupstream);
            aclrtDestroyStream(args.combineteardownstream);
            aclrtDestroyContext(args.context);
            aclrtResetDevice(args.rankId);
        };
        auto guard = Guard<decltype(destroyFunc)>(destroyFunc);

        // 场景参数
        int64_t bs = 16;
        int64_t h = 4096;
        int64_t k = 6;
        int64_t expertSharedType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 32;
        int64_t commQuantMode = 0;
        int64_t tpWorldSize = 1;
        int64_t epWorldSize = DEV_NUM;
        int64_t commType = 2;
        int64_t timeOut = 100000000;

        int64_t globalBS = bs * epWorldSize;
        int64_t localExpertNum;
        int64_t localToken;

        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            localToken = globalBS / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
            localToken = globalBS * (localExpertNum < k ? localExpertNum : k);
        }

        uint64_t assistInfoForCombineOutSize = localToken * 128;

        // 定义shape
        std::vector<int64_t> expandXShape{tpWorldSize * localToken, h};
        std::vector<int64_t> expertIdsShape{bs, k};
        std::vector<int64_t> expertScalesShape{bs, k};
        std::vector<int64_t> expandIdxShape{bs * k};
        std::vector<int64_t> assistInfoForCombineOutShape{assistInfoForCombineOutSize};
        std::vector<int64_t> quantExpandXOutShape{tpWorldSize * localToken,
                                                AlignN(AlignN(h, 32) + AlignN(h, 8) / 8 * sizeof(float), 512)};
        std::vector<int64_t> commCmdInfoOutShapeforcombine{(localToken + epWorldSize) * 16};
        std::vector<int64_t> xOutShape{bs, h};

        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t assistInfoForCombineOutShapeSize = GetShapeSize(assistInfoForCombineOutShape);
        int64_t quantExpandXOutShapeSize = GetShapeSize(quantExpandXOutShape);
        int64_t commCmdInfoOutShapeSizeforcombine = GetShapeSize(commCmdInfoOutShapeforcombine);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        // 构造host数据
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int32_t> assistInfoForCombineOutHostData(assistInfoForCombineOutShapeSize, 0);
        std::vector<int8_t> quantExpandXOutHostData(quantExpandXOutShapeSize, 0);
        std::vector<int32_t> commCmdInfoOutforCombineHostData(commCmdInfoOutShapeSizeforcombine, 0);
        std::vector<float> xOutHostData(xOutShapeSize, 0);

        // 声明device地址和tensor
        void *expandXDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *assistInfoForCombineOutDeviceAddr = nullptr;
        void *quantExpandXOutDeviceAddr = nullptr;
        void *commCmdInfoOutforCombineDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;

        aclTensor *expandX = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *assistInfoForCombineOut = nullptr;
        aclTensor *quantExpandXOut = nullptr;
        aclTensor *commCmdInfoOutforCombine = nullptr;
        aclTensor *xOut = nullptr;

        // 创建tensor
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_FLOAT16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                            &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(assistInfoForCombineOutHostData, assistInfoForCombineOutShape,
                            &assistInfoForCombineOutDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombineOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(quantExpandXOutHostData, quantExpandXOutShape, &quantExpandXOutDeviceAddr,
                            aclDataType::ACL_INT8, &quantExpandXOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(commCmdInfoOutforCombineHostData, commCmdInfoOutShapeforcombine,
                            &commCmdInfoOutforCombineDeviceAddr, aclDataType::ACL_INT32, &commCmdInfoOutforCombine);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 声明算子执行必需变量
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        /******************************调用combine_setup********************************************/
        ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(
            expandX, expertIds, assistInfoForCombineOut, hcomEpName, epWorldSize, args.epRankId, moeExpertNum,
            expertSharedType, sharedExpertNum, sharedExpertRankNum, globalBS, commQuantMode, commType, nullptr,
            quantExpandXOut, commCmdInfoOutforCombine, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetupGetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineSetup aclrtMalloc failed. ret = %d\n", ret);
                    return ret);
        }

        ret = aclnnMoeDistributeCombineSetup(workspaceAddr, workspaceSize, executor, args.combinesetupstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetup failed. ret = %d \n", ret);
                return ret);

        ret = aclrtSynchronizeStreamWithTimeout(args.combinesetupstream, timeOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineSetup execute successfully.\n", args.rankId);

        /******************************调用combine_teardown********************************************/
        ret = aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
            expandX, quantExpandXOut, expertIds, expandIdx, expertScales, commCmdInfoOutforCombine, nullptr, nullptr,
            hcomEpName, epWorldSize, args.epRankId, moeExpertNum, expertSharedType, sharedExpertNum, sharedExpertRankNum,
            globalBS, commQuantMode, commType, nullptr, xOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardownGetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineTeardown aclrtMalloc failed. ret = %d\n", ret);
                    return ret);
        }

        ret = aclnnMoeDistributeCombineTeardown(workspaceAddr, workspaceSize, executor, args.combineteardownstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardown failed. ret = %d \n", ret);
                return ret);

        ret = aclrtSynchronizeStreamWithTimeout(args.combineteardownstream, timeOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineTeardown execute successfully.\n", args.rankId);

        // 释放资源
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        DestroyTensor(expandX);
        DestroyTensor(expertIds);
        DestroyTensor(expertScales);
        DestroyTensor(expandIdx);
        DestroyTensor(assistInfoForCombineOut);
        DestroyTensor(quantExpandXOut);
        DestroyTensor(commCmdInfoOutforCombine);
        DestroyTensor(xOut);

        FreeDeviceAddr(expandXDeviceAddr);
        FreeDeviceAddr(expertIdsDeviceAddr);
        FreeDeviceAddr(expertScalesDeviceAddr);
        FreeDeviceAddr(expandIdxDeviceAddr);
        FreeDeviceAddr(assistInfoForCombineOutDeviceAddr);
        FreeDeviceAddr(quantExpandXOutDeviceAddr);
        FreeDeviceAddr(commCmdInfoOutforCombineDeviceAddr);
        FreeDeviceAddr(xOutDeviceAddr);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

        aclrtStream combineSetupStream[DEV_NUM];
        aclrtStream combineTeardownStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&combineSetupStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&combineTeardownStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }

        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].epRankId = rankId;
            args[rankId].hcclEpComm = comms[rankId];
            args[rankId].combinesetupstream = combineSetupStream[rankId];
            args[rankId].combineteardownstream = combineTeardownStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcess, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
