# aclnnMoeDistributeDispatchSetup

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_setup)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口说明：
    - 对Token数据进行量化（可选），根据token选择的topK专家在EP（Expert Parallelism）域的AllToAllV通信，只进行数据发送和通信状态发送，通信指令发出后算子即刻退出，无需等待通信完成。数据的接收和后处理由aclnnMoeDistributeDispatchTeardown接口完成。

    - 注意该接口必须与aclnnMoeDistributeDispatchTeardown，aclnnMoeDistributeCombineSetup，aclnnMoeDistributeCombineTeardown配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMoeDistributeDispatchSetupGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatchSetup”接口执行计算。为用户提供“aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize”接口计算“aclnnMoeDistributeDispatchSetup”
部分输出的size大小。

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          commType,
    const char*      commAlg,
    aclTensor*       yOut,
    aclTensor*       expandIdxOut,
    aclTensor*       commCmdInfoOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetup(
    void             *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream      stream)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          expertTokenNumsType,
    int64_t          commType,
    const char*      commAlg,
    uint64_t&        tokenMsgSize,
    uint64_t&        expandIdxOutSize,
    uint64_t&        assistInfoForCombineOutSize,
    uint64_t&        commCmdInfoOutSize)
```

## aclnnMoeDistributeDispatchSetupGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px">
    <col style="width: 180px">
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
    </tr>
    </thead>
    <tbody>

    <tr>
    <td>x</td>
    <td>输入</td>
    <td>表示本卡发送的token数据。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求为2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>输入</td>
    <td>每个专家的量化平滑参数。</td>
    <td>要求为2D Tensor。非量化场景传空指针，动态量化可选择传入有效数据或传入空指针。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(sharedExpertNum + moeExpertNum, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信。</td>
    <td>要求为1D Tensor。可选择传入有效数据或传入空指针，传入空指针时是表示所有token都会参与通信。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>(Bs, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行通信域）。</td>
    <td>字符串长度范围为[1, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP域本卡Id。</td>
    <td>取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>取值范围(0, 512]。满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>表示共享专家卡分布类型。</td>
    <td>当前仅支持传入0，表示共享专家卡排在MoE专家卡前面。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>表示共享专家数量。</td>
    <td>取值范围[0, 4]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>表示共享专家卡数量。</td>
    <td>取值范围[0, epWorldSize / 2]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>表示量化模式。</td>
    <td>取值范围[0, 4]。0表示非量化，1表示静态量化，2表示Pertoken动态量化，3表示Pergroup动态量化，4表示MX量化，当前仅支持0和4。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局的batch size大小。</td>
    <td><ul><li>各rank Bs一致时，globalBs = Bs * epWorldSize或0。</li><li>各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>输入</td>
    <td>表示通信方案选择。</td>
    <td>取值范围[0, 2]，0表示AICPU-SDMA方案，1表示CCU方案，2表示URMA方案，当前版本仅支持2。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>表示通信亲和内存布局算法。</td>
    <td>预留字段，当前版本不支持，传空指针或空字符串即可。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yOut</td>
    <td>输出</td>
    <td>表示本卡待发送的通信数据，通信数据对输入token数据做了算法重排；如需量化，先将输入token做量化处理，再对数据做重排。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、INT8、HiFP8、FP8E5M2、FP8E4M3</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum), tokenMsgSize)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdxOut</td>
    <td>输出</td>
    <td>表示给同一专家发送的token个数，对应Combine系列算子中的expandIdx。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * K, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commCmdInfoOut</td>
    <td>输出</td>
    <td>通信的cmd信息</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
    <td>√</td>
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

    <!-- npu="950" id7 -->
    - <term>Ascend 950DT</term>：
        - scalesOptional非量化场景传空指针，动态量化可选择传入有效数据或传入空指针。
        - xActiveMaskOptional可选择传入有效数据或传入空指针，传入空指针时表示所有token都会参与通信。
        - groupEp字符串长度范围为[1, 128)。
        - epWorldSize取值范围[2, 384]。当前仅支持2、8。
        - epRankId取值范围[0, epWorldSize)。同一个EP通信域中各卡的epRankId不能重复。
        - moeExpertNum取值范围(0, 512]。
        - expertShardType当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - sharedExpertNum当前取值范围[0, 4]。
        - sharedExpertRankNum取值范围[0, epWorldSize / 2]。
        - globalBs当每个rank的Bs数一致场景下，globalBs = Bs *epWorldSize或globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs* epWorldSize，其中maxBs表示单卡Bs最大值。
        - commType当前仅支持2。
        - commAlg当前版本不支持，传空指针即可。

    <!-- end id7 -->

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>参数的取值不在支持的范围内。</td>
  </tr>
  </tbody>
  </table>

## aclnnMoeDistributeDispatchSetup

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeDispatchSetupGetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px">
    <col style="width: 180px">
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
    </tr>
    </thead>
    <tbody>

    <tr>
    <td>x</td>
    <td>输入</td>
    <td>表示本卡发送的token数据。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求为2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>输入</td>
    <td>每个专家的量化平滑参数。</td>
    <td>要求为2D Tensor。非量化场景传空指针，动态量化可选择传入有效数据或传入空指针。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(sharedExpertNum + moeExpertNum, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信。</td>
    <td>要求为1D Tensor。可选择传入有效数据或传入空指针，传入空指针时是表示所有token都会参与通信。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>(Bs, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行通信域）。</td>
    <td>字符串长度范围为[1, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP域本卡Id。</td>
    <td>取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>取值范围(0, 512]。满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>表示共享专家卡分布类型。</td>
    <td>当前仅支持传入0，表示共享专家卡排在MoE专家卡前面。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>表示共享专家数量。</td>
    <td>取值范围[0, 4]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>表示共享专家卡数量。</td>
    <td>取值范围[0, epWorldSize / 2]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>表示量化模式。</td>
    <td>取值范围[0, 4]。0表示非量化，1表示静态量化，2表示Pertoken动态量化，3表示Pergroup动态量化，4表示MX量化，当前仅支持0和4。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>EP域全局的batch size大小。</td>
    <td><ul><li>各rank Bs一致时，globalBs = Bs * epWorldSize或0。</li><li>各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>输入</td>
    <td>输出expertTokenNums中值的语义类型。</td>
    <td><ul><li>取值为0：expertTokenNums中的输出为每个专家处理的token数的前缀和。</li><li>取值为1：expertTokenNums中的输出为每个专家处理的token数量。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>输入</td>
    <td>表示通信方案选择。</td>
    <td>取值范围[0, 2]，0表示AICPU-SDMA方案，1表示CCU方案，2表示URMA方案，当前版本仅支持2。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>表示通信亲和内存布局算法。</td>
    <td>预留字段，当前版本不支持，传空指针或空字符串即可。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenMsgSize</td>
    <td>输出</td>
    <td>aclnnMoeDistributeDispatchSetup接口yOut第二维输出，表示每个token在数据通信时的维度信息。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>非量化场景下，tokenMsgSize = Align256(H)。量化场景下，quantmode = 1时，tokenMsgSize = Align512(H)；quantmode = 2时，tokenMsgSize = Align512(Align32(H) + 4)；quantmode = 3时，tokenMsgSize = Align512(Align128(H) + ceilDiv(H，128))；quantmode = 4时，tokenMsgSize = Align512(Align128(H) + Align2(ceilDiv(H，32)))，其中AlignN(x) = ((x + N - 1) / N) * N, ceilDiv(n,m) = ((n + m - 1) / m)。</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandIdxOutSize</td>
    <td>输出</td>
    <td>aclnnMoeDistributeDispatchSetup接口的expandIdxOut的空间大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(BS * K)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>assisInfoForCombineOutSize</td>
    <td>输出</td>
    <td>aclnnMoeDistributeDispatchSetup接口的assisInfoForCombineOut的空间大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(A * 128)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commCmdInfoOutSize</td>
    <td>输出</td>
    <td>aclnnMoeDistributeDispatchSetup接口的commCmdInfoOut的大小。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

1. 确定性计算：
     - aclnnMoeDistributeDispatchSetup默认确定性实现。

2. aclnnMoeDistributeDispatchSetup接口与aclnnMoeDistributeDispatchTeardown，aclnnMoeDistributeCombineSetup，aclnnMoeDistributeCombineTeardown接口必须配套使用。

3. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值所有卡需保持一致，`groupEp`、`epWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值在网络中不同层中也需保持一致，且和`aclnnMoeDistributeDispatchTeardown`，`aclnnMoeDistributeCombineSetup`，`aclnnMoeDistributeCombineTeardown`对应参数也保持一致。

4. 参数说明里shape格式说明：
    * A：表示本卡可能接收的最大token数量，取值范围如下：

      * 对于MoE专家，当`globalBs`为0时，要满足A >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足A >= `globalBs` * min(`localExpertNum`, `K`)。
      * 对于共享专家，当`globalBs`为0时，要满足A = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum`；当`globalBs`非0时，要满足A = `globalBs` * `sharedExpertNum` / `sharedExpertRankNum`。
    * H：表示hidden size隐藏层大小，取值范围[1024, 8192]。当前仅支持4096、7168。
    * BS：表示batch sequence size，即本卡最终输出的token数量，取值范围为0 < BS ≤ 512。当前仅支持8、16、256。
    * K：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moeExpertNum`。当前仅支持6、8。
    * localExpertNum：表示本卡专家数量。

      * 对于共享专家卡，localExpertNum = 1
      * 对于MoE专家卡，localExpertNum = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)。moeExpertNum当前仅支持32。
    * tokenMsgSize：表示每个token在数据通信时的维度信息。
      * 非量化场景下，tokenMsgSize = Align256(H)。
      * 量化场景下，tokenMsgSize = Align512(Align32(H) + 4 )，其中AlignN(x) = ((x + N - 1) / N) * N。

    * 当前版本暂不支持共享专家。sharedExpertNum和sharedExpertRankNum当前仅支持0。

5. HCCL_BUFFSIZE：

    调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 4 *(`localExpertNum`* `maxBs` *`epWorldSize`* Align512(Align32(2 *H) + 44) + (`K` + `sharedExpertNum`)* `maxBs` *Align512(2* `H`))，`localExpertNum`代表使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) *512，Align32(x) = ((x + 32 - 1) / 32)* 32。

6. 通信域使用约束：
    * 一个模型中的aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口仅支持相同EP通信域，且该通信域中不允许有其他算子。

7. 通信方式约束：

  <!-- npu="950" id8 -->
  - <term>Ascend 950DT</term>：仅支持URMA通信。

  <!-- end id8 -->

## 调用示例

- 文件准备：

    1. 按照下方指导创建rank_table_m2.json文件并修改。

    2. 将项目拷贝到两台服务器中，并根据机器的device ip配置rank_table_m2.json文件内容。注意两机rank_table_m2.json文件保持一致。

    3. 安装cann包，并根据[算子调用](../../../docs/zh/invocation/quick_op_invocation.md)编译运行。

- 关于rankTable:

    1. 开发者可以通过ranktable文件配置参与集合通信的NPU资源信息，详细配置请参考[《集合通信用户指南》](https://hiascend.com/document/redirect/CannCommunityHcclUg)中“通信功能开发>集群信息配置>ranktable文件配置资源信息”。

    2. 使用`cat /etc/hccn.conf`或者`for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done`查询机器的device ip。然后参考集合通信文档填写json文件。

    > 注意：两机16卡场景中，两机器的device_id都是0~7，其中一台机器的rank_id为0~7，另一台机器的rank_id为8~15。单机16卡场景中，device_id和rank_id都是0~15。

<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：

    示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

    ```cpp
    #include <cstdint>
    #include <cstdio>
    #include <iostream>
    #include <thread>
    #include <vector>

    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_setup.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_teardown.h"

    #define CHECK_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                return_expr; \
            } \
        } while (0)

    #define LOG_PRINT(message, ...) \
        do { \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    constexpr int32_t DEV_NUM = 2;
    constexpr int64_t BS = 8;
    constexpr int64_t H = 1024;
    constexpr int64_t K = 1;
    constexpr int64_t MOE_EXPERT_NUM = 2;
    constexpr int64_t EP_WORLD_SIZE = DEV_NUM;
    constexpr int64_t GLOBAL_BS = BS * EP_WORLD_SIZE;
    constexpr int64_t LOCAL_EXPERT_NUM = MOE_EXPERT_NUM / EP_WORLD_SIZE;
    constexpr int64_t LOCAL_TOKEN_NUM = GLOBAL_BS * ((LOCAL_EXPERT_NUM < K) ? LOCAL_EXPERT_NUM : K);
    constexpr int64_t EXPERT_SHARD_TYPE = 0;
    constexpr int64_t SHARED_EXPERT_NUM = 0;
    constexpr int64_t SHARED_EXPERT_RANK_NUM = 0;
    constexpr int64_t QUANT_MODE = 0;
    constexpr int64_t EXPERT_TOKEN_NUMS_TYPE = 1;
    constexpr int64_t COMM_TYPE = 2;
    constexpr int64_t TIMEOUT = 100000000;

    template <typename Func>
    class Guard {
    public:
        explicit Guard(Func &func)
            : func_(func)
        {}

        ~Guard()
        {
            func_();
        }

    private:
        Func &func_;
    };

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtContext context;
        aclrtStream setupStream;
        aclrtStream teardownStream;
    };

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t size = 1;
        for (int64_t dim : shape) {
            size *= dim;
        }
        return size;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                        void **deviceAddr, aclTensor **tensor)
    {
        const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
        int ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret = %d\n", ret); return ret);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        CHECK_RET(*tensor != nullptr, LOG_PRINT("[ERROR] aclCreateTensor failed.\n"); return -1);
        return ACL_SUCCESS;
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
        std::cout << "[INFO] device_" << args.rankId << " worker start." << std::endl;
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u aclrtSetCurrentContext failed. ret = %d\n", args.rankId, ret);
                return ret);

        char groupEp[128] = {0};
        ret = HcclGetCommName(args.hcclComm, groupEp);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u HcclGetCommName failed. ret = %d\n", args.rankId, ret);
                return ret);

        auto runtimeCleanup = [&args]() {
            HcclCommDestroy(args.hcclComm);
            aclrtDestroyStream(args.setupStream);
            aclrtDestroyStream(args.teardownStream);
            aclrtDestroyContext(args.context);
            aclrtResetDevice(args.rankId);
        };
        auto runtimeGuard = Guard<decltype(runtimeCleanup)>(runtimeCleanup);

        // setup input/output shapes
        const std::vector<int64_t> xShape{BS, H};
        const std::vector<int64_t> expertIdsShape{BS, K};
        const std::vector<int64_t> yShape{BS * K, H};
        const std::vector<int64_t> expandIdxShape{BS * K};
        const std::vector<int64_t> commCmdInfoShape{(BS * K + EP_WORLD_SIZE * LOCAL_EXPERT_NUM) * 16};

        // teardown output shapes
        const std::vector<int64_t> expandXShape{LOCAL_TOKEN_NUM, H};
        // The current teardown ACLNN wrapper requires a non-null tensor here even when quantMode is 0.
        const std::vector<int64_t> dynamicScalesShape{1};
        const std::vector<int64_t> assistInfoShape{LOCAL_TOKEN_NUM * 128};
        const std::vector<int64_t> expertTokenNumsShape{LOCAL_EXPERT_NUM};

        std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
        std::vector<int32_t> expertIdsHostData(GetShapeSize(expertIdsShape));
        for (int64_t i = 0; i < BS; ++i) {
            expertIdsHostData[i] = static_cast<int32_t>(i % MOE_EXPERT_NUM);
        }
        std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);
        std::vector<int32_t> expandIdxHostData(GetShapeSize(expandIdxShape), 0);
        std::vector<int32_t> commCmdInfoHostData(GetShapeSize(commCmdInfoShape), 0);
        std::vector<int16_t> expandXHostData(GetShapeSize(expandXShape), 0);
        std::vector<float> dynamicScalesHostData(GetShapeSize(dynamicScalesShape), 0);
        std::vector<int32_t> assistInfoHostData(GetShapeSize(assistInfoShape), 0);
        std::vector<int64_t> expertTokenNumsHostData(GetShapeSize(expertTokenNumsShape), 0);

        void *xAddr = nullptr;
        void *expertIdsAddr = nullptr;
        void *yAddr = nullptr;
        void *expandIdxAddr = nullptr;
        void *commCmdInfoAddr = nullptr;
        void *expandXAddr = nullptr;
        void *dynamicScalesAddr = nullptr;
        void *assistInfoAddr = nullptr;
        void *expertTokenNumsAddr = nullptr;
        void *setupWorkspace = nullptr;
        void *teardownWorkspace = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *y = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *commCmdInfo = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *assistInfo = nullptr;
        aclTensor *expertTokenNums = nullptr;

        auto tensorCleanup = [&]() {
            DestroyTensor(x);
            DestroyTensor(expertIds);
            DestroyTensor(y);
            DestroyTensor(expandIdx);
            DestroyTensor(commCmdInfo);
            DestroyTensor(expandX);
            DestroyTensor(dynamicScales);
            DestroyTensor(assistInfo);
            DestroyTensor(expertTokenNums);
            FreeDeviceAddr(xAddr);
            FreeDeviceAddr(expertIdsAddr);
            FreeDeviceAddr(yAddr);
            FreeDeviceAddr(expandIdxAddr);
            FreeDeviceAddr(commCmdInfoAddr);
            FreeDeviceAddr(expandXAddr);
            FreeDeviceAddr(dynamicScalesAddr);
            FreeDeviceAddr(assistInfoAddr);
            FreeDeviceAddr(expertTokenNumsAddr);
            FreeDeviceAddr(setupWorkspace);
            FreeDeviceAddr(teardownWorkspace);
        };
        auto tensorGuard = Guard<decltype(tensorCleanup)>(tensorCleanup);

        ret = CreateAclTensor(xHostData, xShape, aclDataType::ACL_FLOAT16, &xAddr, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, aclDataType::ACL_INT32, &expertIdsAddr, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yHostData, yShape, aclDataType::ACL_FLOAT16, &yAddr, &y);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, aclDataType::ACL_INT32, &expandIdxAddr, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret =
            CreateAclTensor(commCmdInfoHostData, commCmdInfoShape, aclDataType::ACL_INT32, &commCmdInfoAddr, &commCmdInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, aclDataType::ACL_FLOAT16, &expandXAddr, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, aclDataType::ACL_FLOAT, &dynamicScalesAddr,
                            &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(assistInfoHostData, assistInfoShape, aclDataType::ACL_INT32, &assistInfoAddr, &assistInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, aclDataType::ACL_INT64, &expertTokenNumsAddr,
                            &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t setupWorkspaceSize = 0;
        aclOpExecutor *setupExecutor = nullptr;
        ret = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
            x, expertIds, nullptr, nullptr, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
            SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, COMM_TYPE, nullptr, y, expandIdx, commCmdInfo,
            &setupWorkspaceSize, &setupExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u DispatchSetupGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
                return ret);

        if (setupWorkspaceSize > 0) {
            ret = aclrtMalloc(&setupWorkspace, setupWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u setup workspace malloc failed. ret = %d\n", args.rankId, ret);
                    return ret);
        }
        ret = aclnnMoeDistributeDispatchSetup(setupWorkspace, setupWorkspaceSize, setupExecutor, args.setupStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchSetup failed. ret = %d\n", args.rankId, ret);
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.setupStream, TIMEOUT);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u setup synchronize failed. ret = %d\n", args.rankId, ret);
                return ret);
        LOG_PRINT("[INFO] device_%u dispatch setup success.\n", args.rankId);

        uint64_t teardownWorkspaceSize = 0;
        aclOpExecutor *teardownExecutor = nullptr;
        ret = aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
            x, y, expertIds, commCmdInfo, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
            SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, EXPERT_TOKEN_NUMS_TYPE, COMM_TYPE, nullptr,
            expandX, dynamicScales, assistInfo, expertTokenNums, &teardownWorkspaceSize, &teardownExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u DispatchTeardownGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
                return ret);

        if (teardownWorkspaceSize > 0) {
            ret = aclrtMalloc(&teardownWorkspace, teardownWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u teardown workspace malloc failed. ret = %d\n", args.rankId, ret);
                    return ret);
        }
        ret = aclnnMoeDistributeDispatchTeardown(teardownWorkspace, teardownWorkspaceSize, teardownExecutor,
                                                args.teardownStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchTeardown failed. ret = %d\n", args.rankId, ret);
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.teardownStream, TIMEOUT);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u teardown synchronize failed. ret = %d\n", args.rankId, ret);
                return ret);
        LOG_PRINT("[INFO] device_%u dispatch teardown success.\n", args.rankId);
        return ACL_SUCCESS;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtContext contexts[DEV_NUM];
        aclrtStream setupStreams[DEV_NUM];
        aclrtStream teardownStreams[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&contexts[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&setupStreams[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&teardownStreams[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devices[DEV_NUM] = {0, 1};
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

        Args args[DEV_NUM];
        int results[DEV_NUM] = {ACL_SUCCESS, ACL_SUCCESS};
        std::thread threads[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            args[rankId] = {rankId, comms[rankId], contexts[rankId], setupStreams[rankId], teardownStreams[rankId]};
            threads[rankId] =
                std::thread([&args, &results, rankId]() { results[rankId] = LaunchOneProcess(args[rankId]); });
        }
        for (auto &thread : threads) {
            thread.join();
        }

        ret = aclFinalize();
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclFinalize failed. ret = %d\n", ret); return ret);
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            CHECK_RET(results[rankId] == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u failed. ret = %d\n", rankId, results[rankId]);
                    return results[rankId]);
        }
        return ACL_SUCCESS;
    }
    ```

<!-- end id9 -->
