# aclnnMoeDistributeCombineTeardown

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_teardown)

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

- **接口功能**：接收aclnnMoeDistributeCombineSetup发来的数据并整合数据（乘权重再相加）。
- **计算公式**：

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    按MoeDistributeDispatchSetup和MoeDistributeDispatchTeardown算子收集数据的路径原路返还，本算子只做通信状态确认和数据整理，对aclnnMoeDistributeCombineSetup接口发送的数据做通信状态确认，并按照专家的scale权重做加权求和。

- **注意**：该接口必须与aclnnMoeDistributeDispatchSetup、aclnnMoeDistributeDispatchTeardown及aclnnMoeDistributeCombineSetup配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用“`aclnnMoeDistributeCombineTeardownGetWorkspaceSize`”接口获取入参并根据计算流程计算所需workspace大小获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“`aclnnMoeDistributeCombineTeardown`”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* quantExpandX,
    const aclTensor* expertIds,
    const aclTensor* expandIdx,
    const aclTensor* expertScales,
    const aclTensor* commCmdInfo,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* sharedExpertXOptional,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t commQuantMode,
    int64_t commType,
    const char* commAlg,
    aclTensor* xOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineTeardown(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnMoeDistributeCombineTeardownGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1548px"><colgroup>
    <col style="width: 162px">
    <col style="width: 94px">
    <col style="width: 485px">
    <col style="width: 458px">
    <col style="width: 84px">
    <col style="width: 78px">
    <col style="width: 76px">
    <col style="width: 111px">
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
        <td>expandX（aclTensor*）</td>
        <td>输入</td>
        <td>根据expertIds进行扩展过的token特征</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantExpandX（aclTensor*）</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeCombineSetup中的quantExpandX输出</td>
        <td>不支持空Tensor。</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(A, tokenMsgSize)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds（aclTensor*）</td>
        <td>输入</td>
        <td>每个token的topK个专家索引</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expandIdx（aclTensor*）</td>
        <td>输入</td>
        <td>对应aclnnMoeDistributeDispatchSetup中的expandIdx输出</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(BS * K, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertScales（aclTensor*）</td>
        <td>输入</td>
        <td>每个token的topK个专家的权重</td>
        <td>不支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(BS, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>commCmdInfo（aclTensor*）</td>
        <td>输入</td>
        <td>aclnnMoeDistributeCombineSetup的输出，通信的cmd信息</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>((A + epWorldSize) * 16, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>xActiveMaskOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>表示token是否参与通信</td>
        <td>不支持空Tensor。可选择传入有效数据或填空指针，传空指针时表示所有token都参与通信。</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>(BS, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sharedExpertXOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>表示共享专家计算后的Token</td>
        <td>不支持空Tensor。要求是一个2D或3D的Tensor，当Tensor为2D时，shape为(BS,H)；当Tensor为3D时，前两位的乘积需等于BS，第三维需等于H。数据类型需跟expandX保持一致。可选择传入有效数据或填空指针，传入有效数据时，sharedExpertNum需为0。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(BS, H)或(a, b, H)</td>
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
        <td>sharedExpertNum</td>
        <td>输入</td>
        <td>共享专家数量</td>
        <td>当前取值范围[0, 4]。0表示无共享专家。当前仅支持无共享专家。</td>
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
        <td>取值范围[0, 2]，0表示通信时不进行量化，当前仅支持0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>输入</td>
        <td>通信方案选择</td>
        <td>取值范围[0, 2]，0表示AICPU-SDMA方案，1表示CCU方案，2表示URMA方案，当前仅支持2。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法选择</td>
        <td>仅支持传入空指针或空字符串</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>xOut（aclTensor*）</td>
        <td>输出</td>
        <td>表示处理后的token</td>
        <td>不支持空Tensor。数据类型与expandX保持一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(BS, H)</td>
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
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

  <!-- npu="950" id7 -->
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

  <!-- end id7 -->
  - Atlas A3训练系列产品/Atlas A3推理系列产品：
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

## aclnnMoeDistributeCombineTeardown

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineTeardownGetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td rowspan="2">executor</td>
        <td rowspan="2">输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>指定执行任务的stream流。</td>
    </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeDistributeCombineTeardown默认确定性实现。
- aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口必须配套使用。
- 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值所有卡需保持一致，`groupEp`、`epWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commQuantMode`、`commType`、`commAlg`参数取值在网络中不同层中也需保持一致，且和aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口对应参数也保持一致。
- 参数说明里shape格式说明：

  - A：表示本卡需要分发的最大token数量，取值范围如下：
    - 对于共享专家，要满足A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum；当globalBs非0时，要满足A = globalBs\* sharedExpertNum / sharedExpertRankNum。
    - 对于MoE专家，当globalBs为0时，要满足A >= BS \* epWorldSize \* min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs \* min(localExpertNum, K)。
  - H：表示hidden size隐藏层大小。取值为[1024, 8196]。当前仅支持4096、7168。
  - HS：表示通信时的数据大小，Align512( Align32(H) + Align8(H) / 8 *sizeof(float) )，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32，Align8(x) = ((x + 8 - 1) / 8) \* 8。
  - BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0 < BS ≤ 512。当前仅支持8、16、256。
  - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。当前仅支持6、8。
  - localExpertNum：表示本卡专家数量。
    - 对于共享专家卡，localExpertNum = 1
    - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)。moeExpertNum当前仅支持32。
  - tokenMsgSize：表示每个token在数据通信时的维度信息，计算公式为Align512(Align32(H)+Align8(H)/8\*sizeof(float))，其中AlignN(x)=((x+N-1)/N*N)。
  - 当前不支持共享专家。sharedExpertNum和sharedExpertRankNum当前仅支持0。
- HCCL_BUFFSIZE：
  调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。

  <!-- npu="950" id8 -->
  - Ascend 950DT：
    - 要求 >= 2且满足>= 4 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32
  <!-- end id8 -->
  - Atlas A3训练系列产品/Atlas A3推理系列产品：
    - 要求 >= 2且满足>= 2 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32
- 通信域使用约束：
  - 一个模型中的aclnnMoeDistributeDispatchSetup接口，aclnnMoeDistributeDispatchTeardown接口，aclnnMoeDistributeCombineSetup接口，aclnnMoeDistributeCombineTeardown接口仅支持相同EP通信域，且该通信域中不允许有其他算子。
- 通信方式约束：

    <!-- npu="950" id9 -->
    - <term>Ascend 950DT</term>：仅支持URMA通信。

    <!-- end id9 -->

## 调用示例

<!-- npu="950" id10 -->
- Ascend 950DT：aclnnMoeDistributeCombineTeardown必须在aclnnMoeDistributeCombineSetup调用之后调用，请参考[aclnnMoeDistributeCombineSetup](../../moe_distribute_combine_setup/docs/aclnnMoeDistributeCombineSetup.md)中的调用示例。

<!-- end id10 -->
