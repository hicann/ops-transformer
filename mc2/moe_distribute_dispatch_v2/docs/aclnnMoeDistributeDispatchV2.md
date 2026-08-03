# aclnnMoeDistributeDispatchV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：对token数据进行量化（可选），进行EP（Expert Parallelism）域的AllToAllV通信。TP（Tensor Parallelism）域通信当前版本不再支持，TP相关参数为预留参数。

    相较于`aclnnMoeDistributeDispatch`接口，该接口变更如下：
    1. 输出了更详细的token信息辅助CombineV2系列算子高效地进行全卡同步，因此原接口中shape为`(BS * K,)`的`expandIdx`出参替换为shape为`(A * 128,)`的`assistInfoForCombineOut`参数；
    2. 新增`commAlg`入参，代替用于控制分层算法的`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`环境变量。

    详细说明请参考以下参数说明。
- 计算公式：

    - 情形1：如果quantMode=0（非量化场景）：

    $$
    allToAllXOut = AllToAllV(X)\\
    expandXOut = AllToAllV(X)
    $$

    当`x`的数据类型为`INT32`时，`scalesOptional`随对应token进行通信：

    $$
    allToAllDynamicScalesOut = AllToAllV(scalesOptional)\\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - 情形2：如果quantMode=1（静态量化场景）：

    $$
    xFp32 = CastToFp32(X) \times scales \\
    quantOut = Cast(xFp32，dstType) \\
    allToAllXOut = AllToAllV(quantOut)\\
    expandXOut = AllToAllV(quantOut)
    $$

    - 情形3：如果quantMode=2（pertoken动态量化场景）：

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - 情形4：如果quantMode=3（pertile动态量化场景）：

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - 情形5：如果quantMode=4（mx量化场景）：

    $$
    sharedExp = Floor(log_2(max(x))) - emax \\
    dynamicScales = 2^{sharedExp} \\
    quantOut = CastToFp8(X / dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0 / dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    其中，$emax$表示该类型最大正规数对应的指数部分的值。

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该接口必须与`aclnnMoeDistributeCombineV2`配套使用。
<!-- end id7 -->
<!-- npu="950,A3" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：该接口必须与`aclnnMoeDistributeCombineV2`或`aclnnMoeDistributeCombineAddRmsNorm`配套使用。

<!-- end id8 -->

> 说明：
> `aclnnMoeDistributeCombineV2`、`aclnnMoeDistributeCombineAddRmsNorm`算子在后续文档中统称为**CombineV2系列算子**。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMoeDistributeDispatchV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatchV2”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBS,
    int64_t          expertTokenNumsType,
    const char*      commAlg,
    aclTensor*       expandXOut,
    aclTensor*       dynamicScalesOut,
    aclTensor*       assistInfoForCombineOut,
    aclTensor*       expertTokenNumsOut,
    aclTensor*       epRecvCountsOut,
    aclTensor*       tpRecvCountsOut,
    aclTensor*       expandScalesOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchV2(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV2GetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 350px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>要求2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2、INT32</td>
    <td>ND</td>
    <td><code>(BS, H)</code>（BS=batch size，H=hidden size）</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>输入</td>
    <td>每个token的topK个专家索引。</td>
    <td>要求2D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>输入</td>
    <td>每个专家的量化平滑参数或INT32 token的缩放参数。</td>
    <td>-</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>输入</td>
    <td>表示token是否参与通信。</td>
    <td>可选择传入有效数据或传入空指针。<br>当输入为1D时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；<br>当输入为2D时，参数为true表示当前token对应的expert_ids参与通信。若当前token对应的K个BOOL值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>当输入1D时，shape为 <code>(BS,)</code>；当输入2D时，shape为 <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>输入</td>
    <td>每个Token的topK个专家权重。</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称（专家并行通信域）。</td>
    <td>字符串长度范围为<code>[1, 128)</code>。</td>
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
    <td>EP域本卡ID。</td>
    <td><ul><li>取值范围<code>[0, epWorldSize)</code></li><li>同一个EP通信域中各卡的<code>epRankId</code>不重复。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>需要满足<code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>输入</td>
    <td>TP通信域名称（预留参数，当前版本不支持）。</td>
    <td>默认值为""。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>TP通信域大小（预留参数，当前版本不支持，仅支持传0或1）。</td>
    <td>默认值为0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>输入</td>
    <td>TP域本卡ID（预留参数，当前版本不支持，仅支持传0）。</td>
    <td>默认值为0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>输入</td>
    <td>共享专家卡分布类型。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>输入</td>
    <td>共享专家数量（一个共享专家可复制部署到多个卡上）。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>输入</td>
    <td>共享专家卡数量。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>量化模式。</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBS</td>
    <td>输入</td>
    <td>EP域全局batch size。</td>
    <td><ul><li>各卡BS一致时，<code>globalBS = BS * epWorldSize</code> 或0。</li><li>各卡BS不一致时，<code>globalBS = maxBS * epWorldSize</code>（maxBS为单卡BS最大值）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>输入</td>
    <td>输出<code>expertTokenNums</code>中值的语义类型。</td>
    <td>支持0：<code>expertTokenNums</code>中的输出为每个专家处理的token数的前缀和，1：<code>expertTokenNums</code>中的输出为每个专家处理的token数量。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>通信亲和内存布局算法。</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandXOut</td>
    <td>输出</td>
    <td>根据<code>expertIds</code>扩展过的token特征。</td>
    <td>要求为2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2、INT32</td>
    <td>ND</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>输出</td>
    <td>缩放参数。</td>
    <td>要求为1D或2D Tensor。quantMode取值为2、3、4时有输出；quantMode取值为0且`x`的数据类型为`HIFLOAT8`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN`、`FLOAT4_E2M1`、`FLOAT4_E1M2`、`INT32`时也有输出。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>输出</td>
    <td>给同一专家发送的token个数（对应<code>aclnnMoeDistributeCombineV2</code>中的<code>assistInfoForCombine</code>）。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(A*128, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>输出</td>
    <td>每个专家收到的token个数。</td>
    <td>要求为1D Tensor。</td>
    <td>INT64</td>
    <td>ND</td>
    <td><code>(localExpertNum, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>输出</td>
    <td>从EP通信域各卡接收的token数（对应<code>aclnnMoeDistributeCombineV2</code>中的<code>epSendCounts</code>）。</td>
    <td>要求为1D Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>输出</td>
    <td>从TP通信域各卡接收的token数（预留输出，当前版本不支持，传空指针即可）。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>输出</td>
    <td>表示本卡输出token的权重（对应<code>aclnnMoeDistributeCombineV2</code>中的<code>expertScalesOptional</code>）。</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
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
    </tr>
    </tbody>
    </table>

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - `dynamicScalesOut`在quantMode取值为2或`x`的数据类型为INT32时有输出。
        - `commAlg`支持nullptr、""、"fullmesh"、"hierarchy"；推荐配置"hierarchy"并搭配≥25.0.RC1.1版本驱动；nullptr和""依HCCL环境变量选择算法（不推荐）；"fullmesh"通过RDMA直传token；"hierarchy"经跨机、机内两次发送优化通信。
        - `commAlg`为"hierarchy"或HCCL_INTRA_PCIE_ENABLE=1且HCCL_INTRA_ROCE_ENABLE=0时，scalesOptional需传nullptr。`x`的数据类型为INT32时，仅支持`commAlg`配置为"fullmesh"，且scalesOptional必须传入有效数据。
        - `xActiveMaskOptional`依commAlg取值，"fullmesh"要求为1D Tensor，shape为(BS, )；true需排在false前（例：{true, false, true}非法）；"hierarchy"当前版本不支持，传空指针即可。
        - `expertScalesOptional`要求为2D Tensor，shape为(BS, K)。
        - `epWorldSize`依commAlg取值，"fullmesh"支持2、3、4、5、6、7、8、16、32、64、128、192、256、384；"hierarchy"支持16、32、64。
        - `moeExpertNum`依commAlg取值，"fullmesh"支持(0, 1024]，"hierarchy"支持(0, 512]。
        - `groupTp`当前版本不支持，传空字符即可。
        - `tpWorldSize`、`tpRankId`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`当前版本不支持，传0即可。
        - `epRecvCountsOut`的shape为(moeExpertNum + 2 \* globalBS \* K \* serverNum,)（前moeExpertNum个为接收token数，剩余为通信前reduce相关信息）。
        - 当前不支持TP域通信。
        - `expandScalesOut`要求为1D Tensor，shape为(A,)。
        - `quantMode`支持0（非量化）、2（动态量化）。

    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
        - `dynamicScalesOut`仅quantMode取值为2时有输出。
        - `commAlg`当前版本不支持，传空指针即可。
        - `xActiveMaskOptional`要求为1D或2D Tensor（1D时shape为(BS, )，2D时shape为(BS, K)）；1D时true需排在false前，2D时token对应K个值全为false则不参与通信。
        - `expertScalesOptional`当前版本不支持，传空指针即可。
        - `epWorldSize`取值范围[2, 768]。
        - `moeExpertNum`取值范围(0, 1024]。
        - `groupTp`（预留参数，当前版本不支持）。
        - `tpWorldSize`（预留参数，当前版本不支持，仅支持传0或1）。
        - `tpRankId`（预留参数，当前版本不支持，仅支持传0）。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `sharedExpertNum`当前取值范围[0, 4]。
        - `sharedExpertRankNum`取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
        - `epRecvCountsOut`的shape为(epWorldSize \* localExpertNum,)。
        - `tpRecvCountsOut`（预留输出，当前版本不支持，传空指针即可）。
        - `expandScalesOut`当前版本不支持该输出。
        - `quantMode`支持0（非量化）、2（动态量化）。

    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950DT</term>：
        - `dynamicScalesOut`quantMode取值为2、3、4时有输出；quantMode取值为0且`x`的数据类型为`HIFLOAT8`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN`、`FLOAT4_E2M1`、`FLOAT4_E1M2`时也有输出。
        - `commAlg`当前版本不支持，传空指针即可。
        - `xActiveMaskOptional`要求为1D或2D Tensor（1D时shape为(BS, )，2D时shape为(BS, K)）；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信。
        - `expertScalesOptional`可传有效Tensor或空指针。传有效Tensor时使能expertScales专家权重功能，有效Tensor为2D Tensor，shape为(BS, K)；传空指针时不使能expertScales专家权重功能。不支持空Tensor。
        - `epWorldSize`取值范围[2, 1024]。
        - `moeExpertNum`取值范围(0, 1024]。
        - `groupTp`当前版本不支持，传空字符即可。
        - `tpWorldSize`当前版本不支持，传0即可。
        - `tpRankId`当前版本不支持，传0即可。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `sharedExpertNum`当前取值范围[0, 4]。
        - `sharedExpertRankNum`取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
        - `epRecvCountsOut`的shape为(epWorldSize \* localExpertNum,)。
        - `tpRecvCountsOut`当前版本不支持该输出。
        - `expandScalesOut`当使能expertScales专家权重功能时，输出有效；不使能时，该输出无效。
        - `quantMode`支持0（非量化）、1（静态量化）、2（pertoken动态量化）、3（pergroup动态量化）、4（mx动态量化）。

    <!-- end id11 -->

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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

## aclnnMoeDistributeDispatchV2

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeDispatchV2GetWorkspaceSize</code>获取。</td>
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

## 约束说明

- **确定性计算**：
  - aclnnMoeDistributeDispatchV2默认确定性实现。

- **驱动约束**：
  - 算子通信域各节点的驱动版本应当相同。

- **接口配套约束**：
  - `aclnnMoeDistributeDispatchV2`接口与CombineV2系列算子接口必须配套使用，具体参考[调用示例](#调用示例)。
  - 在不同产品型号、不同通信算法或不同版本中，`aclnnMoeDistributeDispatchV2`的Tensor输出`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`中的元素值可能不同，使用时直接将上述Tensor传给CombineV2系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

- **参数一致性约束**：
  - 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBS`、`commAlg`参数及`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和CombineV2系列算子对应参数也保持一致。

- **产品特定约束**：

  <!-- npu="A3" id12 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
  <!-- end id12 -->

- **Shape变量约束**：

  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | 表示本卡需要分发的最大token数量，取值范围如下：<ul> <li>对于共享专家，要满足A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum。</li> <li>对于MoE专家，当globalBS为0时，要满足A >= BS \* epWorldSize \* min(localExpertNum, K)；当globalBS非0时，要满足A >= globalBS \* min(localExpertNum, K)。</li> </ul>|
  | H（hidden size） | 表示hidden size隐藏层大小。<ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当`x`的数据类型为INT32时，取值范围为(0, 5120]且为32的整数倍；其他数据类型的取值范围为(0, 10240]且为32的整数倍。</li> <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：取值范围[1024, 8192]。</li> </ul> |
  | BS           | 表示本卡最终输出token数。<ul><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：依commAlg取值，"fullmesh"取值范围为(0 < BS ≤ 256)；"hierarchy"并且驱动版本≥25.0.RC1.1时取值范围为(0 < BS ≤ 512)；</li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：0 < BS ≤ 512。 </li> </ul> |
  | K    | 表示选取topK个专家，取值范围为(0 < K ≤ 16)且满足(0 < K ≤ moeExpertNum)。|
  | serverNum    | 表示服务器节点数，仅支持2、4、8。<br>Atlas A2训练系列产品/Atlas A2推理系列产品：仅该场景的shape使用了该变量。                                                  |
  | localExpertNum |  本卡专家数：<ul><li>对于共享专家卡，localExpertNum = 1；</li><li>对于MoE专家卡，localExpertNum = <code>moeExpertNum / (epWorldSize - sharedExpertRankNum)</code>，当前版本不支持TP域通信。 </li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：应满足0 < localExpertNum * epWorldSize ≤ 2048。</li></ul>|

- **quantMode相关约束**：

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
      - `quantMode`取值为0时，表示非量化场景：
          - 当`x`的数据类型为`FLOAT16`或`BFLOAT16`时，输入`scalesOptional`传空指针，`expandX`的数据类型与`x`一致。
          - 当`x`的数据类型为`INT32`时，仅支持`commAlg`配置为"fullmesh"，输入`scalesOptional`必须为1D Tensor，数据类型为`FLOAT32`，shape为(`BS`, )；`expandX`的数据类型为`INT32`，输出`dynamicScalesOut`的数据类型为`FLOAT32`，shape为(`A`, )。
      - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
          - 输入`scalesOptional`可传入空指针。
          - 若输入`scalesOptional`传入有效数据时，其shape为(`moeExpertNum`, `H`)。
          - 输出`dynamicScalesOut`shape为`(A, )`
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
      - `quantMode`取值为0时，表示非量化场景，输入`scalesOptional`传空指针，`expandX`的数据类型支持`FLOAT16`、`BFLOAT16`。
      - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
          - 输入`scalesOptional`可传入空指针。
          - 若输入`scalesOptional`传入有效数据且存在共享专家卡时，其shape为(`sharedExpertNum` + `moeExpertNum`, `H`)。
          - 若输入`scalesOptional`传入有效数据且不存在共享专家卡时，其shape为(`moeExpertNum`, `H`)。
          - 输出`dynamicScalesOut`shape为`(A, )`
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950DT</term>：
      - `quantMode`取值为0时，表示非量化场景。
          - 当`x`的数据类型为`FLOAT16`或`BFLOAT16`时，`expandX`的数据类型可与`x`一致，也可为`HIFLOAT8`，输入`scalesOptional`必须传空指针。
          - 当`x`的数据类型为`HIFLOAT8`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN`、`FLOAT4_E2M1`、`FLOAT4_E1M2`时，输入`scalesOptional`必须传入有效数据，`expandX`的数据类型与`x`一致。
              - `x`的数据类型为`HIFLOAT8`时，`scalesOptional`的数据类型为`FLOAT`。
              - `x`的数据类型为`FLOAT8_E5M2`或`FLOAT8_E4M3FN`时，`scalesOptional`的数据类型为`FLOAT`或`FLOAT8_E8M0`。
              - `x`的数据类型为`FLOAT4_E2M1`或`FLOAT4_E1M2`时，`scalesOptional`的数据类型为`FLOAT8_E8M0`，且H必须为偶数。
              - `scalesOptional`的shape为(`BS`, `dim1`)，其中`dim1`需满足小于等于`H`。
      - `quantMode`取值为1时，表示静态量化场景，`expandX`的数据类型支持`INT8`、`HIFLOAT8`。
          - `expandX`的数据类型为`INT8`时有如下场景：
              - 输入的`scalesOptional`代表量化系数，shape为(1, )；
              - 输入的`scalesOptional`表示每个专家共享的平滑权重时，shape为(`H`，)；
              - 输入的`scalesOptional`代表融了每个专家的平滑权重的量化系数时，若有共享专家卡，其shape为(`sharedExpertNum` + `moeExpertNum`, `H`)，若无共享专家卡，其shape为(`moeExpertNum`, `H`)。
          - `expandX`的数据类型为`HIFLOAT8`时，`scalesOptional`的shape必须为(1, )。
      - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
          - 输入`scalesOptional`可传入空指针。
          - 若输入`scalesOptional`传入有效数据且存在共享专家卡时，其shape为(`sharedExpertNum` + `moeExpertNum`, `H`)。
          - 若输入`scalesOptional`传入有效数据且不存在共享专家卡时，其shape为(`moeExpertNum`, `H`)。
          - 输出`dynamicScalesOut`shape为`(A, )`
      - `quantMode`取值为3时，表示pergroup动态量化场景，expandX的数据类型支持`FLOAT8_E4M3FN`、`FLOAT8_E5M2`。
          - 输入`scalesOptional`可传入空指针。
          - 若输入`scalesOptional`传入有效数据且存在共享专家卡时，其shape为(`sharedExpertNum` + `moeExpertNum`, `H`)。
          - 若输入`scalesOptional`传入有效数据且不存在共享专家卡时，其shape为(`moeExpertNum`, `H`)。
          - 输出`dynamicScalesOut`shape为`(A, Ceil(H, 128))`，其中`Ceil(H, 128) = (H + 128 - 1) / 128`
      - `quantMode`取值为4时，表示mx量化场景，`expandX`的数据类型支持`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2`，输入`scalesOptional`必须传入空指针。输出`dynamicScalesOut`shape为`(A, Ceil(H, 64))`，其中`Ceil(H, 64) = (H + 64 - 1) / 64`。当`expandX`的数据类型为`FLOAT4_E2M1`或`FLOAT4_E1M2`时，H必须为偶数。

  <!-- end id15 -->

- **环境变量约束**：
  - **HCCL_BUFFSIZE**：
      调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB：

          <!-- npu="910b" id16 -->
          - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
              - commAlg为""或nullptr：依HCCL环境变量选择“fullmesh”或“hierarchy”公式。
              - commAlg为"fullmesh"：当`x`的数据类型为`INT32`时，设置大小要求(≥ 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* (H \* sizeof(int32) + 32B) + 2MB))；其他数据类型设置大小要求(≥ 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB))。
              - commAlg为"hierarchy"：设置大小要求(≥ (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8  (`K`))) \* 1B + 8MB，其中Align8(x) = ((x + 8 - 1) / 8) \* 8，Align512(x) = ((x + 512 - 1) / 512) \* 512)。
          <!-- end id16 -->
          <!-- npu="A3" id17 -->
          - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
              - ep通信域内：设置大小要求(≥ 2)且满足(≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K +   sharedExpertNum) \* maxBS \* Align512(2 \* H)))（`localExpertNum`需使用MoE专家卡的本卡专家数；`Align512(x) = ((x + 512 - 1) /   512) * 512`；`Align32(x) = ((x + 32 - 1) / 32) * 32`）。当前不支持TP通信域。
          <!-- end id17 -->
          <!-- npu="950" id18 -->
          - <term>Ascend 950DT</term>：ep通信域内设置大小要求(≥ 2)且满足(≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)))（`localExpertNum`需使用MoE专家卡的本卡专家数；`Align512(x) = ((x + 512 - 1) / 512) * 512`；`Align32(x) = ((x + 32 - 1) / 32) * 32`）。不支持TP通信域。

          <!-- end id18 -->

  - **HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE**：

      <!-- npu="910b" id19 -->
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该环境变量不再推荐使用，建议通过`commAlg`配置为"hierarchy"。
      <!-- end id19 -->
      <!-- npu="950,A3" id20 -->
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：不支持该环境变量。

      <!-- end id20 -->

- **通信域使用约束**：
   - 一个模型中的CombineV2系列算子和`aclnnMoeDistributeDispatchV2`仅支持相同EP通信域，且该通信域中不允许有其他算子。
   - 当前不支持TP域通信。

   <!-- npu="A3" id21 -->
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。

   <!-- end id21 -->

- **通信方式约束**：

  <!-- npu="950" id22 -->
  - <term>Ascend 950DT</term>：仅支持UB Memory通信。

  <!-- end id22 -->

- **组网约束**：

   <!-- npu="910b" id23 -->
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。

   <!-- end id23 -->

- **其他约束**：
  - 公式中的“/”表示整除。

## 调用示例

- 环境变量配置：

    ```bash
    # 运行前需设置一个环境变量
    ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量，以两机16卡为例，将两台机器设置为16
    export ENV_DEV_NUM=16
    ```

- 机器数量设置：

    两机16卡场景中，需将参数MACHINE_NUM设置为2，即

    ```Cpp
    const uint32_t MACHINE_NUM = 2;
    ```

    单机16卡场景则无需修改。

<!-- npu="910b" id24 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    无需配置ranktable文件以及环境变量RANK_TABLE_FILE、FIRST_RANK_ID。

    本示例支持A2算子运行在卡数为[2, 8]的单机环境中，运行前需要将示例代码中的IS_TEST_A2设置为true，确保执行A2分支。
    同时，用户可以根据需要在示例代码中设置EP_WORLD_SIZE_A2为卡数，并更改launchOneThreadDispatchV2AndCombineV2_A2函数中的moeExpertNum，使得moeExpertNum可以被EP_WORLD_SIZE_A2整除。

    算子编译命令如下，moe_distribute_dispatch_v2和moe_distribute_combine_v2算子都需要编译，这两个算子需要成对执行：

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
    ```

    示例算子执行命令如下：

    ```bash
    bash build.sh --run_example --ops=moe_distribute_dispatch_v2 eager cust
    ```

<!-- end id24 -->
<!-- npu="950,A3" id25 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  、<term>Ascend 950DT</term>：

    无需配置ranktable文件以及环境变量RANK_TABLE_FILE、FIRST_RANK_ID。

<!-- end id25 -->

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950,A3,910b" id26 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  、<term>Ascend 950DT</term>：

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <memory>
    #include <cstdio>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v2.h"


    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream dispatchV2Stream;
        aclrtStream combineV2Stream;
        aclrtContext context;
    };

    const uint32_t MACHINE_NUM = 1;
    const char* env_dev_num = std::getenv("ENV_DEV_NUM");

    const uint32_t EP_WORLD_SIZE = 2;
    const uint32_t DEV_NUM = EP_WORLD_SIZE;

    const bool IS_TEST_A2 = false;
    const bool IS_TEST_A3A5 = true;
    const uint32_t EP_WORLD_SIZE_A2 = 8;
    const uint32_t DEV_NUM_A2 = EP_WORLD_SIZE_A2;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
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
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
        );
        return 0;
    }

    void DestroyTensor(aclTensor *tensor) {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }

    void FreeDeviceAddr(void *deviceAddr) {
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    int launchOneThreadDispatchV2AndCombineV2_A3A5(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
            args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream, args.context
        );

        // 设置场景
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 1;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 0;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupListType = 0;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // 共享专家卡
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // Moe专家卡
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }
        std::string commAlg = "";

        /* 根据当前场景，构造device侧输入输出变量*/
        // 声明device侧输入输出变量
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;

        // 定义当前场景下各变量维度
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};

        long long xShapeSize = GetShapeSize(xShape);
        long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
        long long expandXShapeSize = GetShapeSize(expandXShape);
        long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
        long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

        // 构造host侧变量
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // 每个token发给moe专家{0, 1, ... k - 1}
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> scalesHostData(scalesShapeSize, 0);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        // 构造device侧变量
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        /* 声明算子执行必需变量 */
        uint64_t dispatchV2WorkspaceSize = 0;
        aclOpExecutor *dispatchV2Executor = nullptr;
        void *dispatchV2WorkspaceAddr = nullptr;

        uint64_t combineV2WorkspaceSize = 0;
        aclOpExecutor *combineV2Executor = nullptr;
        void *combineV2WorkspaceAddr = nullptr;

        /* 依次执行dispatchV2及combineV2算子 */
        // 调用dispatchV2算子第一阶段接口
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds,
            (quantMode > 0 ? scales : nullptr), nullptr,
            expertScales,
            hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, "", 0,
            0, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBS,
            expertTokenNumsType, commAlg.c_str(),
            expandX, dynamicScales,
            expandIdx, expertTokenNums,
            epRecvCounts, tpRecvCounts,
            expandScales, &dispatchV2WorkspaceSize,
            &dispatchV2Executor
        );
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // 根据dispatchV2算子第一阶段接口计算出的workspaceSize申请device内存
        if (dispatchV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // 调用dispatchV2算子第二阶段接口
        ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // 调用combineV2算子第一阶段接口
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0, 0,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
            commAlg.c_str(), x, &combineV2WorkspaceSize, &combineV2Executor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // 根据combineV2算子第一阶段接口计算出的workspaceSize申请device内存
        if (combineV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // 调用combineV2算子第二阶段接口
        ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributeCombineV2 execute successfully.\n", args.rankId);

        // 释放device资源
        if (dispatchV2WorkspaceSize > 0) {
            aclrtFree(dispatchV2WorkspaceAddr);
        }
        if (combineV2WorkspaceSize > 0) {
            aclrtFree(combineV2WorkspaceAddr);
        }
        DestroyTensor(x);
        DestroyTensor(expertIds);
        DestroyTensor(scales);
        DestroyTensor(expertScales);
        DestroyTensor(expandX);
        DestroyTensor(dynamicScales);
        DestroyTensor(expandIdx);
        DestroyTensor(expertTokenNums);
        DestroyTensor(epRecvCounts);
        DestroyTensor(tpRecvCounts);
        DestroyTensor(expandScales);

        FreeDeviceAddr(xDeviceAddr);
        FreeDeviceAddr(expertIdsDeviceAddr);
        FreeDeviceAddr(scalesDeviceAddr);
        FreeDeviceAddr(expertScalesDeviceAddr);
        FreeDeviceAddr(expandXDeviceAddr);
        FreeDeviceAddr(dynamicScalesDeviceAddr);
        FreeDeviceAddr(expandIdxDeviceAddr);
        FreeDeviceAddr(expertTokenNumsDeviceAddr);
        FreeDeviceAddr(epRecvCountsDeviceAddr);
        FreeDeviceAddr(tpRecvCountsDeviceAddr);
        FreeDeviceAddr(expandScalesDeviceAddr);

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchV2Stream);
        aclrtDestroyStream(args.combineV2Stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int launchOneThreadDispatchV2AndCombineV2_A2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, \
                    context = %p\n", args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream,                 \
                    args.context);

        int64_t BS = 32;
        int64_t H = 7168;
        int64_t K = 8;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 256;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE_A2;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;

        std::string commAlg = "fullmesh";
        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE_A2 - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;

        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *assistInfoForCombineDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        void *xOutDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *xActiveMask = nullptr;
        aclTensor *expertScales = nullptr;

        aclTensor *elasticInfo = nullptr; // A3
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *assistInfoForCombine = nullptr; // expandIdx
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;

        aclTensor *activationScale = nullptr; // 预留参数
        aclTensor *weightScale = nullptr; // 预留参数
        aclTensor *groupList = nullptr; // 预留参数

        aclTensor *sharedExpertX = nullptr; // A3


        aclTensor *xOut = nullptr;

        //定义当前场景下各变量维度
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{BS, K};

        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> assistInfoForCombineShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // 不分层
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};

        std::vector<int64_t> xOutShape{BS, H};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);

        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t assistInfoForCombineShapeSize = GetShapeSize(assistInfoForCombineShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);

        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> assistInfoForCombineHostData(assistInfoForCombineShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);


        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(assistInfoForCombineHostData, assistInfoForCombineShape, &assistInfoForCombineDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombine);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);


        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** 调用dispatch ********************************************/
        // 调用第一阶段接口
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                expertScales, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, commAlg.c_str(), expandX, dynamicScales, assistInfoForCombine, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 execute successfully.\n", args.rankId);
        /**************************************** 调用combine ********************************************/
        // 调用第一阶段接口
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds,
                                                            assistInfoForCombine, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            xActiveMask, activationScale, weightScale,
                                                            groupList, expandScales, sharedExpertX,
                                                            hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                            "", 0, 0, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, commAlg.c_str(), xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
            return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2                      \
                    execute successfully.\n", args.rankId);
        // 释放device资源
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        DestroyTensor(x);
        DestroyTensor(expertIds);
        DestroyTensor(scales);
        DestroyTensor(xActiveMask);
        DestroyTensor(expertScales);
        DestroyTensor(elasticInfo);
        DestroyTensor(expandX);
        DestroyTensor(dynamicScales);
        DestroyTensor(assistInfoForCombine);
        DestroyTensor(expertTokenNums);
        DestroyTensor(epRecvCounts);
        DestroyTensor(tpRecvCounts);
        DestroyTensor(expandScales);
        DestroyTensor(activationScale);
        DestroyTensor(weightScale);
        DestroyTensor(groupList);
        DestroyTensor(sharedExpertX);
        DestroyTensor(xOut);

        FreeDeviceAddr(xDeviceAddr);
        FreeDeviceAddr(expertIdsDeviceAddr);
        FreeDeviceAddr(scalesDeviceAddr);
        FreeDeviceAddr(expertScalesDeviceAddr);
        FreeDeviceAddr(expandXDeviceAddr);
        FreeDeviceAddr(dynamicScalesDeviceAddr);
        FreeDeviceAddr(assistInfoForCombineDeviceAddr);
        FreeDeviceAddr(expertTokenNumsDeviceAddr);
        FreeDeviceAddr(epRecvCountsDeviceAddr);
        FreeDeviceAddr(tpRecvCountsDeviceAddr);
        FreeDeviceAddr(expandScalesDeviceAddr);
        FreeDeviceAddr(xOutDeviceAddr);

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchV2Stream);
        aclrtDestroyStream(args.combineV2Stream);
        aclrtDestroyContext(args.context);
        LOG_PRINT("[INFO] device_%d DeStroy.\n", args.rankId);
        aclrtResetDevice(args.rankId);
        LOG_PRINT("[INFO] device_%d Reset.\n", args.rankId);
        return 0;
    }

    int run_example_on_A2()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchV2Stream[DEV_NUM_A2];
        aclrtStream combineV2Stream[DEV_NUM_A2];
        aclrtContext context[DEV_NUM_A2];
        for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[EP_WORLD_SIZE_A2];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE_A2; epId++) {
            devicesEp[epId] = epId;
        }

        HcclComm commsEp[EP_WORLD_SIZE_A2];
        ret = HcclCommInitAll(EP_WORLD_SIZE_A2, devicesEp, commsEp);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

        Args args[DEV_NUM_A2];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM_A2);
        for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A2, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }

    int run_example_on_A3A5()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtStream dispatchV2Stream[DEV_NUM];
        aclrtStream combineV2Stream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[EP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[epId] = epId;
        }
        // 初始化ep通信域.
        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret
        );

        Args args[DEV_NUM];
        // 各线程调用各卡执行算子
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A3A5, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }


    int main(int argc, char *argv[])
    {
        if (!env_dev_num) {
            LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            return 0;
        }
        int actual_env_dev_num = std::stoi(std::string(env_dev_num));
        if (actual_env_dev_num < DEV_NUM) {
            LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
            return 0;
        }
        if (IS_TEST_A2) {
            LOG_PRINT("Example on <Atlas A2> will be executed!\n");
            int ret = run_example_on_A2();
        }
        else if (IS_TEST_A3A5) {
            LOG_PRINT("Example on <Atlas A3> or <Atlas A5> will be executed!\n");
            int ret = run_example_on_A3A5();
        }

        return 0;
    }
    ```

<!-- end id26 -->
