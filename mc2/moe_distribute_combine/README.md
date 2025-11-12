# MoeDistributeCombine


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |



## 功能说明

算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。

- 不存在TP域通信时：
$$
ataOut = AllToAllV(expandX)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

- 存在TP域通信时：
$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

注意该算子必须与MoeDistributeDispatch配套使用，相当于按MoeDistributeDispatch算子收集数据的路径原路返还。



## 参数说明

<table style="undefined;table-layout: fixed; width: 1480px">
<colgroup>
<col style="width: 170px">
<col style="width: 130px">
<col style="width: 900px">
<col style="width: 200px">
<col style="width: 80px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>expandX</td>
<td>输入</td>
<td>根据expertIds进行扩展过的token特征。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>expertIds</td>
<td>输入</td>
<td>每个token的topK个专家索引。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expandIdx</td>
<td>输入</td>
<td>表示同一专家收到的token个数，对应MoeDistributeDispatch中的expandIdx输出。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>epSendCounts</td>
<td>输入</td>
<td>从EP通信域各卡接收的token数，对应MoeDistributeDispatch中的epRecvCounts输出。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expertScales</td>
<td>输入</td>
<td>每个token的topK个专家的权重。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>tpSendCounts</td>
<td>可选输入</td>
<td>从TP通信域各卡接收的token数，对应MoeDistributeDispatch中的tpRecvCounts输出，若有TP域通信需传参，若无TP域通信传空指针。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>xActiveMask</td>
<td>可选输入</td>
<td>预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>activationScale</td>
<td>可选输入</td>
<td>预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>weightScale</td>
<td>可选输入</td>
<td>预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>groupList</td>
<td>可选输入</td>
<td>预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>expandScales</td>
<td>可选输入</td>
<td>表示本卡输入Token的权重，对应MoeDistributeDispatch中的expandScales输出。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>groupEp</td>
<td>属性</td>
<td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)，不能和groupTp相同。</td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>epWorldSize</td>
<td>属性</td>
<td>EP通信域大小。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>epRankId</td>
<td>属性</td>
<td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>moeExpertNum</td>
<td>属性</td>
<td>MoE专家数量，取值范围(0, 512]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupTp</td>
<td>可选属性</td>
<td><li>TP通信域名称（数据并行通信域）。</li><li>默认值为""。</li></td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>tpWorldSize</td>
<td>可选属性</td>
<td><li>TP通信域大小。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>tpRankId</td>
<td>可选属性</td>
<td><li>TP域本卡Id。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertShardType</td>
<td>可选属性</td>
<td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>可选属性</td>
<td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</li><li>默认值为1。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>可选属性</td>
<td><li>表示共享专家卡数量。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>globalBs</td>
<td>可选属性</td>
<td><li>EP域全局的batch size大小。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>outDtype</td>
<td>可选属性</td>
<td><li>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>commQuantMode</td>
<td>可选属性</td>
<td><li>通信量化类型，取值范围0或2；0表示通信不量化，2表示通信int8量化。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupListType</td>
<td>可选属性</td>
<td><li>groupList格式，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>x</td>
<td>输出</td>
<td>表示处理后的token，数据类型、数据格式与expandX保持一致。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
</tbody>
</table>

* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    * 不支持共享专家场景，不支持`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`属性。
    * 当前不支持TP域通信，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，且`tpSendCounts`为无效内容。
    * 仅设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0时，必须传入`expandScales`。

* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    * 不支持`expandScales`。

## 约束说明

- `MoeDistributeDispatch`算子与`MoeDistributeCombine`算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`MoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`属性取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeDispatch`对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum。`
        - 对于MoE专家，当`globalBs`为0时，要满足`A` >= `BS` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足`A` >= `globalBs` * min(`localExpertNum`, `K`)。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)`，localExpertNum` > 1时，不支持TP域通信。

- 本文公式中的"/"表示整除。

- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围(0, 7168]，且保证是32的整数倍。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 256]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ `moeExpertNum`，取值范围为[1, 16]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= (`BS` * `epWorldSize` * min(`localExpertNum`, `K`) * `H` * 4B + 4MB)。
    - `HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`：设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0可以减少跨机通信数据量，可能提升算子性能。 此时，要求`HCCL_BUFFSIZE` >= `moeExpertNum` * `BS` * (`H` * 2 + 16 * Align8(`K`))B + 104MB。并且，对于入参`moeExpertNum`，只要求`moeExpertNum` % `epWorldSize` = 0，不要求`moeExpertNum` / `epWorldSize` <= 24，其中Align8(x) = ((x + 8 - 1) / 8) * 8。
    - 参数约束：
        - `epWorldSize`：取值支持[16, 256]，且保证是8的整数倍。
            - `HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0：最大值为64。
        - `moeExpertNum`：需满足`moeExpertNum` / `epWorldSize` <= 24。
            - 环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0时，无上述约束。
        - `globalBs`：当每个rank的`BS`数一致时，`globalBs` = `BS` * `epWorldSize` 或 `globalBs` = 0；当每个rank的`BS`数不一致时，`globalBs` = `maxBs` * `epWorldSize`或者`globalBs` = 256 * `epWorldSize`，其中`maxBs`表示表示单rank `BS`最大值，建议按`maxBs` * `epWorldSize`传入，固定按256 * `epWorldSize`传入在后续版本BS支持大于256的场景下会无法支持。
        - `commQuantMode`：2，开启通信int8量化，仅当`HCCL_INTRA_PCIE_ENABLE`为1且`HCCL_INTRA_ROCE_ENABLE`为0且驱动版本不低于25.0.RC1.1时支持。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值为7168。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ moeExpertNum，取值范围为[1, 8]。
    - 参数约束：
        - `epWorldSize`：取值支持8、16、32、64、128、144、256、288。
        - `groupTp`：字符串长度范围为[1, 128)，不能和groupEp相同。
        - `tpWorldSize`：取值范围[0, 2]，0和1表示无tp域通信，有tp域通信时仅支持2。
        - `tpRankId`：取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复。无TP域通信时，传0即可。
        - `sharedExpertRankNum`：当前取值范围[0, `epWorldSize`)，不为0时需满足`epWorldSize` % `sharedExpertRankNum` = 0。
        - `globalBs`：当每个rank的`BS`数一致时，`globalBs` = `BS` * `epWorldSize` 或 `globalBs` = 0；当每个rank的`BS`数不一致时，`globalBs` = `maxBs` * `epWorldSize`，其中`maxBs`表示单卡`BS`最大值。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= 2且满足1024 ^ 2 * (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` * 2 * (`H` + 128) * (`epWorldSize` * `localExpertNum` + `K` + 1)，`localExpertNum`需使用MoE专家卡的本卡专家数。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_combine.cpp](./examples/test_aclnn_moe_distribute_combine.cpp) | 通过[aclnnMoeDistributeCombine](./docs/aclnnMoeDistributeCombine.md)接口方式调用moe_distribute_combine算子。 |
