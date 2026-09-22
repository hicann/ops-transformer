# FFNToAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>       |    √     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将FFN节点上的数据发往Attention节点。

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据，2D Tensor，shape为 <code>(Y, H)</code>（H=hidden size）。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sessionIds</td>
    <td>输入</td>
    <td>每个token的Attention Worker节点索引，1D Tensor，shape为 <code>(Y, )</code>sessionIds取值区间为[0, attnRankNum-1]</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>microBatchIds</td>
    <td>输入</td>
    <td>每个token的microBatch索引，1D Tensor，shape为 <code>(Y, )，microBatchIds取值区间为[0, MircoBatchNum-1]</code></td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tokenIds</td>
    <td>输入</td>
    <td>每个token在microBatch中的token索引，1D Tensor，shape为 <code>(Y, )</code>，tokenIds取值区间为[0, Bs-1]</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertOffsets</td>
    <td>输入</td>
    <td>每个token在tokenInfoTableShape中PerTokenExpertNum的索引，1D Tensor，shape为 <code>(Y, )</code>，expertOffsets取值区间为[0, ExpertNumPerToken-1]。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>actualTokenNum</td>
    <td>输入</td>
    <td>本卡发送的实际token总数，1D Tensor，shape为 <code>(1, )，actualTokenNum的取值为[0, Y]。</code>。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>attnRankTableOptional</td>
    <td>可选输入</td>
    <td>映射每一个Attention Worker对应的卡Id。
    <br>Attention Worker必须从0卡开始连续部署；
    <br>若输入空指针，采用默认策略：每张卡的Id作为对应Attention Worker的Id，取值区间为[0, attnRankNum-1]。</td>
    <td>INT32</td>
    <td>ND（支持非连续Tensor）</td>
    </tr>
    <tr>
    <td>group</td>
    <td>属性</td>
    <td>通信域名称（专家并行），字符串长度[1, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>属性</td>
    <td>通信域大小：取值区间[2, 768]。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenInfoTableShape</td>
    <td>属性</td>
    <td>Token信息列表大小，包含microBatch的大小（MircoBatchNum）、BatchSize大小（Bs）、以及每个Token对应的Expert数量（ExpertNumPerToken）。</td>
    <td>INT32</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenDataShape</td>
    <td>属性</td>
    <td>Token数据列表大小，包含microBatch的大小（MircoBatchNum）、BatchSize大小（Bs）、每个Token对应的Expert数量(ExpertNumPerToken)、以及token和scale长度(HS)。</td>
    <td>INT32</td>
    <td>-</td>
    </tr>

  </tbody>
</table>

## 约束说明

- 调用算子过程中使用的`group`、`worldSize`、`tokenInfoTableShape`、`tokenDataShape`参数及`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致。

- 参数说明里shape格式说明：
    - `Y`：表示本卡需要分发的最大token数量。
    - `BS`：示各Attention节点上的发送token数，取值范围为0 < `BS` ≤ 512。
    - `H`：表示hidden size隐藏层大小，取值范围为1024 ≤ `H` ≤ 8192。
    - `HS`：表示hidden与scale隐藏层大小，取值范围为1152 ≤ `HS` ≤ 8320。
    - `ffnRankNum`：表示选取ffnRankNum个卡作为FFnWorker，取值范围为0 < `ffnRankNum` < `worldSize`。
    - `attnRankNum`：表示选取attnRankNum个卡作为AttnWorker，取值范围为0 < `attnRankNum` < `worldSize`。
    - `sharedExpertNum`：表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），取值范围为0 ≤ `sharedExpertNum` ≤ 4。

- 通信域使用约束：
    - FFNToAttention算子的通信域中不允许有其他算子。

- <term>Atlas A3系列产品</term>  ：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_ffn_to_attention.cpp](./examples/test_aclnn_ffn_to_attention.cpp) | 通过[aclnnFFNToAttention](./docs/aclnnFFNToAttention.md)接口方式调用FFNToAttention算子。 |
