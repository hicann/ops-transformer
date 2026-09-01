# FFNToAttentionV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

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
    <td>context</td>
    <td>输入</td>
    <td>本卡通信域信息数据。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
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
    <td>通信域大小：取值区间[2, 1024]。</td>
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
    <tr>
    <td>cclBufferSize</td>
    <td>属性</td>
    <td>CCL通信缓冲区总大小（Bytes），由 <code>get_ffn_to_attention_ccl_buffer_size</code> 接口计算得到。需 >= token_info_size + token_data_size（按2MB向上对齐）。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>

  </tbody>
</table>

## 约束说明

- 所有rank使用的`group`、`worldSize`、`tokenInfoTableShape`、`tokenDataShape`和`cclBufferSize`必须保持一致。
- 所有rank必须先完成MC2 context初始化。FFN Worker执行算子期间，Attention Worker必须保持context、通信window和通信域有效。
- shape取值需满足以下约束：
    - `Y`表示本rank需要分发的最大token数量。
    - `tokenInfoTableShape`为`[1, BS, expertNumPerToken]`，`tokenDataShape`为`[1, BS, expertNumPerToken, HS]`，两个shape的前三维必须相同。
    - `BS`取值范围为`[1, 512]`。
    - `H`表示hidden size，取值范围为`[1024, 8192]`。
    - `HS`表示通信window中每个token槽位的长度，需满足`H <= HS <= 8320`。
    - `worldSize`取值范围为`[2, 1024]`。
- 可选输入`attnRankTableOptional`必须是一维Tensor，其长度小于`worldSize`；表中每个目标rank的取值范围为`[0, worldSize)`。
- 同一目标rank内，多个有效token不得映射到相同的`(microBatchId, tokenId, expertOffset)`槽位，否则写入结果存在覆盖风险。

- `cclBufferSize`为HBM上分配的通信window总大小，单位为Byte，必须大于0并满足：

$$cclBufferSize \ge \mathrm{CeilAlign}(\mathrm{CeilAlign}(\mathrm{tokenInfoSize}, 512) + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

其中：

  - `tokenInfoSize = microBatchNum × BS × expertNumPerToken × 4B`
  - `tokenDataSize = microBatchNum × BS × expertNumPerToken × HS × 2B`

可通过`get_ffn_to_attention_ccl_buffer_size`接口计算所需最小值。

- 通信域使用约束：
    - FFNToAttentionV2算子的通信域中不允许有其他算子。


## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | - | 通过[ffn_to_attention](../../torch_extension/cann_ops_transformer/docs/zh/ffn_to_attention.md)PyTorch接口方式调用ffn_to_attention算子。 |
