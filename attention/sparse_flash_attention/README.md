# SparseFlashAttention

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。
    本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、Q_S和KV_S分别表示query和key/value的Sequence Length、Q_N和KV_N分别表示query和key/value的Head Num、Q_D和KV_D分别表示query和key/value的Head Dim、Dr表示rope的Head Dim、Q_T和KV_T分别表示query和key/value的Total Tokens、sparse_size表示一次离散选取的block数、block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>attention结构的Query输入，不支持空tensor和非连续。layout_query为"BSND"时，shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时，shape为[Q_T, Q_N, Q_D]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>attention结构的Key输入，不支持空tensor。layout_kv为"BSND"时，shape为[B, KV_S, KV_N, KV_D]。layout_kv为"TND"时，shape为[KV_T, KV_N, KV_D]。layout_kv为"PA_BSND"时，shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>attention结构的Value输入，不支持空tensor，shape与key的shape一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输入</td>
      <td>离散取kvCache的索引，不支持空tensor和非连续。sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分、无效值均在后半部分，且sparse_size大于0。layout_query为"BSND"时，shape为[B, Q_S, KV_N, sparse_size]。layout_query为"TND"时，shape为[Q_T, KV_N, sparse_size]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表，不支持空tensor和非连续。第二维长度不小于所有batch中最大的KV_S对应的block数量，即KV_S_max / block_size向上取整。shape为[B, KV_S_max/block_size]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>输入</td>
      <td>表示不同Batch中query的有效token数，不支持空tensor和非连续。可传入None表示与query的Q_S长度相同。shape为[B,]。每个Batch的有效token数不超过query中的Q_S大小且不小于0。layout_query为"TND"时该入参必须传入，且以元素数量作为B值；每个元素表示当前batch与之前所有batch的token数总和（前缀和）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv</td>
      <td>输入</td>
      <td>表示不同Batch中key和value的有效token数，不支持空tensor和非连续。可传入None表示与key的KV_S长度相同。shape为[B,]。每个Batch的有效token数不超过key/value中的KV_S大小且不小于0。layout_kv为"TND"或"PA_BSND"时该入参必须传入；其中layout_kv为"TND"时，每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_rope</td>
      <td>输入</td>
      <td>表示MLA结构中的query的rope信息，不支持空tensor和非连续。layout_query为"BSND"时，shape为[B, Q_S, Q_N, Dr]。layout_query为"TND"时，shape为[Q_T, Q_N, Dr]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_rope</td>
      <td>输入</td>
      <td>表示MLA结构中的key的rope信息，不支持空tensor。layout_kv为"BSND"时，shape为[B, KV_S, KV_N, Dr]。layout_kv为"TND"时，shape为[KV_T, KV_N, Dr]。layout_kv为"PA_BSND"时，shape为[block_num, block_size, KV_N, Dr]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>输入</td>
      <td>attention结构中的可学习的sinks信息。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_value</td>
      <td>可选属性</td>
      <td>代表缩放系数。</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_block_size</td>
      <td>可选属性</td>
      <td>代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>可选属性</td>
      <td>标识输入query的数据排布格式，默认值为"BSND"，支持传入BSND和TND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>标识输入key的数据排布格式，默认值为"BSND"，支持传入TND、BSND和PA_BSND，其中PA_BSND在开启PageAttention时使用。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>可选属性</td>
      <td>表示sparse的模式。sparse_mode为0时代表全部计算；sparse_mode为3时代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>可选属性</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>可选属性</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_mode</td>
      <td>可选属性</td>
      <td>表示attention的模式，仅支持传入2，表示MLA-absorb模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>用于表示是否返回softmax_max和softmax_sum。True表示返回，False表示不返回，默认值为False。该参数仅在训练且layout_kv不为PA_BSND场景支持。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out</td>
      <td>输出</td>
      <td>公式中的输出，不支持空tensor和非连续。layout_query为"BSND"时，shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时，shape为[Q_T, Q_N, Q_D]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_max_out</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果取max得到softmax_max，不支持空tensor和非连续。layout_query为"BSND"时，shape为[B, KV_N, Q_S, Q_N/KV_N]。layout_query为"TND"时，shape为[KV_N, Q_T, Q_N/KV_N]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_sum_out</td>
      <td>输出</td>
      <td>Attention算法query乘key的结果减去softmax_max后取exp并求sum，得到softmax_sum，不支持空tensor和非连续。layout_query为"BSND"时，shape为[B, KV_N, Q_S, Q_N/KV_N]。layout_query为"TND"时，shape为[KV_N, Q_T, Q_N/KV_N]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- KV_N仅支持1。
- block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
- 参数query中的Q_D和key、value的KV_D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 当前只支持query_rope和key_rope传入，不支持rope为空。
- sinks仅支持Ascend 950PR/Ascend 950DT。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - Q_N支持1~128。
  - sparse_block_size仅支持1。
  - 仅在layout_key为PA_BSND时，key、value和key_rope支持0轴非连续。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - Q_N支持1/2/4/8/16/32/64/128。
  - 参数sparse_block_size支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size。
  - 不支持sinks。
  - key、value和key_rope不支持非连续。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_sparse_flash_attention.cpp">test_aclnn_sparse_flash_attention
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashAttention.md">aclnnSparseFlashAttention
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_sparse_flash_attention_v2.cpp">test_aclnn_sparse_flash_attention_v2
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashAttentionV2.md">aclnnSparseFlashAttentionV2
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody>
</table>
