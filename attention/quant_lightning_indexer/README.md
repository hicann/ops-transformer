# QuantLightningIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为SparseFlashAttention的输入。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、Q_S和K_S分别表示query和key的Sequence Length、Q_N和K_N分别表示query和key的Head Num、D表示Head Dim（Q_D和K_D取值相等为128）、Q_T和K_T分别表示query和key的Total Tokens、sparse_count表示最后选取的索引个数（topK）、block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。K_N仅支持1。
<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>
          <ul>
                <li>公式中的输入Q。</li>
                <li>不支持非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N, D]。layout_query为TND时，shape为[Q_T, Q_N, D]。</li>
                <li>Q_N支持[1, 64]。</li>
          </ul>
      </td>
      <td>INT8、FLOAT8_E4M3、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的输入K。</li>
                <li>在layout_key为PA_BSND时，支持0轴非连续。</li>
                <li>layout_key为PA_BSND时，shape为[block_num, block_size, K_N, D]。layout_key为BSND时，shape为[B, K_S, K_N, D]。layout_key为TND时，shape为[K_T, K_N, D]。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
                <li>K_N仅支持1。</li>
          </ul>
      </td>
      <td>INT8、FLOAT8_E4M3、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的输入W。</li>
                <li>不支持非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N]。layout_query为TND时，shape为[Q_T, Q_N]。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_dequant_scale</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中Query的反量化系数Scale_Q。</li>
                <li>不支持非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N]。layout_query为TND时，shape为[Q_T, Q_N]。</li>
          </ul>
      </td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_dequant_scale</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中Key的反量化系数Scale_K。</li>
                <li>在layout_key为PA_BSND时，支持0轴非连续。</li>
                <li>layout_key为BSND时，shape为[B, K_S, K_N]。layout_key为TND时，shape为[K_T, K_N]。</li>
                <li>layout_key为PA_BSND时，shape为[block_num, block_size, K_N]。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
          </ul>
      </td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>输入</td>
      <td>
          <ul>
                <li>每个Batch中，Query的有效token数。</li>
                <li>不支持非连续。</li>
                <li>shape为[B,]</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的Q_S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的Q_S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
                <li>不能出现负值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_key</td>
      <td>输入</td>
      <td>
          <ul>
                <li>每个Batch中，Key的有效token数。</li>
                <li>不支持非连续。</li>
                <li>shape为[B,]</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的K_S长度相同。</li>
                <li>该参数中每个Batch的有效token数不超过key中的K_S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layout_key为TND或PA_BSND时，该入参必须传入，layout_key为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>
          <ul>
                <li>表示PageAttention中KV存储使用的block映射表。</li>
                <li>不支持非连续。</li>
                <li>shape支持[B, K_S_max/block_size]</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual_seq_lengths_key对应的block数量）</li>
                <li>block_size取值为16的整数倍，最大支持到1024。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_quant_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Query的量化模式。</li>
                <li>当前支持Per-Token-Head量化模式。</li>
                <li>当前仅支持传入0。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_quant_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Key的量化模式。</li>
                <li>当前支持Per-Token-Head量化模式。</li>
                <li>当前仅支持传入0。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Query的数据排布格式。</li>
                <li>当前支持BSND、TND。</li>
                <li>默认值为BSND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_key</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Key的数据排布格式。</li>
                <li>当前支持PA_BSND、BSND、TND。</li>
                <li>在非PageAttention场景下，layout_key应与layout_query保持一致。</li>
                <li>默认值为BSND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_count</td>
      <td>属性</td>
      <td>
          <ul>
                <li>topK阶段需要保留的block数量。支持[1, 2048]。</li>
                <li>默认值为2048。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>表示sparse的模式。</li>
                <li>sparse_mode为0时，代表defaultMask模式。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
                <li>默认值为3。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>属性</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>属性</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_stride0</td>
      <td>属性</td>
      <td>
          <ul>
                <li>表示key获取stride第0维的信息。</li>
                <li>默认值为-1。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_dequant_scale_stride0</td>
      <td>属性</td>
      <td>
          <ul>
                <li>表示key_dequant_scale获取stride第0维的信息。</li>
                <li>默认值为-1。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输出</td>
      <td>
          <ul>
                <li>公式中的Indices输出。</li>
                <li>layout_query为BSND时输出shape为[B, Q_S, K_N, sparse_count]。layout_query为TND时输出shape为[Q_T, K_N, sparse_count]。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>

  </table>

Atlas A3训练系列产品/Atlas A3推理系列产品：

  - query和key的数据类型支持`INT8`。
  - 仅支持weights、query_dequant_scale、key_dequant_scale数据类型为`FLOAT16、FLOAT16、FLOAT16`。
  - key和key_dequant_scale在layout_key为PA_BSND时支持0轴非连续，其余轴必须连续。

Ascend 950PR/Ascend 950DT：

  - query Q_N仅支持8、16、24、32、64。
  - query和key的数据类型支持`FLOAT8_E4M3、HIFLOAT8、INT8`。
  - 当query和key的数据类型为`FLOAT8_E4M3`时，支持weights、query_dequant_scale、key_dequant_scale的数据类型为`BFLOAT16、FLOAT、FLOAT`或`FLOAT16、FLOAT16、FLOAT16`；
  - 当query和key的数据类型为`HIFLOAT8`时，仅支持weights、query_dequant_scale、key_dequant_scale数据类型为`BFLOAT16、FLOAT、FLOAT`；
  - 当query和key的数据类型为`INT8`时，仅支持weights、query_dequant_scale、key_dequant_scale数据类型为`FLOAT16、FLOAT16、FLOAT16`。

## 约束说明

- 该接口支持图模式。
- 该接口要求$W \odot Scale_Q$的结果在`float16`的表示范围内。
- 该接口的TopK过程对NAN排序是未定义行为。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">图模式</td>
    <td class="tg-0pky">
    <a href="./examples/test_npu_quant_lightning_indexer.py">test_npu_quant_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过算子IR构图方式调用npu_quant_lightning_indexer算子
    </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_quant_lightning_indexer.cpp">test_aclnn_quant_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnQuantLightningIndexer.md">aclnnQuantLightningIndexer
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
