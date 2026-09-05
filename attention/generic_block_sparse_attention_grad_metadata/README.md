# GenericBlockSparseAttentionGradMetadata

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

+ 算子功能：GenericBlockSparseAttentionGradMetadata根据KV2Q稀疏块索引表`rsvdBlockIdx`/`rsvdBlockCount`，按B → N2 → J → G顺序展开`(b, n2, j, g)`任务列表，并在AIC核间做负载均衡，供后续GenericBlockSparseAttentionGrad算子消费。
+ 该算子不建议单独使用，建议与aclnnGenericBlockSparseAttentionGrad配合使用。

$$
\text{metaSize} = 80 + B \times N1 \times J \times 4
$$

其中`J = ceilDiv(maxKvSeqlen, blockShapeY)`。

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">rsvdBlockIdx</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">稀疏块索引数组，指定每个KV块选择的Q块/token索引。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">rsvdBlockCount</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">指定每个KV块实际选择的Q数量。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsQOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">TND layout下query的累积序列长度。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsKvOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">TND layout下key/value的累积序列长度。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedQOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">各batch中query的实际序列长度。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedKvOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">各batch中kv的实际序列长度。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">maxQSeqlen</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">所有batch中query序列长度的最大值。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">maxKvSeqlen</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">所有batch中kv序列长度的最大值。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">numQHeads</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">query的head数（N1）。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">numKvHeads</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">key/value的head数（N2）。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">headDim</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">embed维度，当前固定128。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">blockShape</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">稀疏块形状[blockShapeX, blockShapeY]，当前支持[1, 128]。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">isPackedGQA</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">同一group内qHead是否共享稀疏pattern，当前仅支持1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">layoutQ</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">query侧layout格式，支持TND/BNSD/BSND。</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">layoutKv</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">key/value侧layout格式，须与layoutQ一致。</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">maskType</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">mask模式，当前仅支持1（RIGHT_DOWN_CAUSAL）。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">softmaxPrecision</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">Softmax精度级别，取值0或1，当前实现传0。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">winLeft</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">滑窗向前包含token数，不使能时须为-1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">winRight</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">滑窗向后包含token数，不使能时须为-1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">metadata</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">分核信息，长度≥80+B×N1×J×4。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## 约束说明

* <term>Ascend 950PR/Ascend 950DT</term>：支持本算子。
* 须与aclnnGenericBlockSparseAttentionGrad配合使用；主算子调用前必须先成功执行本算子。
* layoutQ与layoutKv须相同，取值TND/BNSD/BSND；TND布局下cuSeqLengthsQOptional/cuSeqLengthsKvOptional必选。
* HeadDim固定为128；numQHeads/numKvHeads取值范围[1, 128]，且numQHeads % numKvHeads == 0。
* blockShape当前仅支持[1, 128]；isPackedGQA当前仅支持1；maskType当前仅支持1。
* winLeft/winRight不使能时必须为-1。
* metadata长度须满足shape[0] ≥ 80 + B × numQHeads × J × 4；任务数上界B × numQHeads × J ≤ 1048576。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_generic_block_sparse_attention_grad_metadata](./examples/test_aclnn_generic_block_sparse_attention_grad_metadata.cpp) | 通过[aclnnGenericBlockSparseAttentionGradMetadata](./docs/aclnnGenericBlockSparseAttentionGradMetadata.md)接口方式调用GenericBlockSparseAttentionGradMetadata算子。 |
