# Ascendir_FlashAttentionScoreGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id8 -->

## 功能说明

- 算子功能：训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算：

  - pseType=1时，需要先add再mul。
  - pseType≠1时，需要先mul再add。

- 计算公式：

  已知注意力的正向计算公式为：

    - pseType=1时，公式如下：
      $$
      Y=Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
      $$

    - pseType≠1时，公式如下：
      $$
      Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
      $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

  则注意力的反向计算公式为：

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

  其中增加sink之后的计算逻辑如下，主要修改相关softmax_max和softmax_sum逻辑计算部分：

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + e^{sink - m}}
  $$

  $$
  dSink = reduce(-Softmax(S) * dP * SimpleSoftmax(sink, x\_max, x\_sum))
  $$

## Ascend IR定义

Ascend IR定义所在头文件路径为[op_graph/flash_attention_score_grad_proto.h](../op_graph/flash_attention_score_grad_proto.h)。

```cpp
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_o, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(ds_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dq_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dsink, TensorType({DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(softmax_in_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)
```

### 参数说明

<table style="undefined;table-layout: fixed; width: 1340px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 210px">
  <col style="width: 80px">
  <col style="width: 240px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（tensor）</td>
      <td>输入</td>
      <td>公式中的输入query。</td>
      <td>数据类型与key/value的数据类型一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>key（tensor）</td>
      <td>输入</td>
      <td>公式中的输入key。</td>
      <td>数据类型与query/value的数据类型一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>value（tensor）</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>数据类型与query/key的数据类型一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dy（tensor）</td>
      <td>输入</td>
      <td>公式中的dY，表示输出的梯度。</td>
      <td>-</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>pse_shift（tensor）</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>数据类型与query/key/value的数据类型一致，该参数需要与pse_type配套使用。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv]、[B,N,1,Skv]、[1,N,Sq,Skv]、[B,N,1024,Skv]、[1,N,1024,Skv]</td>
    </tr>
    <tr>
      <td>drop_mask（tensor）</td>
      <td>可选输入</td>
      <td>公式中的Dropout，表示数据丢弃掩码。取值为1代表保留该数据，为0代表丢弃该数据。</td>
      <td>支持[B,N,S,S]。</td>
      <td>uint8</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>padding_mask（tensor）</td>
      <td>可选输入</td>
      <td>预留参数，暂未使用。</td>
      <td>-</td>
      <td>float16、bf16、float32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atten_mask（tensor）</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码。</td>
      <td>取值为1代表该位不参与计算（不生效），为0代表该位参与计算。</td>
      <td>bool、uint8</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv]、[B,1,Sq,Skv]、[1,1,Sq,Skv]、[Sq,Skv]</td>
    </tr>
    <tr>
      <td>softmax_max（tensor）</td>
      <td>可选输入</td>
      <td>注意力正向计算的Max中间结果。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]、[N,T,8]、[T,N,8]</td>
    </tr>
    <tr>
      <td>softmax_sum（tensor）</td>
      <td>可选输入</td>
      <td>注意力正向计算的Sum中间结果。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]、[N,T,8]、[T,N,8]</td>
    </tr>
    <tr>
      <td>softmax_in（tensor）</td>
      <td>可选输入</td>
      <td>注意力正向计算的中间输出。预留参数，暂未使用。</td>
      <td>-</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>attention_in（tensor）</td>
      <td>可选输入</td>
      <td>注意力正向的最终输出。</td>
      <td>数据类型和shape与dy保持一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>prefix（tensor）</td>
      <td>可选输入</td>
      <td>prefix稀疏计算场景中每个Batch的N值。</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>actual_seq_qlen（tensor）</td>
      <td>可选输入</td>
      <td>表示每个Batch的query序列长度。</td>
      <td>使用时input_layout需设置为TND。例如：q的seqlen为[2,2,2,2,2]，该参数需设置为[2,4,6,8,10]。</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>actual_seq_kvlen（tensor）</td>
      <td>可选输入</td>
      <td>表示每个Batch的kv序列长度。</td>
      <td>使用时input_layout需设置为TND。例如：kv的seqlen为[2,2,2,2,2]，该参数需设置为[2,4,6,8,10]。</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>q_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>外切场景下，当前分块Q的sequence在全局中的起始索引。</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>kv_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>外切场景下，当前分块KV的sequence在全局中的起始索引。</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_q（tensor）</td>
      <td>可选输入</td>
      <td>query输入的反量化参数。</td>
      <td>暂不使用。</td>
      <td>float32</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_k（tensor）</td>
      <td>可选输入</td>
      <td>key输入的反量化参数。</td>
      <td>暂不使用。</td>
      <td>float32</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_v（tensor）</td>
      <td>可选输入</td>
      <td>value输入的反量化参数。</td>
      <td>暂不使用。</td>
      <td>float32</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_dy（tensor）</td>
      <td>可选输入</td>
      <td>dy输入的反量化参数。</td>
      <td>暂不使用。</td>
      <td>float32</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_o（tensor）</td>
      <td>可选输入</td>
      <td>attention_in输入的反量化参数。</td>
      <td>暂不使用。</td>
      <td>float32</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>query_rope（tensor）</td>
      <td>可选输入</td>
      <td>Q的RoPE旋转位置编码输入。</td>
      <td>数据类型与query的数据类型一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>key_rope（tensor）</td>
      <td>可选输入</td>
      <td>K的RoPE旋转位置编码输入。</td>
      <td>数据类型与key的数据类型一致。</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>sink（tensor）</td>
      <td>可选输入</td>
      <td>Sinking参数。</td>
      <td>维度为1，长度与query的head_num相同。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[headNum]</td>
    </tr>
    <tr>
      <td>ds_scale（tensor）</td>
      <td>可选输入</td>
      <td>预留参数，暂未使用。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p_scale（tensor）</td>
      <td>可选输入</td>
      <td>预留参数，暂未使用。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_value（float）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的scale，表示缩放系数，作为计算流中Muls的scalar值。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob（float）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的keep_prob，表示数据需要保留的概率。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>取值范围为(0, 1]。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tockens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示sliding window的左边界。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tockens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示sliding window的右边界。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_num（int）</td>
      <td>必要属性</td>
      <td>代表单卡的head个数，即输入query的N轴长度。</td>
      <td>取值必须和传入的query中的N值保持一致。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout（string）</td>
      <td>必要属性</td>
      <td>
        <ul>
          <li>代表输入query、key、value的数据排布格式。</li>
          <li>默认值为空。</li>
        </ul>
      </td>
      <td>支持BSH、SBH、BNSD、BSND、TND。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>预留参数，暂未使用。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示sparse的模式。0表示defaultMask模式，1表示allMask，2表示leftUpCausal，3表示rightDownCausal，4表示band，5表示prefix，6表示prefixCompress，7表示rightDownCausalBand，8表示bandLeftUpCausal。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>支持配置值为0~8。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pse_type（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>控制add与mul的执行次序。</li>
          <li>默认值为1。</li>
        </ul>
      </td>
      <td>支持配置值为0、1、2、3。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>Dropout随机种子，keep_prob小于1.0且无外部传入drop_mask时，根据seed和offset生成DropoutMask。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>Dropout偏移量。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_dtype（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>输出精度控制。值为0表示dq等输出是FLOAT16，为1表示是BFLOAT16。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_in_layout（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>控制softmax_max、softmax_sum的实际数据排布。</li>
          <li>默认值为空。</li>
        </ul>
      </td>
      <td>shape为TND、实际数据排布为NTD时传"same_as_input"，TND时传""。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dq（tensor）</td>
      <td>输出</td>
      <td>公式中的dQ，表示query的梯度。</td>
      <td>数据类型与query保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dk（tensor）</td>
      <td>输出</td>
      <td>公式中的dK，表示key的梯度。</td>
      <td>数据类型与key保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dv（tensor）</td>
      <td>输出</td>
      <td>公式中的dV，表示value的梯度。</td>
      <td>数据类型与value保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dpse（tensor）</td>
      <td>输出</td>
      <td>表示pse的梯度。</td>
      <td>暂未使用，传入的shape必须和pse_shape保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>0、4</td>
    </tr>
    <tr>
      <td>dq_rope（tensor）</td>
      <td>输出</td>
      <td>表示query_rope的梯度。</td>
      <td>数据类型与dq保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dk_rope（tensor）</td>
      <td>输出</td>
      <td>表示key_rope的梯度。</td>
      <td>数据类型与dk保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
    <tr>
      <td>dsink（tensor）</td>
      <td>输出</td>
      <td>公式中的dSink，表示sink的梯度。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 确定性计算：
  - Ascendir_FlashAttentionScoreGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 输入query、key、value、dy的约束：
  - B：batchsize必须相等。
  - input_layout必须一致。
  - D：Head-Dim必须满足query和key的D相等，value和dy的D相等，并且query和key的D大于等于value和dy的D。
- 支持输入query/dy的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。
- 输入query_rope与query的输入shape仅在D维度不同，其他shape参数应该相同；输入key_rope与key的输入shape仅在D维度不同，其他shape参数应该相同。
- 关于数据shape的约束，以input_layout的TND为例，其中：
    - B：取值范围为1\~2K。带prefix的时候B最大支持1K。
    - N：取值范围为1\~256。
    - S：取值范围为1\~1M。
    - D：取值范围为1\~768。
    - KeepProb：取值范围为(0, 1]。
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- 关于softmax_max与softmax_sum参数的约束：输入格式固定为[B, N, S, 8]，TND的输入格式除外，此时为[N, T, 8]，注：T=B*S。
- softmax_max、softmax_sum数据排布为TND时，softmax_in_layout需要为"same_as_input"。
- head_num的取值必须和传入的query中的N值保持一致。
- prefix稀疏计算仅支持压缩场景，sparse_mode=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。当sparse_mode=5、prefix的N > Skv或prefix不传时执行全计算，sparse_mode=6要求prefix必传。
- sparse_mode=7时，不支持可选输入pse_shift。
- sparse_mode=8时，当每个sequence的q、kv等长时支持可选输入pse_shift，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actual_seq_qlen[0] - actual_seq_kvlen[0] + q_start_idx - kv_start_idx == 0（本功能属实验性功能）。
- actual_seq_qlen输入支持某个Batch上的S长度为0，此时不支持可选输入pse_shift。
- TND格式下，支持尾部部分Batch不参与计算，此时actual_seq_qlen和actual_seq_kvlen尾部传入对应个数的0即可。假设真实S长度为[2, 3, 4, 5, 6]，若希望最后两个Batch不参与计算，则传入的actual_seq_qlen为[2, 3, 4, 0, 0]。此时若需要传入prefix，其尾部也需要传入同等数量的0，例如[1, 1, 1, 0, 0]。
- sink不为None时，query与key等输入tensor仅支持float16和bfloat16两种类型。
- pse_type为2或3的时候，当前只支持Sq和Skv等长。
- sparse_mode约束如下:
  - 当所有的atten_mask的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量。
  - 配置为1、2、3、5时，用户配置的pre_tockens、next_tockens不会生效。
  - 配置为0、4时，须保证atten_mask与pre_tockens、next_tockens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode_introduction.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- <term>Ascend 950PR&950DT系列产品</term>：
    - query_rope和key_rope非空时，query和key的每头D必须为128，value和dy的每头D必须为128，query_rope和key_rope的每头D必须为64。
    - seed和offset只在keep_prob小于1.0时生效，否则不生效。
    - keep_prob小于1.0时，若drop_mask非nullptr，则使用输入的drop_mask；否则使用seed和offset生成的drop_mask。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)，完整代码见[test_geir_flash_attention_score_grad](../examples/test_geir_flash_attention_score_grad.cpp)。

```c++
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_flash_attention_score_grad.cpp
 * \brief GE graph construction sample for FlashAttentionScoreGrad (SBH layout, fp16).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/flash_attention_score_grad_proto.h"

#define FAILED (-1)
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kQHeadNum = 1;
constexpr int64_t kKvHeadNum = 1;
constexpr int64_t kQSeqLen = 256;
constexpr int64_t kKvSeqLen = 256;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kH1 = kQHeadNum * kHeadDim;
constexpr int64_t kH2 = kKvHeadNum * kHeadDim;
constexpr double kScale = 0.08838834764831845; // 1.0 / sqrt(128)
constexpr float kRtol = 3e-3f;
constexpr float kAtol = 3e-3f;

std::string GetGeError()
{
    const AscendString errorMessage = GEGetErrorMsgV2();
    return errorMessage.GetString() == nullptr ? "" : errorMessage.GetString();
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// Deterministic value in [-0.45, 0.45]; the magnitude stays in the normal fp16 range.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return ((static_cast<float>(seed & 0xFFU) / 255.0f) - 0.5f) * 0.9f;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return sign; // flush tiny magnitudes to zero
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U); // clamp to infinity
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value & 0x7C00U) >> 10;
    const uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -24) * ((sign != 0) ? -1.0f : 1.0f);
    }
    const uint32_t bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
int32_t AddInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
                 int32_t index, const vector<int64_t>& shape, DataType dtype, const vector<T>& hostData)
{
    auto dataOp = op::Data(name).set_attr_index(index - 1);
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    const int64_t elementCount = GetShapeSize(shape);
    CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
              LOG_PRINT("[ERROR] %s data size mismatch\n", name.c_str());
              return FAILED);
    Tensor tensor;
    auto ret = tensor.SetTensorDesc(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetTensorDesc failed\n", name.c_str()); return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetData failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_input_desc_x failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_output_desc_y failed\n", name.c_str()); return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Graph::AddOp failed for %s\n", name.c_str()); return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

struct ForwardStats {
    vector<float> softmaxMax;  // [S1]
    vector<float> softmaxSum;  // [S1]
    vector<float> attentionIn; // [S1, D] fp32 exact reference of the forward output
};

void ComputeForward(const vector<float>& q, const vector<float>& k, const vector<float>& v, ForwardStats& stats)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    stats.softmaxMax.assign(s1, 0.0f);
    stats.softmaxSum.assign(s1, 0.0f);
    stats.attentionIn.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
            }
            score[j] = static_cast<float>(kScale) * dot;
        }
        float maxScore = score[0];
        for (int64_t j = 1; j < s2; ++j) {
            maxScore = std::fmax(maxScore, score[j]);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < s2; ++j) {
            score[j] = std::exp(score[j] - maxScore);
            sum += score[j];
        }
        stats.softmaxMax[i] = maxScore;
        stats.softmaxSum[i] = sum;
        for (int64_t t = 0; t < d; ++t) {
            float y = 0.0f;
            for (int64_t j = 0; j < s2; ++j) {
                y += (score[j] / sum) * v[j * d + t];
            }
            stats.attentionIn[i * d + t] = y;
        }
    }
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& q, vector<float>& k, vector<float>& v, vector<float>& dy, vector<float>& statsMax,
                   vector<float>& statsSum, vector<float>& attentionIn)
{
    const int64_t qSize = kQSeqLen * kH1;
    const int64_t kvSize = kKvSeqLen * kH2;
    q.resize(qSize);
    k.resize(kvSize);
    v.resize(kvSize);
    dy.resize(qSize);
    for (int64_t i = 0; i < qSize; ++i) {
        q[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
        dy[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        k[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        v[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    ForwardStats stats;
    ComputeForward(q, k, v, stats);
    statsMax = stats.softmaxMax;
    statsSum = stats.softmaxSum;
    attentionIn = stats.attentionIn;

    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};

    vector<uint16_t> qHalf(q.size());
    vector<uint16_t> kHalf(k.size());
    vector<uint16_t> vHalf(v.size());
    vector<uint16_t> dyHalf(dy.size());
    vector<uint16_t> outHalf(attentionIn.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qHalf[i] = FloatToHalf(q[i]);
        dyHalf[i] = FloatToHalf(dy[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kHalf[i] = FloatToHalf(k[i]);
        vHalf[i] = FloatToHalf(v[i]);
    }
    for (size_t i = 0; i < attentionIn.size(); ++i) {
        outHalf[i] = FloatToHalf(attentionIn[i]);
    }
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0);
    // softmax statistics are [B, N, S, 8]; every lane holds the same per-row value.
    vector<float> maxLanes;
    vector<float> sumLanes;
    maxLanes.reserve(statsMax.size() * 8);
    sumLanes.reserve(statsSum.size() * 8);
    for (size_t i = 0; i < statsMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxLanes.push_back(statsMax[i]);
            sumLanes.push_back(statsSum[i]);
        }
    }

    // Input order follows the IR definition: query, key, value, dy, atten_mask, softmax_max, softmax_sum,
    // attention_in, prefix, q_start_idx, kv_start_idx.
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "dy", 4, qShape, DT_FLOAT16, dyHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "atten_mask", 5, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_max", 6, statShape, DT_FLOAT, maxLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_sum", 7, statShape, DT_FLOAT, sumLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "attention_in", 8, qShape, DT_FLOAT16, outHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "prefix", 9, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "q_start_idx", 10, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "kv_start_idx", 11, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);

    auto node = op::FlashAttentionScoreGrad("flash_attention_score_grad");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_dy(inputOps[3]);
    node.set_input_atten_mask(inputOps[4]);
    node.set_input_softmax_max(inputOps[5]);
    node.set_input_softmax_sum(inputOps[6]);
    node.set_input_attention_in(inputOps[7]);
    node.set_input_prefix(inputOps[8]);
    node.set_input_q_start_idx(inputOps[9]);
    node.set_input_kv_start_idx(inputOps[10]);

    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(65536);
    node.set_attr_next_tockens(65536);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(1);
    node.set_attr_seed(0);
    node.set_attr_offset(0);

    TensorDesc dqDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dkDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dvDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp16Desc(Shape({0}), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp32Desc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_dq(dqDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk(dkDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dv(dvDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dpse(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dq_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dsink(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

void ComputeBackward(const vector<float>& q, const vector<float>& k, const vector<float>& v, const vector<float>& dy,
                     const ForwardStats& stats, vector<float>& dq, vector<float>& dk, vector<float>& dv)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    dq.assign(s1 * d, 0.0f);
    dk.assign(s2 * d, 0.0f);
    dv.assign(s2 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        vector<float> prob(s2);
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
            }
            const float score = static_cast<float>(kScale) * dot;
            prob[j] = std::exp(score - stats.softmaxMax[i]) / stats.softmaxSum[i];
        }
        for (int64_t j = 0; j < s2; ++j) {
            float dP = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dP += dy[i * d + t] * v[j * d + t];
                dv[j * d + t] += prob[j] * dy[i * d + t];
            }
            float dYdotY = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dYdotY += dy[i * d + t] * stats.attentionIn[i * d + t];
            }
            const float dS = prob[j] * (dP - dYdotY);
            for (int64_t t = 0; t < d; ++t) {
                dq[i * d + t] += static_cast<float>(kScale) * dS * k[j * d + t];
                dk[j * d + t] += static_cast<float>(kScale) * dS * q[i * d + t];
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT16 &&
                  tensor.GetSize() == count * sizeof(uint16_t),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<uint16_t> halfValues(count);
    std::memcpy(halfValues.data(), tensor.GetData(), count * sizeof(uint16_t));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = HalfToFloat(halfValues[i]);
        const float tolerance = kAtol + kRtol * std::fabs(expected[i]);
        const float err = std::fabs(value - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, value, expected[i],
                          tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& q, const vector<float>& k,
                        const vector<float>& v, const vector<float>& dy, const ForwardStats& stats)
{
    CHECK_RET(outputs.size() == 7, LOG_PRINT("[CHECK] FAIL: expected 7 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> dq;
    vector<float> dk;
    vector<float> dv;
    ComputeBackward(q, k, v, dy, stats, dq, dk, dv);
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    bool ok = true;
    ok = CheckOutput(outputs[0], "dq", qShape, dq) && ok;
    ok = CheckOutput(outputs[1], "dk", kvShape, dk) && ok;
    ok = CheckOutput(outputs[2], "dv", kvShape, dv) && ok;
    // dpse/dq_rope/dk_rope/dsink are optional outputs and stay empty because the matching inputs are not bound;
    // their content is not validated.
    LOG_PRINT("[CHECK] total: %s\n", ok ? "PASS" : "FAIL");
    return ok ? SUCCESS : FAILED;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = GEInitialize(globalOptions);
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEInitialize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);

    Graph graph("flash_attention_score_grad_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> q;
    vector<float> k;
    vector<float> v;
    vector<float> dy;
    vector<float> statsMax;
    vector<float> statsSum;
    vector<float> attentionIn;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, q, k, v, dy, statsMax, statsSum, attentionIn);
    if (ret == SUCCESS) {
        ForwardStats stats;
        stats.softmaxMax = statsMax;
        stats.softmaxSum = statsSum;
        stats.attentionIn = attentionIn;
        graph.SetInputs(inputOps).SetOutputs(outputOps);
        std::map<AscendString, AscendString> buildOptions;
        Session* session = new (std::nothrow) Session(buildOptions);
        CHECK_RET(session != nullptr, LOG_PRINT("[ERROR] Session allocation failed.\n"); return FAILED);
        constexpr uint32_t graphId = 0;
        std::map<AscendString, AscendString> graphOptions;
        auto addRet = session->AddGraph(graphId, graph, graphOptions);
        CHECK_RET(addRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::AddGraph failed, status=%u, error=%s\n", addRet, GetGeError().c_str());
                  delete session; return FAILED);
        vector<Tensor> outputTensors;
        auto runRet = session->RunGraph(graphId, inputTensors, outputTensors);
        CHECK_RET(runRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::RunGraph failed, status=%u, error=%s\n", runRet, GetGeError().c_str());
                  delete session; return FAILED);
        LOG_PRINT("FlashAttentionScoreGrad graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, q, k, v, dy, stats);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
