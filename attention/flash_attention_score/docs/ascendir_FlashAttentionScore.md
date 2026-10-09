# Ascendir_FlashAttentionScore

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

- 算子功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算：

    - pseType=1时，需要先add再mul。
    - pseType≠1时，需要先mul再add。

- 计算公式：

  注意力的正向计算公式如下：

    - pseType=1时，公式如下：
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(pse+(query*d\_scale\_q)*(key*d\_scale\_k)^T), atten\_mask)), keep\_prob)*(value*d\_scale\_v)
      $$

    - pseType≠1时，公式如下：
      $$
      attention\_out=Dropout(Softmax(Mask(scale*((query*d\_scale\_q)*(key*d\_scale\_k)^T) + pse),atten\_mask),keep\_prob)*(value*d\_scale\_v)
      $$

## Ascend IR定义

Ascend IR定义所在头文件路径为[op_graph/flash_attention_score_proto.h](../op_graph/flash_attention_score_proto.h)。

```cpp
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
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
    .ATTR(softmax_out_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScore)
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
      <td>real_shift（tensor）</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>数据类型与query的数据类型一致，该参数需要与pse_type配套使用。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv]、[B,N,1,Skv]、[1,N,Sq,Skv]、[B,N,1024,Skv]、[1,N,1024,Skv]、[B,N]、[N]</td>
    </tr>
    <tr>
      <td>drop_mask（tensor）</td>
      <td>可选输入</td>
      <td>公式中的Dropout，表示数据丢弃掩码。取值为1代表保留该数据，为0代表丢弃该数据。</td>
      <td>-</td>
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
      <td>实际Q序列长度。</td>
      <td>使用时input_layout需设置为TND。例如：q的seqlen为[2,2,2,2,2]，该参数需设置为[2,4,6,8,10]。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>actual_seq_kvlen（tensor）</td>
      <td>可选输入</td>
      <td>实际KV序列长度。</td>
      <td>使用时input_layout需设置为TND。例如：kv的seqlen为[2,2,2,2,2]，该参数需设置为[2,4,6,8,10]。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>q_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>Q起始索引。</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>kv_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>KV起始索引。</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0、1</td>
    </tr>
    <tr>
      <td>d_scale_q（tensor）</td>
      <td>可选输入</td>
      <td>公式中的d_scale_q，FP8场景下query的全量化参数。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d_scale_k（tensor）</td>
      <td>可选输入</td>
      <td>公式中的d_scale_k，FP8场景下key的全量化参数。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d_scale_v（tensor）</td>
      <td>可选输入</td>
      <td>公式中的d_scale_v，FP8场景下value的全量化参数。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_rope（tensor）</td>
      <td>可选输入</td>
      <td>Q的RoPE旋转位置编码输入。</td>
      <td>-</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope（tensor）</td>
      <td>可选输入</td>
      <td>K的RoPE旋转位置编码输入。</td>
      <td>-</td>
      <td>float8_e5m2、float8_e4m3fn、float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sink（tensor）</td>
      <td>可选输入</td>
      <td>Sinking参数。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p_scale（tensor）</td>
      <td>可选输入</td>
      <td>P的缩放因子。</td>
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
      <td>-</td>
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
          <li>用于提升精度。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>0、1为保留值；2支持无效行计算。默认配置为0即可。</td>
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
          <li>Dropout随机种子。</li>
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
          <li>输出精度控制。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_out_layout（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>Softmax输出数据排布格式。</li>
          <li>默认值为空。</li>
        </ul>
      </td>
      <td>支持配置值为""（空）、"same_as_input"。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_max（tensor）</td>
      <td>输出</td>
      <td>Softmax计算的Max中间结果，用于反向计算。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>softmax_sum（tensor）</td>
      <td>输出</td>
      <td>Softmax计算的Sum中间结果，用于反向计算。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>softmax_out（tensor）</td>
      <td>输出</td>
      <td>预留参数，暂未使用。</td>
      <td>-</td>
      <td>float16、bf16、float32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out（tensor）</td>
      <td>输出</td>
      <td>公式中的attention_out。</td>
      <td>数据类型和shape类型与query保持一致。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>[BNSD]、[BSND]、[BSH]、[SBH]、[TND]</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 确定性计算：
  - Ascendir_FlashAttentionScore默认确定性实现。
- 输入query、key、value的约束：
  - B：batchsize必须相等。
  - D：Head-Dim必须满足(qD == kD && kD >= vD)。
  - input_layout必须一致。
- 输入query_rope与query的输入shape仅在D维度不同，其他shape参数应该相同。
- 输入key_rope与key的输入shape仅在D维度不同，其他shape参数应该相同。
- MLA concat场景：仅支持dSize=128，dSizeRope=64。
- 关于数据shape的约束，以input_layout的TND、BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    - T(B*S)：取值范围为1\~1M；TND格式下actual_seq_qlen支持的最大长度为20000。
    - B：取值范围为1\~2M。带prefix的时候B最大支持2K。
    - N：取值范围为1\~256。
    - S：取值范围为1\~1M。
    - D：取值范围为1\~768。
- 空Tensor场景说明：
    - 当B、N1(head_num)、S1(query的sequence length)中任一维度为0时，算子不执行任何计算，直接返回成功，workspaceSize为0。
    - 当N2(key/value的head数)、S2(key/value的sequence length)、D(Head-Dim)中任一维度为0时，query/key/value为空Tensor（shapeSize为0），但softmax_max和softmax_sum的输出shape可能非空（其shape不依赖D维度），算子将执行空输入处理流程，对非空输出进行初始化。
    - 上述空Tensor场景下，算子正常返回，不报错。
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- inner_precise: 当前0、1为保留配置值，2为开启无效行计算，其功能是避免在计算过程中存在整行mask进而导致精度有损失，但是该配置会导致性能下降。如果算子可判断出存在无效行场景，会自动开启无效行计算，例如sparse_mode为3，Sq > Skv场景。
- pse_type各个取值含义

    | pse_type    | 含义                              |      备注   |
    | ----------- | --------------------------------- | ----------|
    | 0           | 调用算子时传入pse先mul再add              | - |
    | 1           | 调用算子时传入pse先add再mul              | 跟FlashAttentionScore实现一致。 |
    | 2           | 算子自动生成pse先mul再add              | - |
    | 3           | 算子自动生成pse先mul再add再sqrt         | - |

- pse_type为2或3的时候，当前只支持Sq和Skv等长。
- sink不为None时，query与key等输入tensor仅支持float16和bfloat16两种类型。
- sparse_mode约束如下:
  - 当所有的atten_mask的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量。
  - 配置为1、2、3、5时，用户配置的pre_tockens、next_tockens不会生效。
  - 配置为0、4时，须保证atten_mask与pre_tockens、next_tockens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode_introduction.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- band场景，pre_tockens和next_tockens之间必须要有交集。
- prefix稀疏计算场景，场景包括sequence长度相等的场景下sparse_mode=5、sparse_mode=6；sequence长度不相等的场景下sparse_mode=6。这两种场景下，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]；当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。当sparse_mode=5、prefix的N > Skv或prefix不传时执行全计算，sparse_mode=6要求prefix必传。
- real_shift：Sq大于1024时如果配置BNHS、1NHS，需要Sq和Skv等长。
- actual_seq_qlen输入支持某个Batch上的S长度为0，此时不支持可选输入real_shift。
- TND场景下atten_mask输入不支持补pad，即atten_mask中不能存在某一行全1的场景。
- 支持actual_seq_qlen中某个Batch上的S长度为0；如果存在S为0的情况，不支持pse输入，
  假设真实的S长度为\[2,2,0,2,2\]，则传入的actual_seq_qlen为\[2,4,4,6,8\]。
- <term>Ascend 950PR&950DT系列产品</term>：
    - seed和offset只在keep_prob小于1.0时生效，否则不生效。
    - keep_prob小于1.0时，若drop_mask非nullptr，则使用输入的drop_mask；否则使用seed和offset生成的drop_mask。
- TND格式下，支持尾部部分Batch不参与计算，此时actual_seq_qlen和actual_seq_kvlen尾部传入对应个数的0即可。假设真实S长度为[2, 3, 4, 5, 6]，若希望最后两个Batch不参与计算，则传入的actual_seq_qlen为[2, 3, 4, 0, 0]。此时若需要传入prefix，其尾部也需要传入同等数量的0，例如[1, 1, 1, 0, 0]。
- softmax_out_layout支持传入：空字符串、"same_as_input"。
- real_shift：如果Sq大于1024且每个batch的Sq与Skv等长且是sparse_mode为0、2、3的下三角掩码场景，可开启alibi位置编码压缩，此时只需要输入原始PSE最后1024行，实现内存优化，即alibi_compress = ori_pse[:, :, -1024:, :]，具体如下：
  - 参数每个batch不相同时，shape为BNHSkv(H=1024)。
  - 每个batch相同时，shape为1NHSkv(H=1024)。
  - 如果pse_type为2或3的时候，数据类型需为FLOAT32,对应shape支持范围是[B,N]或[N]。
  - 如果不开启该参数，real_shift需要传入nullptr，pse_type需要传入1。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)，完整代码见[test_geir_flash_attention_score](../examples/test_geir_flash_attention_score.cpp)。

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
 * \file test_geir_flash_attention_score.cpp
 * \brief GE graph construction sample for FlashAttentionScore (SBH layout, fp32).
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

#include "../op_graph/flash_attention_score_proto.h"

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

// Deterministic pseudo-random value in [-0.5, 0.5), same value reused by the CPU reference.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return (static_cast<float>(seed & 0xFFFFU) / 65536.0f) - 0.5f;
}

template <typename T>
int32_t AddDataInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
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
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetTensorDesc failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_input_desc_x failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_output_desc_y failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Graph::AddOp failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData)
{
    auto node = op::FlashAttentionScore("flash_attention_score");
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> idxShape = {1};

    const int64_t qSize = GetShapeSize(qShape);
    const int64_t kvSize = GetShapeSize(kvShape);
    qData.resize(qSize);
    kData.resize(kvSize);
    vData.resize(kvSize);
    for (int64_t i = 0; i < qSize; ++i) {
        qData[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        kData[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        vData[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0); // 0: the position attends
    const vector<int64_t> prefixData = {0};
    const vector<int64_t> qStartIdxData = {0};
    const vector<int64_t> kvStartIdxData = {0};

    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT, qData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT, kData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT, vData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "atten_mask", 4, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "prefix", 5, idxShape, DT_INT64, prefixData) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "q_start_idx", 6, idxShape, DT_INT64, qStartIdxData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "kv_start_idx", 7, idxShape, DT_INT64, kvStartIdxData) == SUCCESS,
        return FAILED);

    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_atten_mask(inputOps[3]);
    node.set_input_prefix(inputOps[4]);
    node.set_input_q_start_idx(inputOps[5]);
    node.set_input_kv_start_idx(inputOps[6]);
    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(2147483647);
    node.set_attr_next_tockens(2147483647);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(0);

    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};
    TensorDesc softmaxMaxDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxSumDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_softmax_max(softmaxMaxDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_sum(softmaxSumDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_out(softmaxOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference: softmax(scale * Q @ K^T) @ V with SBH layout, N1 == N2 == 1, B == 1.
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      vector<float>& softmaxMax, vector<float>& softmaxSum, vector<float>& attentionOut)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    softmaxMax.assign(s1, 0.0f);
    softmaxSum.assign(s1, 0.0f);
    attentionOut.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += qData[i * d + t] * kData[j * d + t];
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
        softmaxMax[i] = maxScore;
        softmaxSum[i] = sum;
        for (int64_t j = 0; j < s2; ++j) {
            const float prob = score[j] / sum;
            for (int64_t t = 0; t < d; ++t) {
                attentionOut[i * d + t] += prob * vData[j * d + t];
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected,
                 float rtol, float atol)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT &&
                  tensor.GetSize() == count * sizeof(float),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<float> values(count);
    std::memcpy(values.data(), tensor.GetData(), count * sizeof(float));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float tolerance = atol + rtol * std::fabs(expected[i]);
        const float err = std::fabs(values[i] - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, values[i],
                          expected[i], tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& qData, const vector<float>& kData,
                        const vector<float>& vData)
{
    CHECK_RET(outputs.size() == 4, LOG_PRINT("[CHECK] FAIL: expected 4 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> softmaxMax;
    vector<float> softmaxSum;
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, softmaxMax, softmaxSum, attentionOut);
    // softmax_max/softmax_sum are stored as [B, N, S, 8]; all 8 lanes hold the same per-row statistic.
    const int64_t s1 = kQSeqLen;
    vector<float> maxExpanded;
    vector<float> sumExpanded;
    maxExpanded.reserve(softmaxMax.size() * 8);
    sumExpanded.reserve(softmaxSum.size() * 8);
    for (size_t i = 0; i < softmaxMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxExpanded.push_back(softmaxMax[i]);
            sumExpanded.push_back(softmaxSum[i]);
        }
    }
    bool ok = true;
    ok = CheckOutput(outputs[0], "softmax_max", {kBatch, kQHeadNum, kQSeqLen, 8}, maxExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[1], "softmax_sum", {kBatch, kQHeadNum, kQSeqLen, 8}, sumExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[3], "attention_out", {kQSeqLen, kBatch, kH1}, attentionOut, 1e-3f, 1e-3f) && ok;
    // softmax_out is a reserved output that the kernel does not write; its content is not validated.
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

    Graph graph("flash_attention_score_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> qData;
    vector<float> kData;
    vector<float> vData;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, qData, kData, vData);
    if (ret == SUCCESS) {
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
        LOG_PRINT("FlashAttentionScore graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
