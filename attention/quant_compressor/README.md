# QuantCompressor

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：QuantCompressor是推理场景下SMLA和QLI的前处理算子，是[Compressor](../compressor/README.md)的量化版本。用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，QuantCompressor可以有效地减少计算开销。与Compressor的区别在于：输入$x$、$W^{KV}$、$W^{Gate}$为HIFLOAT8量化数据，直接以HIFLOAT8参与Matmul运算（硬件原生支持），再对Matmul输出的FLOAT32结果乘以合并后的缩放因子进行反量化，从而降低显存占用与搬运开销。主要计算过程为：
    1. Matmul与反量化：将HIFLOAT8量化的输入$X$与$W^{KV}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wkv\_descale$完成反量化得到$kv\_state$；将$X$与$W^{Gate}$做Matmul运算得到FLOAT32结果，再乘以合并缩放因子$x\_descale \cdot wgate\_descale$完成反量化得到$score\_state$。其中x_descale为per-tensor缩放（单个标量），wkv_descale与wgate_descale为per-channel缩放（通道数为coff\*D），合并后仍为per-channel缩放。$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$按压缩比分组并与$Ape$相加，然后进行softmax运算，将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。

- 计算公式：

    1. 计算矩阵乘法与反量化（quant_mode=1，A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL）：

    使用HIFLOAT8输入进行Matmul，输出FLOAT32结果后乘以合并缩放因子（x_descale与wkv/wgate_descale的乘积）完成反量化：

    $$
    C4A：\left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
    $$

    $$
    C128A：\left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
    $$

    其中x_descale为per-tensor标量（shape为[1,]），wkv_descale、wgate_descale为per-channel向量（shape为[coff\*D,]，沿权重输出通道维度缩放），两者相乘后合并为per-channel缩放因子。

    2. 计算分组加法：

    $$
    C4A：score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
    $$

    3. 计算分组Softmax：

    $$
    C4A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
    $$

    4. 计算Hadamard乘积：

    $$
    C4A：(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A：S_H = S_i^\prime \odot kv\_state;
    $$

    5. 沿着压缩轴分组求和：

    $$
    C4A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
     C128A：C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 220px">
  <col style="width: 170px">
  <col style="width: 700px">
  <col style="width: 286px">
  <col style="width: 200px">
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">X</span>，表示原始不经压缩的数据，HIFLOAT8量化输入。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wkv</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">W<sup>KV</sup></span>，表示kv压缩权重，HIFLOAT8量化输入。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wgate</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">W<sup>Gate</sup></span>，表示gate压缩权重，HIFLOAT8量化输入。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>state_cache</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">[kv_state, score_state]</span>，表示kv_state和score_state的历史数据。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ape</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">Ape</span>，表示positional biases。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_descale</td>
      <td>可选输入</td>
      <td>x的反量化缩放因子，per-tensor缩放。quant_mode=1时必选。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wkv_descale</td>
      <td>可选输入</td>
      <td>wkv的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wgate_descale</td>
      <td>可选输入</td>
      <td>wgate的反量化缩放因子，per-channel缩放，通道数为coff\*D。quant_mode=1时必选。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>state_block_table</td>
      <td>可选输入</td>
      <td>表示state_cache存储使用的block映射表。<br>当其中元素的值为0时，表示当前位置无需进行更新state_cache操作。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens</td>
      <td>可选输入</td>
      <td>表示不同Batch中的有效token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused</td>
      <td>可选输入</td>
      <td>表示不同Batch中实际参与压缩的token数。<br>如果指定为None时，表示和每个Batch上的Sequence Length长度相同。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>start_pos</td>
      <td>可选输入</td>
      <td>表示计算起始位置。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式。取值范围为[1]，1表示A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL（HIFLOAT8输入，x按per-tensor缩放、wkv/wgate按per-channel缩放反量化）。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>属性</td>
      <td>用于稀疏计算，表示数据压缩率，取值范围为[2, 128]内的整数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>coff</td>
      <td>可选属性</td>
      <td>表示是否进行overlap数据重排。<br>coff=1：无需进行overlap数据重排，coff=2：需要进行overlap数据重排。<br>默认值为1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>可选属性</td>
      <td>表示state_cache的存储模式。<br>cache_mode=1：连续buffer，cache_mode=2：循环buffer。<br>默认值1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>state_cache_stride_dim0</td>
      <td>可选属性</td>
      <td>表示state_cache的0轴stride。默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>输出</td>
      <td>表示压缩后的数据。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - state\_cache支持输入shape[block_num,block_size,2*coff* D]，要求block_num>0，cache_mode=2时，需要满足block_size >= coff * cmp_ratio + S - 1。
    - ape支持输入shape[cmp_ratio,coff* D]
    - x\_descale支持输入shape[1,]，per-tensor缩放。
    - wkv\_descale支持输入shape[coff* D,]，per-channel缩放。
    - wgate\_descale支持输入shape[coff* D,]，per-channel缩放。
    - start\_pos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cu\_seqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须位0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
        - cache_mode=1时，state\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n])。cache_mode=2时，state\_block\_table支持输入shape[B]。
        - cmp\_kv，输出shape为[min(T,T//cmp_ratio+B),D]：compressed_tokens + compressed_tokens + ... + compressed_tokens + pad。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cu\_seqlens，参数必须为空。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - cache_mode=1时，state\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+S。cache_mode=2时，state\_block\_table支持输入shape[B]。
        - cmp\_kv，输出shape为[B,ceil(S/cmp_ratio),D]：(compressed_tokens+pad0) + (compressed_tokens+pad1) + ...  + (compressed_tokens+padN)。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
      - 只支持B、S为0
      - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：
      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>S</th>
      <th>H</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>100</td>
      <td>65525</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>25</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>131072</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      </tbody>
      </table>
      </div>
- 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下state_cache不做更新，输出cmp_kv为空tensor。
- 输入属性限制：
  - quant_mode取值为1（A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL），此时x_descale、wkv_descale、wgate_descale为必选输入。
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持block_size为1~1024。
  - 支持如下三种典型组合场景：
      - C4A: D=512, coff=2, cmp_ratio=4；
      - C4Li: D=128, coff=2, cmp_ratio=4；
      - C128A: D=512, coff=1, cmp_ratio=128。
- 确定性计算与batch一致性：
  - 默认确定性实现，相同输入多次调用结果一致。
  - <term>Ascend 950PR/Ascend 950DT</term>：batch一致性：通过aclrtSetSysParamOpt()配置ACL_OPT_DETERMINISTIC为3开启batch一致性，开启后可以满足计算结果和所在批次大小、位置无关。

## 调用说明

  | 调用方式 | 样例代码 | 说明 |
  | ------- | ------- | ---- |
  | aclnn API | [test_aclnn_quant_compressor](./examples/arch35/test_aclnn_quant_compressor.cpp) | 通过[aclnnQuantCompressor](./docs/aclnnQuantCompressor.md)接口调用QuantCompressor算子。 |
  | PyTorch API | - | 通过[quant_compressor](../../torch_extension/cann_ops_transformer/docs/zh/quant_compressor.md)接口调用QuantCompressor算子。 |
