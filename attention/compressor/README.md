# Compressor

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：Compressor是推理场景下SMLA和QLI的前处理算子，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，Compressor可以有效地减少计算开销。主要计算过程为：
    1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens完成更新。
    2. 在coff为2的情况下对$kv\_state$和$score\_state$进行数据重排。
    3. 对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算。

- 计算公式：

    1. 计算矩阵乘法：

        $$
        C4A：\left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right], \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right];
        $$

        $$
        C128A：\left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]
        $$

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

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 250px">
  <col style="width: 170px">
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
    <td>公式中的$X$，表示原始不经压缩的数据。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wkv</td>
    <td>输入</td>
    <td>公式中的$W^{KV}$，表示kv压缩权重。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wgate</td>
    <td>输入</td>
    <td>公式中的$W^{Gate}$，表示gate压缩权重。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>state_cache</td>
    <td>输入</td>
    <td>公式中的$\left[kv\_state, score\_state\right]$，表示kv_state和score_state的历史数据。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ape</td>
    <td>输入</td>
    <td>公式中的$Ape$，表示positional biases。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cmp_ratio</td>
    <td>属性</td>
    <td>用于稀疏计算，表示数据压缩率，取值范围为[2, 128]内的整数。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>state_block_table</td>
    <td>可选输入</td>
    <td>表示state_cache存储使用的block映射表。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cu_seqlens</td>
    <td>可选输入</td>
    <td>表示不同Batch中的有效token数。该参数为前缀和数组，后一个元素的值必须大于等于前一个元素的值，且第一位必须为0。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seqused</td>
    <td>可选输入</td>
    <td>表示不同Batch中实际参与压缩的token数。指定为None时，表示和每个Batch上的Sequence Length长度相同。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>start_pos</td>
    <td>可选输入</td>
    <td>表示计算起始位置。指定为None时，表示从0开始进行计算。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>coff</td>
    <td>可选属性</td>
    <td>表示是否进行overlap数据重排。coff=1：无需进行overlap数据重排，coff=2：需要进行overlap数据重排。默认值为1。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cache_mode</td>
    <td>可选属性</td>
    <td>表示state_cache的存储模式。cache_mode=1：连续buffer，cache_mode=2：循环buffer。默认值为1。</td>
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
    <td>grad_enabled</td>
    <td>可选属性</td>
    <td>是否导出softmax_score/kv中间结果（供反向CompressorGrad使用）。true时导出，false时不导出。默认值为false。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cmp_kv</td>
    <td>输出</td>
    <td>表示压缩后的数据。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmax_score</td>
    <td>输出</td>
    <td>压缩块的softmax权重中间结果。仅在grad_enabled为true时输出有效。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kv</td>
    <td>输出</td>
    <td>softmax权重与kv_state的Hadamard乘积中间结果。仅在grad_enabled为true时输出有效。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：
  - cache_mode不支持输入2，且state_cache不支持0轴非连续。
  - cmp_ratio仅支持2/4/8/16/32/64/128。

## 约束说明

- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape为[coff*D,H]。
    - wgate支持输入shape为[coff*D,H]。
    - state_cache支持输入shape为[block_num, block_size, 2*coff*D]，要求block_num>0，cache_mode=2时，需要满足block_size >= coff*cmp_ratio + S - 1。
    - ape支持输入shape为[cmp_ratio, coff*D]。
    - start_pos支持输入shape为[B,]。
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]：
        - cu_seqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须为0。
        - seqused，支持输入shape为[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu_seqlens[n+1] - cu_seqlens[n]，且不小于0。
        - cache_mode=1时，state_block_table支持输入shape为[B, ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n])。cache_mode=2时，state_block_table支持输入shape为[B]。
        - cmp_kv，输出shape为[min(T, T/cmp_ratio+B), D]：compressed_tokens + compressed_tokens + ... + compressed_tokens + pad。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]：
        - cu_seqlens，参数必须为空。
        - seqused，支持输入shape为[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - cache_mode=1时，state_block_table支持输入shape为[B, ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start_pos)+S。cache_mode=2时，state_block_table支持输入shape为[B]。
        - cmp_kv，输出shape为[B, ceil(S/cmp_ratio), D]：(compressed_tokens+pad0) + (compressed_tokens+pad1) + ... + (compressed_tokens+padN)。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
      - 只支持B、S为0。
      - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：

      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 100px">
      </colgroup>
      <thead>
        <tr>
          <th>B</th>
          <th>S</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>100</td><td>65525</td><td>4096</td></tr>
        <tr><td>25</td><td>261120</td><td>4096</td></tr>
        <tr><td>100</td><td>131072</td><td>4096</td></tr>
        <tr><td>100</td><td>261120</td><td>4096</td></tr>
      </tbody>
      </table>
  - 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下state_cache不做更新，输出cmp_kv为空tensor。
- 输入属性限制：
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持block_size为1~1024。
  - 支持如下三种典型组合场景：
      - C4A：D=512，coff=2，cmp_ratio=4。
      - C4Li：D=128，coff=2，cmp_ratio=4。
      - C128A：D=512，coff=1，cmp_ratio=128。
- 确定性计算与batch一致性：
  - 默认确定性实现，相同输入多次调用结果一致。
  - <term>Ascend 950PR/Ascend 950DT</term>：batch一致性：通过aclrtSetSysParamOpt()配置ACL_OPT_DETERMINISTIC为3来开启batch一致性，开启后可以满足计算结果和所在批次大小、位置无关。

## 调用说明

- <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 、 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  | 调用方式   | 样例代码 | 说明                                          |
  | ---------- | -------- | --------------------------------------------- |
  | aclnn API  | -        | 通过[aclnnCompressor](./docs/aclnnCompressor.md)接口调用Compressor算子。 |
  | PyTorch API | -        | 通过[compressor](../../torch_extension/cann_ops_transformer/docs/zh/compressor.md)接口调用Compressor算子。 |
