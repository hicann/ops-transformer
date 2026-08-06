# aclnnCompressor

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：Compressor是推理场景下SMLA和QLI的前处理算子，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，Compressor可以有效地减少计算开销。主要计算过程为：
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

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnCompressorGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnCompressor”接口执行计算。

```cpp
aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor       *stateCacheRef,
    const aclTensor *ape,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          cmpRatio,
    int64_t          coff,
    int64_t          cacheMode,
    int64_t          stateCacheStrideDim0,
    const aclTensor *cmpKvOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

``` cpp
aclnnStatus aclnnCompressor(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCompressorGetWorkspaceSize

- **参数说明**

    | 参数名                      | 输入/输出 | 描述  |  使用说明  | 数据类型       | 数据格式   | 维度（shape） | 非连续Tensor |
    |----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|-|-|-|
    | x | 输入 | 公式中的$X$，表示原始不经压缩的数据。 |  支持B=0,S=0,T=0的空Tensor。  | FLOAT16、BFLOAT16 | ND         | BS合轴：[T,H]、BS非合轴：[B,S,H]|×|
    | wkv | 输入 | 公式中的$W^{KV}$，表示kv压缩权重。  |不支持空Tensor。| FLOAT16、BFLOAT16 | ND |[coff* D,H]|×|
    | wgate | 输入 | 公式中的$W^{Gate}$，表示gate压缩权重。 |不支持空Tensor。| FLOAT16、BFLOAT16 | ND |[coff* D,H]|×|
    | stateCacheRef | 输入 | 公式中的$\left[kv\_state, score\_state\right]$, 表示kv\_state和score\_state的历史数据。 |不支持空Tensor| FLOAT32     | ND         |[block_num,block_size,2*coff* D]|支持0轴非连续|
    | ape | 输入 | 公式中的$Ape$，表示positional biases。 | 不支持空Tensor。|FLOAT32       | ND         |[cmp_ratio,coff* D]|×|
    | stateBlockTable | 可选输入 | 表示state\_cache存储使用的block映射表。|当其中元素的值为0时，表示当前位置无需进行更新state\_cache操作；支持S=0,T=0的空Tensor。| INT32 | ND         |cache_mode=1时，shape为[B,ceil(Smax/block_size)]，Smax为每个Batch中最大的Sequence Length，当x的shape为[B,S,H]时，Smax=max(start_pos)+S。当x的shape为[T,H]时，Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n])。cache_mode=2时，shape为[B]。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作|×|
    | cuSeqlens | 可选输入 | 表示不同Batch中的有效token数。  |支持B=0,S=0,T=0的空Tensor；当x的shape为[B,S,H]时，参数必须为空。| INT32          | ND         |当x的shape为[T,H]时，输入shape为[B+1,]|×|
    | seqused | 可选输入 | 表示不同Batch中实际参与压缩的token数。 |如果指定为None时，表示和每个Batch上的Sequence Length长度相同；支持B=0的空Tensor；如果指定为None时，表示和每个Batch上的Sequence Length长度相同。该入参中每个Batch的有效token数要求小于等于对应Sequence Length长度。当x的shape为[B,S,H]时，要求seqused[n] <= S，且不小于0；当x的shape为[T,H]时，要求seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。| INT32          | ND         |[B,]|×|
    | startPos | 可选输入 | 表示计算起始位置。 |支持B=0,T=0的空Tensor；当输入为None时，表示从0开始进行计算。| INT32          | ND         |[B,]|×|
    | cmpRatio | 输入 | 用于稀疏计算，表示数据压缩率。 |取值范围为[2, 128]内的整数。| INT32          | -         |-|-|
    | coff | 可选输入 | 表示是否进行overlap数据重排。 |取值范围为[1, 2]。当coff=1时，无需进行overlap数据重排。当coff=2时，需要进行overlap数据重排。| INT32          | -         |-|-|
    | cacheMode | 可选输入 | 表示state_cache的存储模式。 |取值范围为[1, 2]；1表示连续buffer，2表示循环buffer。| INT32          | -         |-|-|
    | stateCacheStrideDim0 | 可选输入 | 表示state_cache的0轴stride。 |-| INT32     | -         |-|-|
    | cmpKv | 输出 | 表示压缩后的数据。 |支持B=0,S=0,T=0的空Tensor。| FLOAT16、BFLOAT16         | ND          |BS合轴：[min(T,T//cmp_ratio+B),D]、BS非合轴：[B,ceil(S/cmp_ratio),D]|×|

<!-- npu="A3" id6 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
 cacheMode不支持输入2，且不支持0轴非连续。
 cmp_ratio仅支持2/4/8/16/32/64/128

<!-- end id6 -->
- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>返回值</th>
          <th>错误码</th>
          <th>描述</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>必须传入的参数（如接口核心依赖的输入/输出参数）中存在空指针。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>输入参数的shape（维度/尺寸）、dtype（数据类型）不在接口支持的范围内。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>API内存调用NPU Runtime接口时发生异常（如Runtime服务未启动、内存申请失败等）。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>tiling发生异常，入参的dtype类型或者shape错误。</td>
        </tr>
      </tbody>
    </table>

## aclnnCompressor

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCompressorGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnCompressor默认确定性实现。
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - stateCache支持输入shape[block_num,block_size,2*coff* D]，要求blockNum>0，cacheMode=2时，需要满足blockSize >= coff * cmp_ratio + S - 1。
    - ape支持输入shape[cmp_ratio,coff* D]
    - startPos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cuSeqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须位0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
        - cacheMode=1时，state\_block\_table支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n])。cacheMode=2时，state\_block\_table支持输入shape[B]。
        - cmpKv，输出shape为[min(T,T//cmp_ratio+B),D]：compressed_tokens + compressed_tokens + ... + compressed_tokens + pad。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cuSeqlens，参数必须为空。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
        - cacheMode=1时，stateBlockTable支持输入shape[B,ceil(Smax/block_size)]。Smax为每个Batch中最大的Sequence Length，即Smax=max(start\_pos)+S。cacheMode=2时，stateBlockTable支持输入shape[B]。
        - cmpKv，输出shape为[B,ceil(S/cmp_ratio),D]：(compressed_tokens+pad0) + (compressed_tokens+pad1) + ...  + (compressed_tokens+padN)。
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
- 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下stateCache不做更新，输出cmpKv为空tensor。
- 输入属性限制：
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持blockSize为1~1024。
  - 支持如下三种典型组合场景：
      - C4A: D=512, coff=2, cmp_ratio=4；
      - C4Li: D=128, coff=2, cmp_ratio=4；
      - C128A: D=512, coff=1, cmp_ratio=128。

## 调用示例

  无
