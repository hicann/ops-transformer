# ScatterPaKvCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：更新KvCache中指定位置的key和value。

- 输入输出场景根据构造的参数来区别，输入输出支持以下场景，其中场景一、场景二、场景六没有compressLensOptional、seqLensOptional、compressSeqOffsetOptional这三个可选参数，场景四没有compressSeqOffsetOptional可选参数：

  - 场景一：

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]/[num_blocks, num_head, k_head_size // last_dim_k, block_size, last_dim_k]
    valueCache:[num_blocks, num_head * v_head_size // last_dim_v, block_size, last_dim_v]/[num_blocks, num_head, v_head_size // last_dim_v, block_size, last_dim_v]
    slotMapping:[batch * seq_len]
    cacheMode:"PA_NZ"

    last_dim_k = 32 / sizeof(dtypeKey)
    last_dim_v = 32 / sizeof(dtypeValue)
    (k_head_size * sizeof(dtypeKey)) % 32 == 0
    (v_head_size * sizeof(dtypeValue)) % 32 == 0
    ```

  - 场景二：

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[batch * seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[num_blocks, block_size, num_head, v_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

    其中k_head_size与v_head_size可以不同，也可以相同。

  - 场景三：

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    ```

  - 场景四：

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    scatter_mode:"Alibi"
    ```

  - 场景五：

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    valueCache:[num_blocks, block_size, 1, v_head_size]
    slotMapping:[batch * num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    compressSeqOffsetOptional:[batch * num_head]
    cacheMode:"Norm"
    scatter_mode:"Rope"/"Omni"
    ```

  - 场景六：

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    value:[]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    scatter_mode:"None"/"Nct"
    ```

  - 场景七：

    ```python
    key:[num_tokens, num_head, k_head_size]
    value:[num_tokens, num_head, v_head_size]
    keyCache:[num_blocks, num_head, block_size, k_head_size]
    valueCache:[num_blocks, num_head, block_size, v_head_size]
    slotMapping:[num_tokens]
    cacheMode:"Norm"
    scatter_mode:"NHSD"
    ```

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持场景一、二、四、五、六、七。
- <term>Ascend 950PR/Ascend 950DT</term>：仅支持场景一、二、三、四、五。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>key</td>
      <td>输入</td>
      <td>待更新的key值，当前step多个token的key。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyCacheRef</td>
      <td>输入/输出</td>
      <td>需要更新的key cache，当前layer的key cache。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>输入</td>
      <td>每个token key或value在cache中的存储偏移。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>待更新的value值，当前step多个token的value。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valueCacheRef</td>
      <td>输入/输出</td>
      <td>需要更新的value cache，当前layer的value cache。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressLensOptional</td>
      <td>可选输入</td>
      <td>压缩量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressSeqOffsetOptional</td>
      <td>可选输入</td>
      <td>每个batch每个head的压缩起点。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqLensOptional</td>
      <td>可选输入</td>
      <td>每个batch的实际seqLens。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheMode</td>
      <td>输入</td>
      <td>表示keyCacheRef和valueCacheRef的内存排布格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scatterMode</td>
      <td>输入</td>
      <td>表示更新的key和value的状态。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入</td>
      <td>key和value在非连续状态下的步长，数组长度为2。其值应该大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>key和value在非连续状态下的偏移，数组长度为2。其值应该大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

- 各产品对key/value/keyCacheRef/valueCacheRef数据类型的支持范围：
  - <term>Ascend 950PR/Ascend 950DT</term>：支持参数表中列出的全部数据类型；其中FLOAT4_E1M2、FLOAT4_E2M1仅场景一、场景二且scatterMode为"None"时支持。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持FLOAT16、BFLOAT16、INT8。

## 约束说明

  * 输入shape限制：
      * 除了key和value，输入参数不支持非连续。
      * 当key和value都是3维，则key和value的前两维shape必须相同。
      * 当key和value都是4维，则key和value的前三维shape必须相同，且keyCacheRef和valueCacheRef的第三维必须是1。
      * 当key和value是4维时，compressLensOptional、seqLensOptional为必选参数；当key和value是3维时，compressLensOptional、compressSeqOffsetOptional、seqLensOptional为可选参数。
      * 当cacheMode为“PA_NZ”时，keyCacheRef和valueCacheRef的倒数第二维必须小于UINT16_MAX（对应场景一）。
      * k_head_size和v_head_size必须32字节对齐（对应场景七）。
      * num_head必须小于4095（对应场景七）。
  * 输入值域限制：
      * slotMapping的取值范围[0,num_blocks*block_size-1]，且slotMapping内的元素值保证不重复，重复时不保证正确性。
      * 当key和value都是4维时，slotMapping是二维，且slotMapping的第一维值等于key的第一维为batch，slotMapping的第二维值等于key的第三维为num_head（对应场景三）。
      * 当key和value都是4维时，seqLensOptional是一维，且seqLensOptional的值等于key的第一维为batch（对应场景三）。
      * 当key和value是3维且存在seqLensOptional时，seqLensOptional中所有值的和等于key的第一维为num_blocks（对应场景四、五）。
      * seqLensOptional和compressLensOptional里面的每个元素值必须满足公式：reduceSum(seqLensOptional[i] - compressLensOptional[i]) <= num_blocks * block_size（对应场景三、四、五）。
      * block_size *k_head_size和block_size* v_head_size必须小于UINT32_MAX（对应场景七）。
  * 输入属性限制：
      * key、value、keyCacheRef、valueCacheRef的数据类型必须一致。
      * 当cacheModeOptional为“PA_NZ”时key、keyCacheRef和value、valueCacheRef的数据类可以不一致。
      * slotMapping、compressLensOptional、compressSeqOffsetOptional、seqLensOptional的数据类型必须一致。
      * cacheMode当传空指针或"Norm"时，仅支持ND内存排布格式，当传"PA_NZ"时，仅支持FRACTAL_NZ内存排布格式。
      * scatterMode当传空指针或"None"时，表示更新的key和value是非压缩状态且连续，当传"Alibi"时，表示更新key和value是基于Alibi结构的压缩状态，当传"Rope"时，表示更新key和value是基于Rope结构的压缩状态，当传"Omni"时，表示更新key和value是基于Omni结构的压缩状态，当传"Nct"时，表示更新的key和value是非压缩状态但非连续。
      * strides和offsets仅当scatterMode为"Nct"时生效，分别表示strideK和strideV、offsetK和offsetV。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_ScatterPaKvCache](./examples/test_aclnn_scatter_pa_kv_cache.cpp) | 通过[aclnnScatterPaKvCache](./docs/aclnnScatterPaKvCache.md)调用ScatterPaKvCache算子 |
| 图模式 | [test_geir_ScatterPaKvCache](./examples/test_geir_scatter_pa_kv_cache.cpp) | 通过[算子IR](./op_graph/scatter_pa_kv_cache_proto.h)调用ScatterPaKvCache算子 |
