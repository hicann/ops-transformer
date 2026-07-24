# aclnnScatterPaCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：更新KCache中指定位置的key。

- 计算公式：
  - 场景一：

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    ```  

    $$
    keyCache = slotMapping(key)
    $$

  - 场景二：

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    compressSeqOffsetOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    \begin{aligned}
    keyCache =\ & slotMapping(key[: compressSeqOffset], \\
    & ReduceMean(key[compressSeqOffset : compressSeqOffset + compressLens]), \\
    & key[compressSeqOffset + compressLens : seqLens])
    \end{aligned}
    $$

  - 场景三：

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key[seqLens - compressLens : seqLens])
    $$
  
  上述场景根据构造的参数来区别，符合第一种入参构造走场景一，符合第二种构造走场景二，符合第三种构造走场景三。场景一没有compressLensOptional、seqLensOptional、compressSeqOffsetOptional这三个可选参数，场景三没有compressSeqOffsetOptional可选参数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
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
      <td>待更新的key值，公式中的key。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyCacheRef</td>
      <td>输入/输出</td>
      <td>需要更新的keyCache，公式中的keyCache。</td>
      <td>与key一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>输入</td>
      <td>key的每个token在cache中的存储偏移，公式中的slotMapping。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressLensOptional</td>
      <td>可选输入</td>
      <td>压缩量，公式中的compressLens。</td>
      <td>与slotMapping一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>compressSeqOffsetOptional</td>
      <td>可选输入</td>
      <td>每个batch中每个head的压缩起点，公式中的compressSeqOffset。</td>
      <td>与slotMapping一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqLensOptional</td>
      <td>可选输入</td>
      <td>每个batch的实际seqLens，公式中的seqLens。</td>
      <td>与slotMapping一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheMode</td>
      <td>输入</td>
      <td>keyCacheRef的内存排布格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 参数说明中shape使用的变量说明如下：
  - batch：当前输入的序列数量（一次处理的样本数），取值为正整数。
  - seq_len：序列的长度，取值为正整数。
  - num_head：多头注意力中“头”的数量，取值为正整数。
  - k_head_size：每个注意力头中key的特征维度（单头key的长度），取值为正整数。
  - num_blocks：keyCache中预分配的块总数，用于存储所有序列的key数据，取值为正整数。
  - block_size：每个缓存块包含的token数量，取值为正整数。
- 输入值域限制：seqLensOptional和compressLensOptional里面的每个元素值必须满足公式：reduceSum(seqLensOptional[i] - compressLensOptional[i] + 1) <= num_blocks * block_size（对应场景二、三）。
- 输入数据类型限制：HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2仅支持key是3维的场景（对应场景一）。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_ScatterPaCache](./examples/test_aclnn_scatter_pa_cache.cpp) | 通过[aclnnScatterPaCache](./docs/aclnnScatterPaCache.md)调用ScatterPaCache算子 |
