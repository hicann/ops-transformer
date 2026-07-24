# ScatterPaKvCacheWithKScale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：更新KvCache中指定位置的key和value，同时更新key的scale值。

- 输入输出支持以下场景：
  - 场景一：

    ```python
    key:[batch * seq_len, num_head, head_size]
    value:[batch * seq_len, num_head, head_size]
    key_cache:[num_blocks, num_head, block_size, head_size]
    value_cache:[num_blocks, num_head, block_size, head_size]
    slot_mapping:[batch * seq_len]
    key_scale:[batch * seq_len, num_head]
    key_scale_cache:[num_blocks, num_head, block_size, 1]
    cache_layout:"BNBD"
    ```

    其中key和value的dtype为FLOAT8_E5M2或FLOAT8_E4M3FN，key_scale和key_scale_cache的dtype为FLOAT。

    计算公式：

    对于每个token（i ∈ [0, num_tokens)）和每个头（j ∈ [0, num_head)）：

    ```python
    block_idx = slot_mapping[i] // block_size
    block_offset = slot_mapping[i] % block_size

    key_cache[block_idx][j][block_offset][:] = key[i][j][:]
    value_cache[block_idx][j][block_offset][:] = value[i][j][:]
    key_scale_cache[block_idx][j][block_offset][0] = key_scale[i][j]
    ```

    其中：
    - num_tokens = batch * seq_len
    - block_idx：slot_mapping映射到的block索引
    - block_offset：block内的偏移量

- <term>Ascend 950PR/Ascend 950DT</term>：仅支持场景一。

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
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>待更新的value值，当前step多个token的value。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_cache</td>
      <td>输入/输出</td>
      <td>需要更新的key cache，当前layer的key cache。</td>
      <td>与key保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_cache</td>
      <td>输入/输出</td>
      <td>需要更新的value cache，当前layer的value cache。</td>
      <td>与value保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slot_mapping</td>
      <td>输入</td>
      <td>每个token key或value在cache中的存储偏移。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_scale</td>
      <td>输入</td>
      <td>待更新的key scale值，当前step多个token的key scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_scale_cache</td>
      <td>输入/输出</td>
      <td>需要更新的key scale cache，当前layer的key scale cache。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cache_layout</td>
      <td>属性</td>
      <td>表示key_cache和value_cache的内存排布格式。当传空指针或"BNBD"时，表示格式为[num_blocks, num_head, block_size, head_size]。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - 默认确定性实现。
- key、value、key_cache、value_cache的数据类型必须一致；
- slot_mapping的取值范围[0, num_blocks*block_size-1]，且slot_mapping内的元素值保证不重复，重复时不保证正确性；
- key和value的前两维shape必须相同；
- key_scale是两维tensor，shape为[batch * seq_len, num_head]，尾轴可以不连续；
- key_scale_cache是四维tensor，shape为[num_blocks, num_head, block_size, 1]，最后一维必须为1，尾轴必须连续。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_ScatterPaKvCacheWithKScale](./examples/test_aclnn_scatter_pa_kv_cache_with_k_scale.cpp) | 通过[aclnnScatterPaKvCacheWithKScale](./docs/aclnnScatterPaKvCacheWithKScale.md)调用ScatterPaKvCacheWithKScale算子 |
| 图模式 | [test_geir_ScatterPaKvCacheWithKScale](./examples/test_geir_scatter_pa_kv_cache_with_k_scale.cpp) | 通过[算子IR](./op_graph/scatter_pa_kv_cache_with_k_scale_proto.h)调用ScatterPaKvCacheWithKScale算子 |
