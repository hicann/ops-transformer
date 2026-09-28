# GatherPaKvCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                 |    √     |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term> |      ×     |
| <term>Atlas推理系列产品</term> |      ×     |
| <term>Atlas训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。

- 示例：

  ```python
    keyCache_shape: [128, 128, 16, 144]
    valueCache_shape: [128, 128, 16, 128]
    blockTables_shape: [16, 12]
    seqLens_shape: [16]
    keyRef_shape: [8931, 16, 144]
    valueRef_shape: [8931, 16, 128]
    seqOffset_shape: [16]
    out1_shape: [8931, 16, 144]
    out2_shape: [8931, 16, 128]
  ```

## 公式/算法说明

- 算子根据`blockTables`中的物理块索引定位`keyCache`和`valueCache`中的token，并按batch顺序将token搬运到`keyRef`和`valueRef`。算子不执行数值计算或数据类型转换。
- `isSeqLensCumsum`为`true`时，`seqLens`为累加和，`seqLens[0]`为0，第`i`个batch的序列长度为`seqLens[i + 1] - seqLens[i]`，输出token总数为`seqLens[-1]`。
- `isSeqLensCumsum`为`false`时，`seqLens[i]`为第`i`个batch的序列长度，输出token总数为`sum(seqLens)`。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| keyCache                     | 输入 | 当前层存储的key向量缓存 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND、FRACTAL_NZ |
| valueCache                     | 输入 | 当前层存储的value向量缓存 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND、FRACTAL_NZ |
| blockTables                     | 输入 | 每个batch中KV Cache的逻辑块到物理块的映射关系 | INT32、INT64       | ND         |
| seqLens                     | 输入 | 每个batch对应的序列长度 | INT32、INT64       | ND         |
| keyRef                     | 输入/输出 | 当前层的key向量 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| valueRef                     | 输入/输出 | 当前层的value向量 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| seqOffset                     | 可选输入 | blockTables获取blockId时存在的首偏移；不传表示无首偏移 | INT32、INT64       | ND         |
| cacheMode                     | 输入（IR属性） | 表示输入的数据排布格式，支持`Norm`、`PA_NZ`；对应IR属性`cache_mode`，缺省值为`Norm` | String      | -         |
| isSeqLensCumsum                     | 输入（IR属性） | 表示seqLens是否为累加和；对应IR属性`is_seq_lens_cumsum`，缺省值为`true` | BOOL       | -         |

说明：`keyRef`、`valueRef`是aclnn接口参数名，分别对应算子IR中的`key`、`value`。上表列出所有支持产品的数据类型和数据格式并集，具体产品约束见下文。

## 约束说明

- `cacheMode`为`Norm`时，`keyCache`和`valueCache`的数据格式均为ND，shape分别为`[num_blocks, block_size, num_heads, head_size_k]`和`[num_blocks, block_size, num_heads, head_size_v]`；`keyRef`和`valueRef`的shape分别为`[num_tokens, num_heads, head_size_k]`和`[num_tokens, num_heads, head_size_v]`。
- `cacheMode`为`PA_NZ`时，`keyCache`和`valueCache`的数据格式均为FRACTAL_NZ，shape分别为`[num_blocks, num_heads * head_size_k / elenum_aligned, block_size, elenum_aligned]`和`[num_blocks, num_heads * head_size_v / elenum_aligned, block_size, elenum_aligned]`；`keyRef`和`valueRef`的shape分别为`[num_tokens, num_heads * head_size_k]`和`[num_tokens, num_heads * head_size_v]`。8bit、16bit、32bit数据类型对应的`elenum_aligned`分别为32、16、8。
- `blockTables`的shape为`[batch, block_indices]`，元素取值范围为`[0, num_blocks)`；`seqLens`的shape为一维。`isSeqLensCumsum`为`false`时，`seqLens`长度为`batch`；为`true`时，`seqLens`长度为`batch + 1`且首元素为0。
- `seqOffset`可不传；传入时shape为`[batch]`，数据类型与`blockTables`保持一致。
- `keyRef`与`keyCache`的数据类型保持一致，`valueRef`与`valueCache`的数据类型保持一致；`blockTables`、`seqLens`和`seqOffset`的数据类型保持一致。
- 单个key或value token的数据量不超过148 KiB。例如FLOAT16或BFLOAT16场景下，`num_heads * head_size`可取`128 * 576`。
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - `cacheMode`仅支持`Norm`，`keyCache`和`valueCache`仅支持ND格式。
  - `keyCache`、`valueCache`、`keyRef`和`valueRef`仅支持INT8、FLOAT16、BFLOAT16；`blockTables`、`seqLens`和`seqOffset`仅支持INT32。
- <term>Ascend 950PR&950DT系列产品</term>：
  - `cacheMode`支持`Norm`和`PA_NZ`；`Norm`模式下`keyCache`和`valueCache`均为ND格式，`PA_NZ`模式下均为FRACTAL_NZ格式。
  - `keyCache`为FLOAT8_E4M3FN时，`valueCache`还允许使用FLOAT16或BFLOAT16；其他场景下key和value的数据类型保持一致。

## 精度说明

GatherPaKvCache仅搬运缓存中的数据，不执行数值运算或数据类型转换。输入满足约束时，`keyRef`和`valueRef`中的每个元素与其在`keyCache`和`valueCache`中对应的元素按位一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_gather_pa_kv_cache](./examples/test_aclnn_gather_pa_kv_cache.cpp) | 通过[aclnnGatherPaKvCache](./docs/aclnnGatherPaKvCache.md)调用GatherPaKvCache算子 |
