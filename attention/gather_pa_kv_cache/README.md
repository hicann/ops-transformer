# GatherPaKvCache

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

- 算子功能：根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。
- 计算逻辑：
  - keyRef/valueRef的第一个维度取决于seq_lens大小。
  - 如果isSeqLensCumsum为true，则seqLens中最后一个值即为keyRef/valueRef的第一个维度大小： keyRef[dim0] = seqLens[-1]
  - 如果isSeqLensCumsum为false，则seqLens中所有值的累加和即为keyRef/valueRef的第一个维度大小：keyRef[dim0] = sum(seqLens)

  关于**keyRef**、**valueRef**的一些限制条件如下：
  - 每个token大小控制在148k以内，例如，对于fp16/bf16类型，num_heads *head_size(keyRef/valueRef)取128*576。

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

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| keyCache                     | 输入 | 当前层存储的key向量缓存 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND         |
| valueCache                     | 输入 | 当前层存储的value向量缓存 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | FRACTAL_NZ |
| blockTables                     | 输入 | 每个batch中KV Cache的逻辑块到物理块的映射关系 | INT32、INT64       | ND         |
| seqLens                     | 输入 | 每个batch对应的序列长度 | INT32、INT64       | ND         |
| keyRef                     | 输入/输出 | 当前层的key向量 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| valueRef                     | 输入/输出 | 当前层的value向量 | INT8, FLOAT16, BFLOAT16, FLOAT, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN       | ND         |
| seqOffset                     | 输入 | blockTables获取blockId时存在的首偏移 | INT32、INT64       | ND         |
| cacheMode                     | 输入 | 表示输入的数据排布格式，支持Norm、PA_NZ | String      | ND         |
| isSeqLensCumsum                     | 输入 | 表示seqLens是否为累加和。false表示非累加和 | BOOL       | ND         |

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_gather_pa_kv_cache](./examples/test_aclnn_gather_pa_kv_cache.cpp) | 通过[aclnnGatherPaKvCache](./docs/aclnnGatherPaKvCache.md)调用GatherPaKvCache算子 |
