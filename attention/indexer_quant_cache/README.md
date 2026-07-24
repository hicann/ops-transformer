# IndexerQuantCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在Indexer注意力机制的Epilog阶段对KV Cache进行原地压缩更新。算子按逐块动态量化（Per-Block Dynamic Quantization）将`float16`/`bfloat16`激活值`x`压缩为FP8（E4M3/E5M2）、INT8（uint8）或MX-FP4格式，并按`slot_mapping`将量化结果与对应`cache_scale`散写（scatter）到`cache`，`slot_mapping`中值为 -1的token跳过不处理。支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。

- 计算公式：

  对`x`的最后一维（d轴）进行量化，记第g组为$x_g$：

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - `quant_mode=0`（MX-FP8）：scale存储为`float8_e8m0`，`round_scale=true`时对scale进行舍入。
  - `quant_mode=1`（Normal）：动态逐块量化，scale存储为`float32`。
  - `quant_mode=2`（HiFloat8）：输出为$x \times x\_scale$后的hifloat8。
  - `quant_mode=3`（MX-FP4）：按标准MX块（每32元素）量化为FP4（uint8每字节打包2个fp4值），scale存储为`float8_e8m0`。

## 参数说明

| 参数名       | 输入/输出/属性 | 描述                                                                          | 数据类型                                                      | 数据格式 |
| ------------ | -------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ | -------- |
| cache        | 输入/输出      | 当前层的KV Cache向量缓存，原地更新，即公式中的量化目标。MX-FP4模式下为uint8，每字节打包2个fp4值。 | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输入/输出      | 每块量化的scale因子，原地更新，即公式中`scale`。                            | FLOAT、FLOAT8_E8M0                                            | ND       |
| x            | 输入           | 待量化的激活输入，即公式中`x`。shape为[bs, d]。                              | FLOAT16、BFLOAT16                                             | ND       |
| slot_mapping | 输入           | token到cache slot的索引映射，值为 -1表示跳过该token。shape为[bs]。       | INT32                                                        | ND       |
| cache        | 输出           | 原地更新后的KV Cache，与输入`cache`为同一tensor。                            | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输出           | 原地更新后的scale，与输入`cache_scale`为同一tensor。                         | FLOAT、FLOAT8_E8M0                                            | ND       |
| quant_mode   | 可选属性       | 量化模式：0=MX-FP8，1=Normal，2=HiFloat8，3=MX-FP4。默认值为1。                | INT                                                          | -        |
| round_scale  | 可选属性       | MX-FP8模式（quant_mode=0）下是否对scale进行舍入。默认值为true。              | BOOL                                                         | -        |
| x_scale      | 可选属性       | HiFloat8模式（quant_mode=2）下的全局scale乘数。默认值为1.0。                 | FLOAT                                                        | -        |

## 约束说明

- `x`的最后一维`d`须能被32整除且 `d ≤ 8192`。
- `slot_mapping`的维度应等于`x`的维度减1，即`slot_mapping`为`x`除最后一维外的所有维度展平。
- 数据类型组合需匹配量化模式：Normal/HiFloat8模式cache为FP8/UINT8且cache_scale为FLOAT；MX-FP8模式cache为FP8且cache_scale为FLOAT8_E8M0；MX-FP4模式cache为FP4且cache_scale为FLOAT8_E8M0。
- `slot_mapping`中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用说明

| 调用方式  | 样例代码                                                                          | 说明                                                                          |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_indexer_quant_cache](./examples/test_aclnn_indexer_quant_cache.cpp)    | 通过[aclnnIndexerQuantCache](./docs/aclnnIndexerQuantCache.md)调用IndexerQuantCache算子。 |
| 图模式    | -                                                                                 | 通过[算子IR](./op_graph/indexer_quant_cache_proto.h)接入GE图模式调用IndexerQuantCache算子。 |
