# KvCompressEpilog

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

- 算子功能：在KV Cache的Epilog阶段对缓存进行原地压缩更新。算子将`bfloat16`激活值`x`量化压缩后，按`slot_mapping`将结果散写（scatter）到`cache`，`slot_mapping`中值为 -1的token跳过不处理。`x`尾轴`d`的后64列为rope段、前`d-64`列为nope段。支持三种量化模式：
  - `quant_mode=0/1`：对`x`逐组动态量化（Group-wise Dynamic Quantization）压缩为FP8（以`uint8`打包），rope段保留为`bfloat16`；
  - `quant_mode=2`：rope段做hifloat8**静态**量化（乘`x_scale`后取hifloat8），nope段做per-group **FLOAT4_E2M1 动态**量化（`scale=组内amax/6`以`bfloat16`写出）。

- 计算公式：

  对`x`的最后一维（d轴）进行量化，记第g组为$x_g$：

  - `quant_mode=0/1`（group量化，FP8）：

    $$
    scale_g = \frac{\max(|x_g|)}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

    `scale`分别以`bfloat16` / `float8_e8m0`存储（`round_scale=true`时`scale`向上取到2的幂）。

  - `quant_mode=2`（rope hifloat8 静态 + nope FLOAT4_E2M1 动态，`FP4\_MAX=6.0`）：

    $$
    rope_i = \mathrm{hifloat8}(x_i \cdot x\_scale), \qquad
    scale_g = \frac{\max(|x_g|)}{FP4\_MAX}, \quad nope_i = \mathrm{FLOAT4\_E2M1}\left(\frac{x_i}{scale_g}\right)
    $$

    `scale`以`bfloat16`写出；该模式下`round_scale`不生效。

## 参数说明

| 参数名           | 输入/输出/属性 | 描述                                                             | 数据类型      | 数据格式 |
| ---------------- | -------------- |----------------------------------------------------------------| ------------- | -------- |
| cache            | 输入/输出      | 当前层的KV Cache，原地更新，即公式中的量化目标。     | UINT8         | ND       |
| x                | 输入           | 待量化的激活输入，即公式中`x`。shape为[bs, d]。                                | BFLOAT16      | ND       |
| slot_mapping     | 输入           | token到cache slot的索引映射，值为 -1表示跳过该token。shape为[bs]。              | INT32、INT64  | ND       |
| cache            | 输出           | 原地更新后的KV Cache，与输入`cache`为同一tensor。                            | UINT8         | ND       |
| quant_group_size | 可选属性       | 量化分组大小。建议值为64。`quant_mode=2`时仅支持16/32/64，且要求`(d-64) % quant_group_size == 0`。 | INT           | -        |
| quant_mode       | 可选属性       | 量化模式：0=group(bf16 scale)，1=group(e8m0 scale)，2=rope hifloat8静态 + nope FLOAT4_E2M1动态。建议值为1。 | INT           | -        |
| round_scale      | 可选属性       | group模式下是否对每组scale向上取到2的幂。建议值为true。`quant_mode=2`下不生效。                  | BOOL          | -        |
| x_scale          | 可选属性       | `quant_mode=2`时为rope段hifloat8静态量化的缩放系数；`quant_mode=0/1`下预留未使用。建议值为1.0。     | FLOAT         | -        |

## 约束说明

- `cache`仅支持四维`[blockNum, blockSize, 1, headDim]`（num_slots = blockNum × blockSize），倒数第二维固定为1；仅在blockNum维支持非连续。headDim须 ≥ 每行写出字节数kvCacheCol。
- `slot_mapping`的维度应等于`x`的维度减1，即`slot_mapping`为`x`除最后一维外的所有维度展平。
- `x`的最后一维（d轴）需满足`d % 64 == 0`且`64 < d ≤ 8192`，按每64个元素一组进行逐组量化。
- `quant_mode=2`时，`quant_group_size`仅支持16/32/64，且nope段长度`(d-64)`需能被`quant_group_size`整除；`x`需为`bfloat16`。
- `slot_mapping`中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用说明

| 调用方式  | 样例代码                                                                       | 说明                                                                        |
| --------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_kv_compress_epilog](./examples/test_aclnn_kv_compress_epilog.cpp)   | 通过[aclnnKvCompressEpilog](./docs/aclnnKvCompressEpilog.md)调用KvCompressEpilog算子。 |
| 图模式    | -                                                                              | 通过[算子IR](./op_graph/kv_compress_epilog_proto.h)接入GE图模式调用KvCompressEpilog算子。 |
