# GenPositionIdsFromMask

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    x     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：根据注意力掩码 `attention_mask` 生成对应的位置索引 `position_ids`。用于变长序列打包（packing）等场景下，为每个有效 token 恢复其在序列中从 0 开始的位置编号，padding 位置填充指定值。

- 计算逻辑：对输入 `attention_mask` 的每一行（每个 Batch），沿序列维 S 从左到右扫描，维护有效 token 的累计计数。当前位置为有效 token（掩码值为 1）时，输出该 token 在本行中的位置索引，即截至当前位置的有效 token 个数减 1（从 0 开始递增编号）；当前位置为 padding（掩码值为 0）时，输出属性 `padding_fill_value` 指定的填充值。

- 等价参考实现：

```python
# mask: [B, S], 取值 0/1
out = np.empty((B, S), dtype=np.int64)
for r in range(B):
    running = 0
    for i in range(S):
        m = int(mask[r, i])
        running += m
        out[r, i] = fill if m == 0 else running - 1
```

- 示例：输入 `attention_mask = [0, 0, 1, 1, 1, 1]`，`padding_fill_value = 1`，输出 `position_ids = [1, 1, 0, 1, 2, 3]`。

## 参数说明

| 参数名             | 输入/输出/属性 | 描述                                                                                                                | 数据类型           | 数据格式 |
| ------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------ | -------- |
| attention_mask     | 输入           | 注意力掩码，二维 [B, S]。取值 1 代表该位置为有效 token，0 代表 padding 位置。BOOL 类型时 True/False 分别对应 1/0。   | INT32、INT64、BOOL | ND       |
| padding_fill_value | 可选属性       | padding 位置（掩码值为 0）在输出中的填充值。默认值为 1。                                                             | INT                | -        |
| position_ids       | 输出           | 生成的位置索引，二维 [B, S]，与输入 shape 一致。有效 token 位置为从 0 开始的递增编号，padding 位置为 `padding_fill_value`。 | INT64              | ND       |

## 约束说明

- 输入 `attention_mask` 仅支持二维 [B, S]，B 与 S 均需大于 0。
- 输出 `position_ids` 数据类型固定为 INT64，不随输入数据类型变化。
- 输出 `position_ids` 的 shape 与输入 `attention_mask` 一致。
- 输入 `attention_mask` 取值应为 0 或 1（BOOL 类型对应 False/True）；非 0/1 的其他取值行为未定义。

## 调用说明

| 调用方式  | 调用样例                                                                                     | 说明                                                                                              |
| --------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_gen_position_ids_from_mask](./examples/test_aclnn_gen_position_ids_from_mask.cpp) | 通过 [aclnnGenPositionIdsFromMask](./docs/aclnnGenPositionIdsFromMask.md) 接口方式调用该算子。 |
