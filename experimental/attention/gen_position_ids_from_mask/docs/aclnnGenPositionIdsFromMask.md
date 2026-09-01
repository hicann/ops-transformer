# aclnnGenPositionIdsFromMask

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：x
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：√
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：√
- <term>Atlas 200I/500 A2 推理产品</term>：×
- <term>Atlas 推理系列产品</term>：×
- <term>Atlas 训练系列产品</term>：×

## 功能说明

- 接口功能：根据注意力掩码 `attentionMask` 生成位置索引 `positionIds`。用于变长序列打包（packing）等场景下，为每个有效 token 恢复其在序列中从 0 开始的位置编号，padding 位置填充 `paddingFillValue` 指定的值。语义对齐 HuggingFace 的 `prepare_inputs_for_generation`。

- 计算逻辑：对 `attentionMask` 的每一行（每个 Batch），沿序列维计算前缀和后减 1 得到有效 token 的递增位置编号，padding 位置替换为填充值。等价表达：

  ```python
  c = cumsum(mask, axis=-1)
  p = c - 1
  p[mask == 0] = paddingFillValue
  ```

  即掩码值为 1 的位置输出该 token 的位置索引（从 0 开始递增）；掩码值为 0 的位置输出 `paddingFillValue`。

- 示例：输入 `attentionMask = [0, 0, 1, 1, 1, 1]`，`paddingFillValue = 1`，输出 `positionIds = [1, 1, 0, 1, 2, 3]`。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnGenPositionIdsFromMaskGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnGenPositionIdsFromMask` 接口执行计算。

```cpp
aclnnStatus aclnnGenPositionIdsFromMaskGetWorkspaceSize(
    const aclTensor  *attentionMask,
    int64_t           paddingFillValue,
    aclTensor        *positionIds,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)

aclnnStatus aclnnGenPositionIdsFromMask(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnGenPositionIdsFromMaskGetWorkspaceSize

### 参数说明

| 参数名           | 输入/输出 | 描述                                        | 使用说明                                                     | 数据类型           | 数据格式 | 维度(shape) | 非连续Tensor |
| ---------------- | --------- | ------------------------------------------- | ------------------------------------------------------------ | ------------------ | -------- | ----------- | ------------ |
| attentionMask    | 输入      | 注意力掩码，对应计算逻辑中的输入 mask。      | 不支持空 Tensor。取值为 0 或 1（BOOL 对应 False/True）。      | INT32、INT64、BOOL | ND       | [B, S]      | √            |
| paddingFillValue | 输入      | padding 位置（掩码值为 0）在输出中的填充值。 | 默认值为 1。                                                 | INT64              | -        | -           | -            |
| positionIds      | 输出      | 生成的位置索引，对应计算逻辑中的输出。       | 不支持空 Tensor。shape 与 attentionMask 一致，数据类型固定为 INT64。 | INT64              | ND       | [B, S]      | ×            |
| workspaceSize    | 输出      | 返回需要在 Device 侧申请的 workspace 大小。  | -                                                            | -                  | -        | -           | -            |
| executor         | 输出      | 返回 op 执行器，包含了算子计算流程。         | -                                                            | -                  | -        | -           | -            |

### 返回值

aclnnStatus：返回状态码，具体参见 aclnn 返回码。

第一段接口完成入参校验，出现以下场景时报错：

| 返回值                  | 错误码 | 描述                                                                                     |
| ----------------------- | ------ | ---------------------------------------------------------------------------------------- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | attentionMask、positionIds 存在空指针。                                                  |
| ACLNN_ERR_PARAM_INVALID | 161002 | attentionMask 的数据类型不在支持范围内；或 attentionMask 的 shape 维度不为 2；或 B、S 不大于 0。 |

## aclnnGenPositionIdsFromMask

### 参数说明

| 参数名        | 输入/输出 | 描述                                                    |
| ------------- | --------- | ------------------------------------------------------- |
| workspace     | 输入      | 在 Device 侧申请的 workspace 内存地址。                 |
| workspaceSize | 输入      | 在 Device 侧申请的 workspace 大小，由第一段接口获取。   |
| executor      | 输入      | op 执行器，包含了算子计算流程。                          |
| stream        | 输入      | 指定执行任务的 Stream。                                  |

### 返回值

aclnnStatus：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- aclnnGenPositionIdsFromMask 默认确定性实现。
- 输入 `attentionMask` 仅支持二维 [B, S]，且 B、S 均需大于 0。
- 输出 `positionIds` 数据类型固定为 INT64，不随输入数据类型变化，shape 与输入一致。
- 输入 `attentionMask` 取值应为 0 或 1（BOOL 类型对应 False/True）；非 0/1 的其他取值行为未定义。

## 调用示例

编译与运行方法请参考编译与运行样例。完整示例见 [test_aclnn_gen_position_ids_from_mask.cpp](../examples/test_aclnn_gen_position_ids_from_mask.cpp)，该示例覆盖 int32/int64/bool 三种输入类型与左/右/中间 padding、全 0、全 1、单元素、多行等多种掩码模式，并内置 host golden 逐位比对。
