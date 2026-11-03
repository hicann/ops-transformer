# aclnnKvCacheTurboQuant

`aclnnKvCacheTurboQuant` 对标准 MHA/GQA KV 向量执行低比特主量化和 QJL 残差符号编码。

## 函数原型

```cpp
aclnnStatus aclnnKvCacheTurboQuantGetWorkspaceSize(
    const aclTensor* kvVectors,
    const aclTensor* rotationMatrix,
    const aclTensor* qjlMatrix,
    int64_t mseBits,
    int64_t outputDtype,
    const aclTensor* quantIdxOut,
    const aclTensor* quantQjlOut,
    const aclTensor* quantNormOut,
    const aclTensor* quantGammaOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnKvCacheTurboQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 参数说明

| 参数 | 说明 |
|---|---|
| kvVectors | BF16、ND、连续 `[T,H,D]` |
| rotationMatrix | FP32 或 BF16、ND、连续 `[D,D]` |
| qjlMatrix | FP32 或 BF16、ND、连续 `[J,D]` |
| mseBits | 主量化位宽，取 2、3 或 4 |
| outputDtype | metadata 类型，0 表示 BF16，1 表示 FP16 |
| quantIdxOut | UINT8 `[T,H,ceil(D*mseBits/8)]` |
| quantQjlOut | UINT8 `[T,H,ceil(J/8)]` |
| quantNormOut | BF16 或 FP16 `[T,H]` |
| quantGammaOut | BF16 或 FP16 `[T,H]` |
| workspaceSize | 第一阶段返回的工作区字节数 |
| executor | 第一阶段创建、第二阶段使用的执行器 |
| workspace | 按 `workspaceSize` 分配的设备内存 |
| stream | ACL Runtime stream |

## 调用流程

1. 创建输入、输出 `aclTensor`，输出 Shape 和 dtype 必须与属性一致。
2. 调用 `aclnnKvCacheTurboQuantGetWorkspaceSize` 获取工作区大小和执行器。
3. 根据返回值分配 workspace。
4. 调用 `aclnnKvCacheTurboQuant`，随后同步 stream。
5. 释放 workspace、Tensor、输入输出内存和 stream。

主索引按低位优先打包；QJL 投影值大于等于零时对应符号位为 1。`quant_norm` 保存输入向量
L2 范数，`quant_gamma` 保存原尺度残差范数。
