# aclnnMatmulSwiglu

## 产品支持情况

（对应 op_def 注册的 `ascend910b` / `ascend910_93`。）

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910b" id6 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id2 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id3 -->

## 功能说明

- **算子功能**：融合 `gate_proj + up_proj + SiLU + Mul`（SwiGLU 门控 MLP 前向）。
- **计算公式**：

  令 `[gate | up] = x @ weight (+ bias)`，沿最后一维对半切分，则

  ```
  y = SiLU(gate) * up,   SiLU(g) = g * sigmoid(g) = g / (1 + e^{-g})
  ```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|----------------|------|----------|----------|
| x | 输入 | 激活，形状 `[M, K]` | FLOAT16、BFLOAT16、FLOAT32 | ND |
| weight | 输入 | 打包权重 `[K, 2N]`（前 N 列 gate，后 N 列 up）。`transposeWeight=true` 时为 `[2N, K]` | 同 x | ND |
| bias | 输入（可选） | 偏置 `[2N]`。原型可选，但自动生成的 aclnn 接口会非空校验，无偏置时传全 0 张量 | FLOAT32（固定，在 cube 内以 fp32 累加，不随 x 变化） | ND |
| transposeWeight | 属性 | weight 是否转置，默认 false | BOOL | - |
| y | 输出 | 结果 `[M, N]` | 同 x | ND |

## 接口原型

两段式接口：先调用 `aclnnMatmulSwigluGetWorkspaceSize` 获取 workspace 大小与执行器，再调用 `aclnnMatmulSwiglu` 执行。

> **bias 不能传 nullptr**：本算子无手写 `op_api` L2 实现，`aclnn` 接口由 opbuild 自动生成，
> 其 `GetWorkspaceSize` 会对 `bias` 做非空校验，传 `nullptr` 会返回 `161001 (ACLNN_ERR_PARAM_NULLPTR)`。
> 不需要偏置时请传入一个形状为 `[2N]`、值全 0 的 bias 张量。参数顺序为 输入 → 属性 → 输出。

```cpp
aclnnStatus aclnnMatmulSwigluGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* bias,        // 不能为 nullptr; 无偏置时传全 0 的 [2N] 张量
    bool             transposeWeight,
    const aclTensor* out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);

aclnnStatus aclnnMatmulSwiglu(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream);
```

## 约束说明

- `weight` 的 `2N` 维必须能被 2 整除。
- `x` 的最后一维 `K` 必须与 `weight` 的 `K` 维一致。
- 中间结果以 FLOAT32 累加；FLOAT32 输入下 matmul 精度模式由 tiling 决定，必要时调整 `cubeMathType`/`SetBufferSpace`。

## 调用样例

参见 [examples/test_aclnn_matmul_swiglu.cpp](../examples/test_aclnn_matmul_swiglu.cpp)。
