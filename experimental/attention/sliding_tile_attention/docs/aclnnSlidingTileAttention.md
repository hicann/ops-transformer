# aclnnSlidingTileAttention

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | 未验证 |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |

## 功能说明

该接口实现 FastVideo Sliding Tile Attention。图像 query 访问本 head 三维滑动窗口
内的图像 KV 和全部有效文本 KV；文本区 query 访问全部图像 KV 和有效文本 KV。
输入输出布局为 BNSD，内部使用 FLOAT32 完成 score、online softmax 和输出累加。

$$
O_i=\operatorname{Softmax}\left(\frac{Q_iK_{\mathcal{V}(i)}^T}{\sqrt D}\right)
V_{\mathcal{V}(i)}
$$

## 函数原型

接口采用两段式调用。先调用 `aclnnSlidingTileAttentionGetWorkspaceSize` 获取
workspace 大小和执行器，再调用 `aclnnSlidingTileAttention` 执行。

```cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor          *query,
    const aclTensor          *key,
    const aclTensor          *value,
    aclTensor                *output,
    const aclIntArray *const *windowSize,
    uint64_t                  windowSizeLen,
    int64_t                   textLength,
    bool                      hasText,
    const char               *seqShape,
    uint64_t                 *workspaceSize,
    aclOpExecutor           **executor);

aclnnStatus aclnnSlidingTileAttention(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnSlidingTileAttentionGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor |
| -- | -- | -- | -- | -- | -- | -- | -- |
| query（const aclTensor*） | 输入 | Query，对应公式中的 Q。 | 与 key/value/output 的 shape、dtype 一致；`1 <= D <= 128`。 | FLOAT16、BFLOAT16 | ND（BNSD） | 4D `[B,N,S,D]` | √ |
| key（const aclTensor*） | 输入 | Key，对应公式中的 K。 | 约束同 query。 | 与 query 一致 | ND（BNSD） | 与 query 一致 | √ |
| value（const aclTensor*） | 输入 | Value，对应公式中的 V。 | 约束同 query。 | 与 query 一致 | ND（BNSD） | 与 query 一致 | √ |
| output（aclTensor*） | 输出 | Attention 输出。 | shape、dtype 与 query 一致。 | 与 query 一致 | ND（BNSD） | 与 query 一致 | - |
| windowSize（const aclIntArray* const*） | 输入 | 三维窗口数组。 | 每项恰含 3 个正整数（tile 数），顺序为 `(Kt,Kh,Kw)`；与 FastVideo 一致，偶数值实际覆盖 `k+1` 个 tile。 | INT64 | - | `windowSizeLen × 3` | - |
| windowSizeLen（uint64_t） | 输入 | `windowSize` 的项数。 | 取 1 时广播到所有 head；否则必须等于 N。 | UINT64 | - | - | - |
| textLength（int64_t） | 输入 | 有效文本 KV 数。 | `hasText=true` 时范围 `[0,384]`；false 时按 0 处理。 | INT64 | - | - | - |
| hasText（bool） | 输入 | 是否存在文本区。 | true 时 S 可包含最多 384 个文本预留槽。 | BOOL | - | - | - |
| seqShape（const char*） | 输入 | 图像三维画布。 | `"TxHxW"`，`T/H/W` 在 `[1,200]` 内且分别能被 6、8、8 整除；ATK 用例的 `"1"/"2"/"3"` 表示 `"30x48x80"/"36x48x48"/"18x48x80"`。 | STRING | - | - | - |
| workspaceSize（uint64_t*） | 输出 | Device workspace 字节数。 | 由接口写入。 | UINT64 | - | - | - |
| executor（aclOpExecutor**） | 输出 | 算子执行器。 | 由接口创建，传给第二段接口。 | - | - | - | - |

### 返回值

返回 `aclnnStatus`。

| 返回值 | 错误码 | 描述 |
| -- | -- | -- |
| ACLNN_SUCCESS | 0 | 成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | query/key/value/output/windowSize/seqShape/workspaceSize/executor 中存在空指针，或窗口项为空。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | dtype、format、shape、窗口、文本长度或 seqShape 不满足约束（包括 seqShape 格式错误、T/H/W 超出 `[1,200]` 或不能被 6、8、8 整除，以及 `D > 128`）。 |

## aclnnSlidingTileAttention

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor |
| -- | -- | -- | -- | -- | -- | -- | -- |
| workspace（void*） | 输入 | Device workspace 地址。 | 当 workspaceSize 大于 0 时不能为空。 | - | - | - | - |
| workspaceSize（uint64_t） | 输入 | workspace 字节数。 | 使用第一段接口返回值。 | UINT64 | - | - | - |
| executor（aclOpExecutor*） | 输入 | 算子执行器。 | 使用第一段接口返回值。 | - | - | - | - |
| stream（aclrtStream） | 输入 | 执行流。 | 算子在该流上异步执行。 | - | - | - | - |

### 返回值

返回 `aclnnStatus`；`ACLNN_SUCCESS` 表示成功，其余返回值表示执行失败。

## 约束说明

- query/key/value/output 的 `B/N/S/D` 均大于 0，shape 和 dtype 完全一致；`D <= 128`，`B`、`N` 仅受
  device 内存限制。
- 图像 tile 固定为 `(6,8,8)`，每个 tile 含 384 个 token。`seqShape` 为 `"TxHxW"`（分隔符 `x`、`X` 或 `×`），
  `T/H/W` 为 `[1,200]` 内的十进制整数，`T` 能被 6 整除、`H/W` 能被 8 整除；图像 token 数为 `T*H*W`。
- `hasText=false` 时，S 等于 `seqShape` 对应图像 token 数。
- `hasText=true` 时：
  `imageSeqLen + textLength <= S <= imageSeqLen + 384`。图像区后的前
  `textLength` 个位置参与 KV attention，剩余位置仅作为 query 输出。
- window 的每个维度必须为正整数（tile 数）；窗口以 query tile 为中心、每侧 `k/2` 个 tile（偶数值覆盖 `k+1` 个 tile），大于 tile grid 的部分按 grid 截断。
- 接口默认使用确定性实现，不提供关闭或切换确定性计算的选项。
- 当前真机验收环境为 Atlas A2、CANN 9.0.0。

## 调用示例

完整可运行示例见
[`examples/test_aclnn_sliding_tile_attention.cpp`](../examples/test_aclnn_sliding_tile_attention.cpp)。
核心调用顺序如下：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnSlidingTileAttentionGetWorkspaceSize(
    query, key, value, output, windows, windowSizeLen,
    textLength, hasText, seqShape, &workspaceSize, &executor);
if (ret != ACLNN_SUCCESS) {
    return ret;
}

void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
ret = aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
aclrtFree(workspace);
```
