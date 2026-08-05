# RainFusionAttention算子

## 概述

RainFusionAttention是一个基于CATLASS模板库实现的高性能稀疏注意力算子，支持灵活的块级稀疏模式。该算子从catlass的32_sparse_attention_infer示例迁移而来,针对ops-transformer-dev框架进行了适配。

## 功能特性

- **灵活的稀疏块模式**: 支持自定义的x*y块级稀疏模式，通过blockShape参数指定
- **TND格式KV Cache**: 支持TND [T, N, D]格式的KV cache布局（注：BNSD格式暂不支持）
- **多种Q输入布局**: 支持BSH、TND两种Query输入布局（注：BNSD格式暂不支持）
- **高性能计算**: 基于CATLASS模板库，充分利用昇腾A2硬件特性
- **多精度支持**: 支持FP16和BF16两种精度
- **非对齐序列**: 支持序列长度不能被块大小整除的场景，自动处理边界情况

## 接口定义

### aclnnRainFusionAttentionGetWorkspaceSize

```cpp
aclnnStatus aclnnRainFusionAttentionGetWorkspaceSize(
    const aclTensor *query,                    // Query输入 [B, S, H] or [T, N, D] or [B, N, S, D]
    const aclTensor *key,                      // Key输入(TND or BNSD格式)
    const aclTensor *value,                    // Value输入(TND or BNSD格式)
    const aclTensor *attenMask,                // Attention mask (可选)
    const aclIntArray *actualSeqLengths,       // 实际Q序列长度(可选)
    const aclIntArray *actualSeqLengthsKv,     // 实际KV序列长度(可选)
    const aclTensor *blockTable,               // Block表(可选，用于PagedAttention)
    const aclTensor *selectIdx,                // 稀疏块索引 [T, headNum, maxKvBlockNum]
    const aclTensor *selectNumIdx,             // 每个Q块的KV块数量 [T, headNum]
    const char *qInputLayout,                  // Q输入布局: "BSH", "TND", "BNSD"
    const char *kvInputLayout,                 // KV输入布局: "TND", "BNSD"
    int64_t numKeyValueHeads,                  // KV头数
    int64_t maskType,                          // Mask类型
    double scaleValue,                         // 缩放因子
    const aclIntArray *blockShape,             // 块形状 [blockShapeX, blockShapeY]
    const aclTensor *attentionOut,             // 输出tensor
    const aclTensor *softmaxLse,               // Softmax LSE输出(可选)
    uint64_t *workspaceSize,                   // 返回workspace大小
    aclOpExecutor **executor);                 // 返回executor
```

### aclnnRainFusionAttention

```cpp
aclnnStatus aclnnRainFusionAttention(
    void *workspace,                           // workspace地址
    uint64_t workspaceSize,                    // workspace大小
    aclOpExecutor *executor,                   // executor
    const aclrtStream stream);                 // ACL stream
```

## 参数说明

### 输入参数

- **query**: Query tensor,支持以下布局
  - BSH: [batch, seqlen, num_heads * head_dim]
  - TND: [total_tokens, num_heads, head_dim]

- **key/value**: KV cache tensor,仅支持TND布局
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - 注：BNSD格式暂不支持，如需使用请转换为TND格式

- **selectIdx**: 稀疏块索引数组 [T, headNum, maxKvBlockNum]
  - T: 所有batch中Q方向切块的总数
  - headNum: 头数
  - maxKvBlockNum: 最大KV块数量
  - 存储每个Q块选择的KV块索引

- **selectNumIdx**: KV块数量数组 [T, headNum]
  - 存储每个Q块实际选择的KV块数量

- **blockShape**: 稀疏块形状 [blockShapeX, blockShapeY]
  - blockShapeX: Q方向块大小
  - blockShapeY: KV方向块大小

### 输出参数

- **attentionOut**: 注意力输出，shape与query相同
- **softmaxLse**: Softmax log-sum-exp输出(可选)

## 使用示例

```cpp
// 1. 准备输入数据
int64_t batch = 1;
int64_t qSeqlen = 256;
int64_t kvSeqlen = 512;
int64_t numHeads = 8;
int64_t numKvHeads = 8;
int64_t headDim = 128;
int64_t blockShapeX = 64;
int64_t blockShapeY = 64;

// 2. 创建tensor (略)

// 3. 设置blockShape
int64_t blockShapeData[2] = {blockShapeX, blockShapeY};
aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);

// 4. 调用算子
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnRainFusionAttentionGetWorkspaceSize(
    query, key, value, nullptr, nullptr, nullptr, nullptr,
    selectIdx, selectNumIdx, "BSH", "TND",
    numKvHeads, 0, scaleValue, blockShape,
    attentionOut, nullptr, &workspaceSize, &executor);

if (ret == ACLNN_SUCCESS) {
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ret = aclnnRainFusionAttention(workspace, workspaceSize, executor, stream);
    
    aclrtFree(workspace);
}
```

## KV Cache布局说明

### TND格式 [T, N, D] (当前支持)

- T: 总token数量
- N: KV头数
- D: 头维度
- 适用于变长序列场景
- **这是当前唯一支持的格式**

### BNSD格式 [B, N, S, D] (暂不支持)

- BNSD格式暂未实现，计划在未来版本中支持
- 如需使用BNSD格式的KV cache，请先转换为TND格式
- 转换方法：将BNSD [B, N, S, D] reshape为TND [B*S, N, D]

## 稀疏模式说明

### SelectIdx索引格式

RainFusionAttention使用新的selectIdx索引格式，不需要传统的sBlockIdx计算:

- **selectIdx**: 形状为`[T, headNum, maxKvblockNum]`的稀疏块索引数组
- **selectNumIdx**: 形状为`[T, headNum]`的稀疏块数量数组
- **T**: 所有batch中Q方向切块的总数(按blockShapeX计算)
- **blockShapeX**: Q方向基本块大小
- **blockShapeY**: KV方向基本块大小

### 示例

对于qSeqlen=256, kvSeqlen=512, blockShapeX=64, blockShapeY=64:

```
Q方向: ceil(256/64)=4块 [0, 1, 2, 3]
KV方向: ceil(512/64)=8块 [0, 1, 2, 3, 4, 5, 6, 7]

每个Q块可以选择任意KV块组合：
- Q块0选择的KV块: [0, 2, 5]
- Q块1选择的KV块: [1, 3, 6]
- Q块2选择的KV块: [0, 4, 7]
- Q块3选择的KV块: [2, 5, 6]
```

## 性能特点

- **内存效率**: 相比密集注意力，显著减少内存使用
- **计算效率**: 只计算指定的稀疏块，减少计算量
- **灵活配置**: 支持任意blockShape的稀疏块配置
- **硬件优化**: 充分利用昇腾A2的AI Core和Vector Core

## 注意事项

1. **KV Cache格式**: 当前仅支持TND格式，BNSD格式暂不支持
2. **Q输入格式**: 支持BSH和TND格式，BNSD格式暂不支持
3. qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景
4. 实际分块数通过向上取整计算: `ceil(qSeqlen/blockShapeX)`和`ceil(kvSeqlen/blockShapeY)`
5. 稀疏块索引必须在有效范围内
6. 最后一块可能不完整，系统会自动处理边界情况
7. 建议稀疏度不要过低，以保持计算效率
8. kvInputLayout参数必须设置为"TND"

## 编译说明

算子编译集成在ops-transformer-dev的统一编译流程中:

```bash
bash build.sh --soc=ascend910b
```

## 版本历史

- v1.0.0: 初始版本，从catlass 32_sparse_attention_infer迁移
