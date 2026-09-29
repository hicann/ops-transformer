# aclnnSparseLightningIndexer

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id3 -->

## 功能说明

aclnnSparseLightningIndexer 实现 candidate consumer 语义：在
LightningIndexerV2 加权 ReLU 打分框架（score = Σg W_g·ReLU(Q_g·Kᵀ)）上消费上游 source 算子
（aclnnLightningIndexerV2 candidateTopkBlocks 开启时输出的 candidateTopkIndices）的候选块索引，
对候选外可达位置做 NEG_HUGE 泄漏（leak）降级后走原 topk 管线。
算子仅注册 ascend910b / ascend910_93 平台（arch22）。

候选块 j 表示该行可达前缀在压缩后 K 位置空间上的区间
[j×candidateBlockSize, (j+1)×candidateBlockSize)；块号按 batch 内相对编号消费，
每行的块总数 numBlocks = ceil(actS2 / candidateBlockSize)，
actS2 为由 sequsedKOptional/cuSeqlensKOptional/k shape 与 cmpRatio（含 cmpResidualKOptional
余数）推算的该 batch 压缩后 K 有效长度。

## 接口定义

算子执行接口为[两段式接口](../../../../docs/zh/context/two_phase_api.md)。

### aclnnSparseLightningIndexerGetWorkspaceSize

```cpp
aclnnStatus aclnnSparseLightningIndexerGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional, const aclTensor *metadataOptional,
    const aclTensor *candidateTopkIndices, const aclTensor *candidateBlockLengthOptional,
    int64_t topk, int64_t maxSeqlenQ, char *layoutQOptional, char *layoutKOptional,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue, int64_t candidateBlockSize,
    const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```

### aclnnSparseLightningIndexer

```cpp
aclnnStatus aclnnSparseLightningIndexer(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 | 约束 |
|---|---|---|---|
| q | 输入 | 查询张量 | BF16/FP16，与 k 类型一致；BSND [B,S1,N1,D] / TND [T,N1,D]；D=128 |
| k | 输入 | 键张量 | 与 q 同 dtype；BSND [B,S2,N2,D] / TND [T,N2,D] / PA_BBND [blockNum,blockSize,N2,D]；N2=1；PA_BBND 时 blockNum≠0、blockSize 为 (0,1024] 内 16 的倍数；PA_BBND 时仅允许 0 轴非连续，非 PA 时必须全连续 |
| w | 输入 | 权重张量 | FP32；BSND [B,S1,N1] / TND [T,N1] |
| cuSeqlensQOptional | 输入 | q 有效长度前缀和 | INT32 (B+1,)；layoutQ=TND 时必传；BSND 时不建议传入 |
| cuSeqlensKOptional | 输入 | k 有效长度前缀和 | INT32 (B+1,)；layoutK=TND 时必传、BSND 时传入报错、PA_BBND 时不传 |
| sequsedQOptional | 输入 | q 每 batch 实际长度 | INT32 (B,)；接口对齐保留，本算子（arch22）kernel 不消费，q 长度以 shape S1 或 cuSeqlensQOptional 为准 |
| sequsedKOptional | 输入 | k 每 batch 实际长度 | INT32 (B,)；PA_BBND 必传；BSND/TND 可选传入，传入则参与 k 有效长度计算（actS2Orig = sequsedK×cmpRatio + cmpResidualK） |
| cmpResidualKOptional | 输入 | k 压缩余数 | INT32 (B,)；host 侧不校验 shape/dtype（跨算子契约），与 sequsedKOptional/cuSeqlensKOptional 组合参与压缩长度计算 |
| blockTableOptional | 输入 | PA block 映射表 | INT32 二维 (B, maxBlockNum)，dim0 必须等于 B；PA_BBND 必传、非 PA 时传入报错 |
| outputIdxOffsetOptional | 输入 | 输出索引偏移 | INT32；接口对齐保留，本算子（arch22）kernel 不消费，输出索引不加偏移 |
| metadataOptional | 输入 | 元数据 | arch22 不消费且非空即 host 拒绝（输入位仅为 IR 兼容保留），传 nullptr |
| candidateTopkIndices | 输入 | **候选块索引（必选，不可为空）** | INT32；BSND [B,S1,N2,candBlocks]（rank 4，前三维与 B/S1/N2 逐一校验）/ TND [query.dim0,N2,candBlocks]（rank 3）；candBlocks = shape 末维 ∈ (0,2048] 且 64 的倍数。值域 [0,numBlocks) 或 -1、槽位无序、允许重复/越界/-1 脏数据为跨算子契约，host/device 均不校验：越界块号不命中任何窗口按非候选处理，重复块号结果不变，-1 沉底不入选 |
| candidateBlockLengthOptional | 输入 | 预留 | 仅接受 nullptr 或元素个数为 0 的空 tensor（任意 shape，numel==0），非空报错 |
| topk | 属性 | Top-k 的 k | (0, 2048]（恒校验，无 over-2K 回退路径） |
| maxSeqlenQ | 属性 | q 最大序列长度 | 仅接受 -1（缺省），非 -1 即 host 拒绝（arch22 不消费） |
| layoutQ / layoutK | 属性 | 数据排布 | layoutQ：BSND/TND；layoutK：BSND/TND/PA_BBND；layoutK 非 PA_BBND 时两侧必须一致；缺省均为 BSND |
| maskMode | 属性 | mask 类型 | 0（无）/ 3（因果，rightDownCausal） |
| cmpRatio | 属性 | 压缩比 | host 校验 (0,128] 且必须为 2 的幂（非幂值拒绝，与 source 侧一致） |
| returnValue | 属性 | 必须 0 | ≠0 报错拒绝（leak 降级语义下 Values 输出无意义） |
| candidateBlockSize | 属性 | 候选块粒度 | [2,64] 内的 2 的幂，缺省 8；必须与上游 source 算子的 candidateBlockSize 一致 |
| sparseIndicesOut | 输出 | topk 索引 | INT32；BSND [B,S1,N2,topk] / TND [T,N2,topk]；无效位填 -1 |
| sparseValuesOut | 输出 | 恒空 | FP32；aclnn 层仍做非空校验，须传入非空 tensor，实际输出 shape (0,)，kernel 不写入 |

- **返回值：** aclnnStatus，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。第一段接口完成入参校验，参数为空指针时报 ACLNN_ERR_PARAM_NULLPTR(161001)，数据类型/shape/属性取值非法时报 ACLNN_ERR_PARAM_INVALID(161002)。

## 输出语义（leak）

- 候选内可达位置：保持原始加权 ReLU 分数参与排序；
- 候选外**可达**位置：分数降级为 NEG_HUGE（-1e30f，位型 0xF149F2CA）参与排序（不取消入选资格，仍可作填充入选）；
- 不可达位置（p ≥ 该行可达前缀长度，含 causal/cmpRatio 前缀之外的位置与 BSND padding 行）：以 (-inf, -1) 参与，最终输出槽位填 -1；两档顺序保证：NEG_HUGE > -inf，候选外降级位优先于不可达位入选；
- regime 2（候选内有效数 < topk < 可达数）：候选外可达位置以降级分数填充入选；
- 与模型 `where(idxs < compress_lens, idxs+offset, -1)` 等价。

## 已知限制

- 输出契约：仅保证有效槽**集合**一致与 -1 槽数正确；槽位顺序（含 regime 2 候选外填充次序）为
  实现细节，不作对外承诺。
- 性能提示：超长上下文（如 candidateBlockSize=8、压缩后 S2 超过约 22016）时掩码计算走兼容路径，
  语义等价，性能收益受限。
- seqused_q / output_idx_offset / metadata / max_seqlen_q≠-1 由 arch22 host 侧校验拒绝。

## 跨算子契约

- candidateTopkIndices 直接对接 aclnnLightningIndexerV2（开启 candidateTopkBlocks 时）的 candidateTopkIndicesOut 输出，块号为同一 forward 内 batch 相对压缩块号，槽位无序（消费侧按集合消费）；
- 候选块号不含 output_idx_offset 偏移；
- candidateBlockSize、cmpRatio 必须与 source 侧配置一致；
- 候选宽度 candBlocks 由输入 tensor shape 末维推导（本算子无 candidate_topk_blocks 属性）。

## 调用示例

参见 [../examples/test_aclnn_sparse_lightning_indexer.cpp](../examples/test_aclnn_sparse_lightning_indexer.cpp)
（覆盖基本 consumer 调用 + 空 candidate_block_length 传法）。

关键序列：

```cpp
// candidate_block_length 空传法（预留接口：numel==0 即可）
std::vector<int64_t> shape = {0};
std::vector<int64_t> strides = {1};
aclTensor *candidateBlockLengthTensor =
    aclCreateTensor(shape.data(), 1, ACL_INT32, strides.data(), 0, ACL_FORMAT_ND,
                    shape.data(), 1, nullptr);

aclnnSparseLightningIndexerGetWorkspaceSize(
    ..., /* metadataOptional */ nullptr, candidateTopkIndicesTensor, candidateBlockLengthTensor,
    topk, maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio,
    /* returnValue */ 0, /* candidateBlockSize */ 8,
    sparseIndicesTensor, sparseValuesTensor, &workspaceSize, &executor);
aclnnSparseLightningIndexer(workspace, workspaceSize, executor, stream);
```
