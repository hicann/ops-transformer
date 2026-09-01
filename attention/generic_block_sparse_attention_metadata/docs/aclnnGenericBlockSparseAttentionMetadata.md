# aclnnGenericBlockSparseAttentionMetadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：生成`aclnnGenericBlockSparseAttention`计算所需的`metadataOptional`。
- 输出`metadataOptional`必须作为主算子`aclnnGenericBlockSparseAttention`的`metadataOptional`输入使用；每次调用主算子前均须重新生成。
- `metadataOptional`为不透明数据，调用者不应解析或修改其中的内容。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize`获取workspace大小，再调用`aclnnGenericBlockSparseAttentionMetadata`执行计算。

```cpp
aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    int64_t maxQSeqLen,
    int64_t maxKvSeqLen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    const char *layoutQ,
    const char *layoutKv,
    int64_t maskType,
    int64_t quantType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    const aclTensor *metadataOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1800px"><colgroup>
<col style="width: 190px"><col style="width: 90px"><col style="width: 420px"><col style="width: 430px">
<col style="width: 100px"><col style="width: 90px"><col style="width: 280px"><col style="width: 100px">
</colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th></tr></thead><tbody>
<tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引数组，指定每个Q块选择的KV块索引。</td>
      <td>
        存储每个Q块选择的KV块索引，支持的shape随query布局变化：
        <ul>
          <li>query为TND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks, maxSparseBlockCount]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</li>
          </ul>
          <li>query为BNSD/BSND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX), maxSparseBlockCount]</li>
          </ul>
        </ul>
        其中 totalQBlocks = Σ ceilDiv(qStorageLen_i, blockShapeX)，i 为 batch 索引；qStorageLen_i 为 cuSeqLengthsQOptional 前缀和差分得到的<strong>存储长度</strong>（与seqused双长度语义见约束说明）。
<br>maxSparseBlockCount为sparseBlockCount tensor中所有元素的最大值，即所有Q块选择的KV块数量的最大值。传入值须 >= 该最大值，当前上限为256。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
<tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量。</td>
      <td>
        存储每个Q块实际选择的KV块数量，支持的shape随query布局变化：
        <ul>
          <li>query为TND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[headNum, totalQBlocks]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[numKeyValueHeads, totalQBlocks]</li>
          </ul>
          <li>query为BNSD/BSND布局时：</li>
          <ul>
            <li>每个qHead对应的KV稀疏pattern不一致（isPackedGQA=0）：<br>[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX)]</li>
            <li>GQA/MQA下，同group每个qHead对应的KV稀疏pattern一致（isPackedGQA=1）：<br>[batch, numKeyValueHeads, ceilDiv(maxQSeqLength, blockShapeX)]</li>
          </ul>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2/3</td>
      <td>×</td>
    </tr>
<tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutQ为"TND"时：该项输入必须配置</li>
          <li>当layoutQ为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照query的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
<tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度，以前缀和形式存储。</td>
      <td>可选输入，用于变长序列场景：
        <ul>
          <li>当layoutKv为"TND"/"PAGED_BBND"/"PAGED_BNBD"时：该项输入必须配置</li>
          <li>当layoutKv为"BNSD""BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；<br>如不配置该项输入(传入nullptr)，算子内会按照key/value的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
<tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>各batch中query的实际序列长度。</td>
      <td>
        <ul>
          <li>不指定序列长度可传入nullptr，表示和cuSeqLengthsQOptional差分得到的存储长度相同。</li>
          <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
<tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各batch中kv的实际序列长度。</td>
      <td>
        <ul>
          <li>不指定序列长度可传入nullptr，表示和cuSeqLengthsKvOptional差分得到的存储长度相同。</li>
          <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
<tr>
      <td>maxQSeqLen</td>
      <td>输入</td>
      <td>主算子Query的最大Sequence Length。</td>
      <td>必须大于0，并与主算子Query及长度类输入保持一致。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>maxKvSeqLen</td>
      <td>输入</td>
      <td>主算子Key/Value的最大Sequence Length。</td>
      <td>必须大于0，并与主算子Key/Value及长度类输入保持一致。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>numQHeads</td>
      <td>输入</td>
      <td>主算子Query的head数。</td>
      <td>必须大于0，且须满足`numQHeads >= numKvHeads`、`numQHeads % numKvHeads == 0`和`numQHeads / numKvHeads <= 128`。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>numKvHeads</td>
      <td>输入</td>
      <td>主算子Key/Value的head数。</td>
      <td>必须大于0，并与主算子Key/Value的head数保持一致。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>headDim</td>
      <td>输入</td>
      <td>主算子Query、Key和Value每个head的特征维度。</td>
      <td>当前仅支持128。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>blockShape</td>
      <td>输入</td>
      <td>代表稀疏块形状数组。</td>
      <td>含两个元素[blockShapeX, blockShapeY]。<br>blockShapeX支持任意值，不可超过int64表示范围。<br>blockShapeY支持按16对齐的任意值，不可超过int64表示范围。<br>开启量化时的额外约束见「量化相关说明」。<br>当前仅支持blockShape=[1,128]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>isPackedGQA</td>
      <td>输入</td>
      <td>代表进行块状稀疏时，同一个group内的qHead是否共享同样的稀疏pattern<br>（注：不同batch之间不会共享同样的稀疏pattern，该入参仅区分head维度的共享情况）。</td>
      <td>若取值为0，则代表同一个group内的qHead不共享同样的稀疏pattern；<br>若取值为1，则代表同一个group内的qHead共享同样的稀疏pattern。<br>当前仅支持1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND"，详见「layout对应关系说明」。当前仅支持"TND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>目标支持"TND""BNSD""BSND""PAGED_BBND""PAGED_BNBD"，详见「layout对应关系说明」。当前仅支持"PAGED_BBND"。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>maskType</td>
      <td>输入</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
        取值0~5，含义见「掩码说明」。<br>
        当前仅支持1（causal mask）。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>quantType</td>
      <td>输入</td>
      <td>代表采用的量化手段。</td>
      <td>
        取值0~5，完整配置见「量化相关说明」：<br>
        当前可用：0；Ascend 950上可选5。取值1~4传入将校验失败。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>softmaxPrecision</td>
      <td>输入</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。取值0或1：
        <ul>
          <li>0：online softmax和rescale全部采取fp32，适合追求计算精度的场景。</li>
          <li>1：混合精度；online softmax采取fp16/bf16（与attentionOut相同），rescale采取fp32，online softmax阶段可能数值溢出。</li>
        </ul>
        芯片约束见「约束说明」（Ascend 950仅1；Atlas A2/A3上FP16可0/1，BF16仅0；FP8路径仅1）。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>winLeft</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见「掩码说明」。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>winRight</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention；不使能时必须为-1，需与maskType配合，见「掩码说明」。当前只支持传入-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>AICPU算子aclnnGenericBlockSparseAttentionMetadata的分核结果。</td>
      <td>
        必须传入。由aclnnGenericBlockSparseAttentionMetadata算子生成，每次调用主算子前须重新生成。调用者须预先申请INT32、shape为[1024]的Device侧Tensor；其内容为不透明格式，不应解析、修改或跨不同输入和属性复用。
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
<tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回Device侧workspace大小。</td>
      <td>调用者应按返回值申请workspace。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
<tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回包含算子计算流程的执行器。</td>
      <td>不可为空。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
</tbody></table>

- **返回值**

  返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnGenericBlockSparseAttentionMetadata

- **参数说明**

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 900px">
</colgroup>
<thead>
<tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
</thead>
<tbody>
<tr>
      <td>workspace</td>
      <td>输入</td>
      <td>Device侧workspace地址；workspaceSize为0时可传nullptr。</td>
    </tr>
<tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>第一段接口返回的workspace大小。</td>
    </tr>
<tr>
      <td>executor</td>
      <td>输入</td>
      <td>第一段接口返回的执行器。</td>
    </tr>
<tr>
      <td>stream</td>
      <td>输入</td>
      <td>执行任务的ACL Stream。</td>
    </tr>
</tbody>
</table>

- **返回值**

  返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 本接口必须与`aclnnGenericBlockSparseAttention`配套使用。所有对应的Tensor和属性须与随后调用的主算子完全一致，每次调用主算子前均须重新生成`metadataOptional`。
- `layoutQ`当前仅支持`"TND"`；`layoutKv`当前仅支持`"PAGED_BBND"`。
- `isPackedGQA`当前仅支持1。
- `maskType`当前仅支持1，表示算子内置causal mask；`winLeft`和`winRight`当前仅支持-1。
- `blockShape`当前仅支持`[1, 128]`，`headDim`当前仅支持128。
- 当前TND且`isPackedGQA=1`时，`sparseBlockIdx`的shape为`[numKvHeads, totalQBlocks, maxSparseBlockCount]`，`sparseBlockCount`的shape为`[numKvHeads, totalQBlocks]`。`totalQBlocks`按`cuSeqLengthsQOptional`前缀和差分得到的Query存储长度分块；`maxSparseBlockCount`须不小于`sparseBlockCount`中所有元素的最大值，且当前不超过256。
- `cuSeqLengthsQOptional`和`cuSeqLengthsKvOptional`当前必须传入，两者均为前缀和形式。
- `sequsedQOptional`和`sequsedKvOptional`可选。不传时，实际长度等于对应cu前缀和差分得到的存储长度；传入时，分核和任务空间按各Batch实际有效长度累加，稀疏块分块与QKV寻址仍按cu存储长度计算，每个元素须不大于对应Batch的存储长度。
- `quantType`当前支持0；Ascend 950上可选5。取值1~4当前不支持。取5时，主算子Query、Key、Value须为FLOAT8_E4M3FN，attentionOut须为FLOAT16或BFLOAT16。
- `softmaxPrecision`取值为0或1：Ascend 950仅支持1；Atlas A2/A3上，FLOAT16支持0或1，BFLOAT16仅支持0；FLOAT8路径仅支持1。
- `numQHeads`和`numKvHeads`须满足`numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`；`groupSize = numQHeads / numKvHeads`当前须不超过128。
- 输出`metadataOptional`固定为INT32、shape为`[1024]`，对应主算子的`metadataOptional`输入，不得解析、修改或跨不同输入和属性复用。

## 调用示例

以下示例展示两段式接口调用顺序。Tensor创建、输入拷贝和资源释放方式请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

aclnnStatus RunMetadata(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                        const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
                        aclTensor *metadataOptional, aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_BAD_ALLOC;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, nullptr, nullptr,
        16, 2048, 32, 8, 128, blockShape, 1, "TND", "PAGED_BBND",
        1, 0, 0, -1, -1, metadataOptional, &workspaceSize, &executor);
    aclDestroyIntArray(blockShape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
