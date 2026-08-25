# aclnnBlockSparseAttentionV3

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **接口功能**：BlockSparseAttentionV3稀疏注意力计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。

  相比于BlockSparseAttentionV3,本接口新增pQuantScaleOptional、quantMode、dstTypeMax参数。
- **计算公式**：稀疏块大小：$blockShapeX \times blockShapeY$，selectIdx指定稀疏模式

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  BlockSparseAttentionV2输入query、key、value的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。

  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

  当前支持的布局：

  - qInputLayout: "TND" "BNSD" "BSND"
  - kvInputLayout: "TND" "BNSD" "BSND"
- **MXFP4特性说明(仅Ascend 950PR/Ascend 950DT支持)**

  本接口新增支持MXFP4数据类型(FLOAT4_E2M1)的输入，以提供计算效率并降低显存占用。当使用MXFP4输入时，需要提供相应的量化缩放因子用于反量化计算。
- <summary><a id="MXFP4量化模式的量化缩放因子"></a><strong>MXFP4量化模式的量化缩放因子</strong></summary>

  - qDequantScaleOptional：

    - MX 量化模式，仅支持Rowwise E8M0 Micro Scaling
    - 数据类型: FLOAT8_E8M0
    - shape:
      - BNSD: [B, Nq, Sq, CeilDiv(D, 64), 2]；
      - BSND: [B, Sq, Nq, CeilDiv(D, 64), 2]；
      - TND: [Tq, Nq, CeilDiv(D, 64), 2]。
  - kDequantScaleOptional：

    - MX 量化模式，仅支持Rowwise E8M0 Micro Scaling
    - 数据类型: FLOAT8_E8M0
    - shape:
      - BNSD:[B, Nkv, Skv, CeilDiv(D, 64), 2]；
      - BSND:[B, Skv, Nkv, CeilDiv(D, 64), 2]；
      - TND: [Tkv, Nkv, CeilDiv(D, 64), 2]。
  - vDequantScaleOptional：

    - MX 量化模式，仅支持Columnwise E8M0 Micro Scaling
    - 数据类型: FLOAT8_E8M0
    - shape

      - BNSD:[B, Nkv, CeilDiv(Skv, 64), D, 2]；
      - BSND:[B, CeilDiv(Skv, 64), Nkv, D, 2]；
      - TND: [BTkv//64, Nkv, D, 2]。

      $$
      BTkv = \sum_{b=0}^{B-1} \left\lceil \frac{Skv_b}{64} \right\rceil
      $$

## 函数原型

每个算子分为[两段式接口](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md)，必须先调用"aclnnBlockSparseAttentionV3GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBlockSparseAttentionV3"接口执行计算。

```c++
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    const aclTensor *pQuantScaleOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t softmaxLseFlag,
    int64_t quantMode,
    double dstTypeMax,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```c++
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnBlockSparseAttentionV3GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
  <col style="width: 269px">
  <col style="width: 132px">
  <col style="width: 270px">
  <col style="width: 350px">
  <col style="width: 185px">
  <col style="width: 119px">
  <col style="width: 146px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的query。</td>
      <td>支持的shape为：
        <ul><li>TND: [totalQTokens, headNum, headDim]。</li>
        <li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li>
        <li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1 </td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>支持的shape为：
        <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1、 </td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>
        支持的shape为：
        <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li>
          <li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT4_E2M1、</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示实际的稀疏pattern。</td>
      <td>
        可选输入（当前版本为必选）
        <ul>
          <li>shape为[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]。</li>
          <li>表示按block划分后哪些block需要参与计算（为1），哪些block不参与计算（为0）。</li>
          <li>如传入nullptr，则视为不开启块稀疏计算，即所有token之间的注意力分数都会被计算。</li>
        </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的atten_mask。</td>
      <td>atten_mask会与稀疏pattern叠加产生作用。当前不支持，必须传入nullptr。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockShapeOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>稀疏块形状数组。</td>
      <td>
        与blockSparseMaskOptional配合使用：
        <ul>
          <li>当配置了blockSparseMaskOptional时：如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为[128,128]。</li>
          <li>当未配置blockSparseMaskOptional时：无论此项如何配置，算子均将忽略。</li>
        </ul>
        当配置此输入时：必须包含两个元素[blockShapeX, blockShapeY]
        <ul>
          <li>blockShapeX: Q方向块大小，值必须大于0，MXFP4的量化时只支持64的倍数。</li>
          <li>blockShapeY: KV方向块大小，值必须大于0；在Ascend 950PR/Ascend 950DT上须为16的倍数，MXFP4的量化时只支持64的倍数，在Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品上须为128的倍数。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>描述每个Batch对应的query序列长度。</td>
      <td>
        可选输入，用于变长序列场景：
        <ul>
          <li>当qInputLayout为"TND"时：该项输入必须配置。</li>
          <li>当qInputLayout为"BNSD"或"BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入（传入nullptr），算子内会按照query的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>描述每个Batch对应的key/value序列长度。</td>
      <td>
        可选输入，用于变长序列场景：
        <ul>
          <li>当kvInputLayout为"TND"时：该项输入必须配置。</li>
          <li>当kvInputLayout为"BNSD"或"BSND"时：如配置该项输入，算子内会按该输入指定的实际序列长度进行处理；如不配置该项输入（传入nullptr），算子内会按照key/value的shape中的S进行处理。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>Block表用于PagedAttention。</td>
      <td>当前不支持，必须传入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>query的反量化缩放因子。</td>
      <td>
        可选输入：
        <ul>
          <li>当quantMode不为0时，此参数必须传入。</li>
          <li>当quantMode为0时，此参数应传入nullptr。</li>
        </ul>
        当配置此输入时：
        <ul>
          <li>当quantMode=1, shape为[batch, headNum, ceilDiv(maxQSeqLength, 128), 1]。</li>
          <li>quantMode=2/3, 约束见<a href="#MXFP4量化模式的量化缩放因子">MXFP4量化模式的量化缩放因子。</a></li>
        </ul>
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>key的反量化缩放因子。</td>
      <td>
        可选输入：
        <ul>
          <li>当quantMode不为0时，此参数必须传入。</li>
          <li>当quantMode为0时，此参数应传入nullptr。</li>
        </ul>
        当配置此输入时：
        <ul>
          <li>当quantMode=1,shape为[batch, kvHeadNum, ceilDiv(maxKVSeqLength, 256), 1]或shape为[batch, kvHeadNum, ceilDiv(maxKVSeqLength, 512), 1]。</li>
          <li>quantMode=2/3, 约束见<a href="#MXFP4量化模式的量化缩放因子">MXFP4量化模式的量化缩放因子。</a></li>
        </ul>
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>value的反量化缩放因子。</td>
      <td>
        可选输入：
        <ul>
          <li>当quantMode不为0时，此参数必须传入。</li>
          <li>当quantMode为0时，此参数应传入nullptr。</li>
        </ul>
        当配置此输入时：
        <ul>
          <li>当quantMode=1, shape为[batch, kvHeadNum, ceilDiv(maxKVSeqLength, 256), 1]或shape为[batch, kvHeadNum, ceilDiv(maxKVSeqLength, 512), 1]。</li>
         <li>quantMode=2/3, 约束见<a href="#MXFP4量化模式的量化缩放因子">MXFP4量化模式的量化缩放因子。</a></li>
        </ul>
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4/5</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>softmax结果p的量化缩放因子。</td>
      <td>
        可选输入：
        <ul>
          <li> 预留参数, 暂不支持, 此参数应传入nullptr。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qInputLayout（char*）</td>
      <td>输入</td>
      <td>代表输入query的数据排布格式。</td>
      <td>当前仅支持"TND"和"BNSD"和"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout（char*）</td>
      <td>输入</td>
      <td>代表输入key、value的数据排布格式。</td>
      <td>当前仅支持"TND"和"BNSD"和"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads（int64_t）</td>
      <td>输入</td>
      <td>代表key/value的head个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType（int64_t）</td>
      <td>输入</td>
      <td>表示attention计算中的掩码类型。</td>
      <td>
        当前只支持传0。
        <ul>
          <li>0：代表不加mask场景</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue（double）</td>
      <td>输入</td>
      <td>公式中的scale，代表缩放系数。</td>
      <td>一般设置为D^-0.5。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise（int64_t）</td>
      <td>输入</td>
      <td>Softmax计算采取的精度级别。</td>
      <td>
        控制online softmax阶段以及rescale阶段运算使用的数据类型。当前只支持传0或1或4，其中，<term>Ascend 950PR/Ascend 950DT</term>仅支持配置为4，<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持配置为0或1
        <ul>
          <li>0：表示online softmax和rescale全部采取fp32数据类型，适合追求计算精度的场景使用。</li>
          <li>1：仅支持输入的query、key、value均为fp16数据类型时配置，表示online softmax和rescale全部采取fp16数据类型，性能更好，但精度较低，且可能发生计算时的数值溢出，使用者需根据值域范围自行判断是否使用。</li>
          <li>4：表示混合精度运算，在性能与精度上取得一个折中。online softmax采取fp16/bf16数据类型（与query、key、value数据类型相同），rescale采取fp32数据类型，在online softmax阶段可能发生数值溢出。mxfp4量化模式下（quantMode=2/3），online softmax仅支持fp16数据类型。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>PagedAttention的block大小。</td>
      <td>用于PagedAttention场景，当前不支持PagedAttention功能，因此只支持传0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens（int64_t）</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向前包含多少个token。</td>
      <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens（int64_t）</td>
      <td>输入</td>
      <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
      <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag（int64_t）</td>
      <td>输入</td>
      <td>是否使能softmaxLse输出的标志位。</td>
      <td>
        当前只支持传0或1。
        <ul>
          <li>0：表示不输出softmaxLse。</li>
          <li>1：表示输出softmaxLse，相比不输出softmaxLse可能存在性能损失。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式选择标志位。</td>
      <td>
        当前只支持传0、1、2、3。
        <ul>
          <li>0：表示非量化。</li>
          <li>1：表示FP8量化。</li>
          <li>2：表示MXFP4 OCP量化(量化scale向下截断)。</li>
          <li>3：表示MXFP4 CX量化(自定义量化量程, 量化scale向上截断)。</li>
          <li>详见<a href="#constraint_rule">约束说明</a>。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（double）</td>
      <td>输入</td>
      <td>MXFP4 CX量化时,传入的自定义量化量程。</td>
      <td>
        当前仅当quantMode=2,3时候，支持设置该值0.0，或者[6.0, 12.0]。
        <ul>
          <li>0.0：代表Amax(DType)为量化结果数据类型的最大值。</li>
          <li>[6.0, 12.0]：取值为6.0-12.0代表Amax(DType)为传入值。</li>
        </ul>
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的attentionOut。</td>
      <td>数据类型和shape与query保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional（aclTensor*）</td>
      <td>输出</td>
      <td>Softmax计算的log-sum-exp中间结果。</td>
      <td>
        支持的shape随着query的shape改变：
        <ul>
          <li>query为"TND": [totalQTokens, headNum, 1]。</li>
          <li>query为"BNSD": [batch, headNum, maxQSeqLength, 1]。</li>
          <li>query为"BSND": [batch, maxQSeqLength, headNum, 1]。</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值**
  aclnnStatus：
  返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入query，key，value传入的是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>query，key，value数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>qInputLayout或kvInputLayout不合法。</td>
    </tr>
    <tr>
      <td>blockShape不合法（元素数量少于2或值小于等于0）。</td>
    </tr>
    <tr>
      <td>innerPrecise不合法（必须为0、1或4）。</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockSparseAttentionV3

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockSparseAttentionV3GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>
- **返回值**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明<a id="constraint_rule"></a>

- 确定性计算：

  - aclnnBlockSparseAttentionV3默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- qInputLayout当前仅支持"TND"和"BNSD"和"BSND"。
- kvInputLayout当前仅支持"TND"和"BNSD"和"BSND"。
- 当前query、key、value的InputLayout必须保持一致。
- 输入query、key、value的数据类型必须一致，支持FLOAT16和BFLOAT16。
- query、key、value的D轴当前仅支持配置为64或128
- blockShapeOptional如果传入，则必须包含至少两个元素[blockShapeX, blockShapeY]，且值必须大于0，blockShapeY必须为128的倍数。
- blockSparseMaskOptional当前必须传入，且shape必须为[batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)]。
- attentionMaskOptional当前只支持传入nullptr。
- actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
- actualSeqLengthsOptional与actualSeqLengthsKvOptional当前必须同时配置或同时不配置，仅配置其中之一的行为将被算子拦截。
- blockTableOptional当前只支持传入nullptr，表示不开启PagedAttention特性。
- innerPrecise仅支持配置4，表示混合精度运算，在性能与精度上取得一个折中。
- softmaxLseFlag仅支持配置0或1，分别表示不开启/开启softmaxLse输出。
- qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景，实际分块数通过向上取整计算。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0。
- maskType当前只支持输入0，表示不加mask。
- blockSize当前只支持输入0，表示不支持paged cache。
- preTokens和nextTokens当前只支持输入2147483647，表示当前token的前后所有token都参与attention运算，即不支持滑窗attention。
- **quantMode=1(FP8静态量化模式)相关约束**：

  - 仅Ascend 950PR/Ascend 950DT支持。
  - 输入的query、key、value采用FLOAT8_E4M3FN数据类型，需要提供以下量化缩放因子参数：
  - 输入的query、key、value采用FLOAT8_E4M3FN数据类型，需要提供以下量化缩放因子参数：
    - qDequantScale（query量化缩放因子）

      - 数据类型：FLOAT32
      - shape：(Batch, HeadNum, CeilDiv(maxQSeqLength, 128), 1)
      - 用途：在QK矩阵乘法时对query进行反量化。
    - kDequantScale（key量化缩放因子）

      - 数据类型：FLOAT32
      - shape：(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 256), 1)或(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 512), 1)
      - 用途：在QK矩阵乘法时对key进行反量化。
    - vDequantScale（value量化缩放因子）

      - 数据类型：FLOAT32
      - shape：(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 256), 1)或(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 512), 1)
      - 用途：在PV矩阵乘法时对value进行反量化。
  - 当query、key、value中任意一个数据类型为FLOAT8_E4M3FN时，query、key、value必须同时为FLOAT8_E4M3FN数据类型。
  - 使用FP8输入时，必须提供对应的量化缩放因子输入qDequantScale、kDequantScale、vDequantScale。
  - 量化缩放因子的数据类型必须为FLOAT32。
  - qDequantScale的shape必须为(Batch, HeadNum, CeilDiv(maxQSeqLength, 128), 1)。
  - kDequantScale和vDequantScale的shape必须一致，为(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 256), 1)或(Batch, KVHeadNum, CeilDiv(maxKVSeqLength, 512), 1)。
  - 当query、key、value中任意一个数据类型不为FLOAT8_E4M3FN时，qDequantScale、kDequantScale、vDequantScale必须传入nullptr。
  - blockShapeOptional必须传入。
  - q和kv的量化块大小必须与blockShapeOptional的两个元素大小分别保持一致。
- **quantMode=2/3 (MXFP4量化模式)相关约束（新增）**

  - 仅Ascend 950PR/Ascend 950DT支持
  - blockShapeX 仅支持64的倍数；
  - D仅支持64/128;
  - attentionOut数据类型仅支持FLOAT16或BFLOAT16；
  - 暂不支持attenMaskOptional、PagedAttention、softmaxLse等高阶特性。
  - **量化公式**
    - 以P的量化为例, 其中 $P_{\text{max}}$ 为一个group（32个元素）内元素绝对值的最大值。
    - **OCP 量化公式**

      $$ P_{\text{Scale}} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor / 4} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor - 2} $$

    - **CX量化公式**

      $$ P_{\text{Scale}} = 2^{\lceil \log_2 \left( P_{\text{max}} / dstTypeMax \right) \rceil} = 2^{\lceil \log_2 P_{\text{max}} - \log_2 dstTypeMax \rceil} $$

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_v3.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
            return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *value, aclTensor *blockSparseMask,
                  aclTensor *attentionOut, aclIntArray *actualSeqLengths, aclIntArray *actualSeqLengthsKv,
                  aclIntArray *blockShape, void *queryDeviceAddr, void *keyDeviceAddr, void *valueDeviceAddr,
                  void *blockSparseMaskDeviceAddr,void *attentionOutDeviceAddr, void *actualSeqLengthsDeviceAddr,
                  void *actualSeqLengthsKvDeviceAddr, void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    // 释放资源
    if (query) {
        aclDestroyTensor(query);
    }
    if (key) {
        aclDestroyTensor(key);
    }
    if (value != nullptr) {
        aclDestroyTensor(value);
    }
    if (blockSparseMask) {
        aclDestroyTensor(blockSparseMask);
    }
    if (attentionOut) {
        aclDestroyTensor(attentionOut);
    }
    if (actualSeqLengths) {
        aclDestroyIntArray(actualSeqLengths);
    }
    if (actualSeqLengthsKv) {
        aclDestroyIntArray(actualSeqLengthsKv);
    }
    if (blockShape) {
        aclDestroyIntArray(blockShape);
    }

    if (queryDeviceAddr) {
        aclrtFree(queryDeviceAddr);
    }
    if (keyDeviceAddr) {
        aclrtFree(keyDeviceAddr);
    }
    if (valueDeviceAddr) {
        aclrtFree(valueDeviceAddr);
    }
    if (blockSparseMaskDeviceAddr) {
        aclrtFree(blockSparseMaskDeviceAddr);
    }
    if (attentionOutDeviceAddr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (actualSeqLengthsDeviceAddr) {
        aclrtFree(actualSeqLengthsDeviceAddr);
    }
    if (actualSeqLengthsKvDeviceAddr) {
        aclrtFree(actualSeqLengthsKvDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    // 1.（固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置参数
    const char *socName = aclrtGetSocName();
    std::string socVersion = (socName != nullptr) ? std::string(socName) : std::string();
    bool isArch22 = socVersion.find("Ascend910") != std::string::npos;
    int64_t innerPrecise = isArch22 ? 0 : 4;

    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 256;

    // 计算TND格式维度
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Q块的X维度数量
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // KV块的Y维度数量

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockSparseMaskTensor = nullptr;
    aclTensor *qDequantScaleTensor = nullptr;
    aclTensor *kDequantScaleTensor = nullptr;
    aclTensor *vDequantScaleTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclIntArray *blockShape = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockSparseMaskDeviceAddr = nullptr;
    void *qDequantScaleDeviceAddr = nullptr;
    void *kDequantScaleDeviceAddr = nullptr;
    void *vDequantScaleDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *actualSeqLengthsDeviceAddr = nullptr;
    void *actualSeqLengthsKvDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    // 3. 创建Query tensor (TND format: [totalQTokens, numHeads, headDim])
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(GetShapeSize(queryShape), 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 4. 创建Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(GetShapeSize(kvShape), 1.0f);
    std::vector<op::fp16_t> valueHostData(GetShapeSize(kvShape), 1.0f);
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 5. 创建blockSparseMask tensor ([batch, numHeads, qBlockNum, kvBlockNum])
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    std::vector<int8_t> blockSparseMaskHostData(GetShapeSize(blockSparseMaskShape), 1);
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape, &blockSparseMaskDeviceAddr, aclDataType::ACL_INT8, &blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create block sparse mask tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 6. 创建输出tensor
    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0f);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 7. 创建blockShape数组
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 8. 创建actualSeqLengths和actualSeqLengthsKv (必需参数)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));

    size_t seqLengthsSize = batch * sizeof(int64_t);

    ret = aclrtMalloc(&actualSeqLengthsDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    ret = aclrtMemcpy(actualSeqLengthsDeviceAddr, seqLengthsSize, actualSeqLengthsHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDeviceAddr, seqLengthsSize, actualSeqLengthsKvHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // aclCreateIntArray期望的是host侧的数据指针，而不是device侧的数据
    actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 9. 准备字符串参数（确保缓冲区大小足够，包含null terminator）
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);

    // 10. 计算scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));

    // 11. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockSparseAttentionV3GetWorkspaceSize(
        queryTensor,              // query
        keyTensor,                // key
        valueTensor,              // value
        blockSparseMaskTensor,    // blockSparseMask
        nullptr,                  // attenMaskOptional
        blockShape,               // blockShape
        actualSeqLengths,         // actualSeqLengthsOptional
        actualSeqLengthsKv,       // actualSeqLengthsKvOptional
        nullptr,                  // blockTableOptional
        nullptr,                  // qDequantScale
        nullptr,                  // kDequantScale
        nullptr,                  // vDequantScale
        nullptr,                  // pDequantScale
        qLayoutBuffer,            // qInputLayout
        kvLayoutBuffer,           // kvInputLayout
        numKvHeads,               // numKeyValueHeads
        0,                        // maskType
        scaleValue,               // scaleValue
        innerPrecise,             // innerPrecise（随芯片型号自动选择）
        0,                        // blockSize
        2147483647,               // preTokens
        2147483647,               // nextTokens
        0,                        // softmaxLseFlag
        0,                        // quantMode
        0.0,                      // dstTypeMax
        attentionOutTensor,       // attentionOut
        nullptr,                  // softmaxLseOptional
        &workspaceSize,           // workspaceSize (out)
        &executor);               // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 12. 分配workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                              actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                              valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                              actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 13. 调用第二段接口
    ret = aclnnBlockSparseAttentionV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionV3 failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 14. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 15. 获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);
    std::vector<op::fp16_t> resultData(attentionOutSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), attentionOutDeviceAddr,
                      attentionOutSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 16. 打印部分结果
    uint64_t printNum = 10;
    LOG_PRINT("attentionOut results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }

    // 17. 释放资源
    FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                  actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                  valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                  actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
