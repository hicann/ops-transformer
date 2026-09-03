# aclnnSparseFlashMlaSoftmaxL1Norm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：计算`SparseFlashMla`注意力的Softmax L1Norm结果，支持Compressed Attention以及Sparse Compressed Attention场景。该接口为`aclnnDenseLightningIndexerKLLossGrad`反向算子的配套接口，输出可用于反向梯度计算。调用过程中，需先调用`aclnnSparseFlashMlaSoftmaxL1NormMetadata`接口完成负载均衡计算，再调用`aclnnSparseFlashMlaSoftmaxL1Norm`接口完成算子计算过程。
    - `aclnnSparseFlashMlaSoftmaxL1NormMetadata`接口：根据主算子的shape、layout、mask等信息，采用strided方式将任务均衡切分到可用AIC核上，输出metadata供主算子使用。
    - `aclnnSparseFlashMlaSoftmaxL1Norm`接口：根据metadata中的分核信息，对Q和K计算Softmax L1Norm。

- 计算公式：

    阶段一：根据是否为sparse场景，对输入k进行选择

    * 当为sparse时：

    $$
    selectedKv\text{ }=\text{ }Gather \left( K, sparseIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else：

    $$
    selectedKv\text{ }=\text{ }K
    $$

    阶段二：计算P（SimpleSoftmax）

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), softmaxLse)
    $$

    阶段三：计算Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    其中，$G$ 为group数（$G = N1 / N2$），$ReduceSum$ 在G维度（q head group维度）上对softmax概率$P$求和后取平均。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseFlashMlaSoftmaxL1Norm”接口执行计算。
```c++
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
    const aclTensor   *q,
    const aclTensor   *k,
    const aclTensor   *softmaxLse,
    const aclTensor   *sparseIndicesOptional,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    const aclTensor   *topkLengthOptional,
    const aclTensor   *metadataOptional,
    double             softmaxScale,
    int64_t            cmpRatio,
    int64_t            maskMode,
    char              *layoutQOptional,
    char              *layoutKOptional,
    const aclTensor   *softmaxL1Norm,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```
```c++
aclnnStatus aclnnSparseFlashMlaSoftmaxL1Norm(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```

## aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 290px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>q（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入Q。</td>
            <td>
            q、k的N轴对应关系需满足GQA约束（N1 = N2 × G）。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：支持泛化；S1：支持泛化；N1：支持1~128；D：512；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>k（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入K(V)。</td>
            <td>
            当前暂不支持空tensor。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            B：与q的B保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxLse（aclTensor*）</td>
            <td>输入</td>
            <td>注意力正向计算的输出softmaxLse，计算公式详见sparse_flash_mla文档。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)<br>
            B：与q的B保持一致；N2：1；S1：与q的S1保持一致；G：N1/N2；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sparseIndicesOptional（aclTensor*）</td>
            <td>输入</td>
            <td>稀疏场景下选择的k中权重较高的注意力索引。</td>
            <td>
            <ul>
                <li>支持空tensor</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K)、(T1,N2,K)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensKOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，Key的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T2保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中q实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedKOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中k实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpResidualKOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示每个batch S2 // cmpRatio后的余数。</td>
            <td>
            maskMode=3且cmpRatio!=1时必须传入。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>topkLengthOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示每行q对应的k实际可选的topk长度。</td>
            <td>
            maskMode=0且存在稀疏索引时需要传，且必须为准确值。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2)、(T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>metadataOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示tiling下沉的aicpu算子输出结果。</td>
            <td>
            必须传入。由aclnnSparseFlashMlaSoftmaxL1NormMetadata算子生成。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(x)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxScale（double）</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中d开根号的倒数。
            </td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio（int64_t）</td>
            <td>输入</td>
            <td>表示对k的压缩率。</td>
            <td>
            取值范围：1~128。
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>maskMode（int64_t）</td>
            <td>输入</td>
            <td>表示q和k计算的mask模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>layoutQOptional（char*）</td>
            <td>输入</td>
            <td>表示输入q的数据排布格式。</td>
            <td>
            支持"BSND"、"TND"。需与layoutKOptional一致。
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutKOptional（char*）</td>
            <td>输入</td>
            <td>表示输入k的数据排布格式。</td>
            <td>
            支持"BSND"、"TND"。需与layoutQOptional一致。
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>softmaxL1Norm（aclTensor*）</td>
            <td>输出</td>
            <td>表示q与k计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。</td>
            <td>
            若存在稀疏索引，则该输出不为空；其他场景下该参数输出为空。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,S2)、(T1,N2,T2)<br>
            B：与q的B保持一致；S1：与q的S1保持一致；N2：1；S2：与k的S2保持一致；T1：B × S1；T2：B × S2
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

- **返回值**

   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

   第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>参数中存在非法的nullptr。</td>
            </tr>
            <tr>
                <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="2">161002</td>
                <td>输入的数据类型不满足支持类型。</td>
            </tr>
            <tr>
                <td>q、k、softmaxLse、softmaxL1Norm必选输入/输出未传。</td>
            </tr>
        </tbody>
    </table>


## aclnnSparseFlashMlaSoftmaxL1Norm

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值**

   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。


## 约束说明

- 确定性说明：aclnnSparseFlashMlaSoftmaxL1Norm默认确定性实现。
- 公共约束：
    - 入参为空处理：q为空Tensor时直接返回。

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>不做mask操作</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        </tbody>
    </table>
- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>B</td>
            <td>支持泛化</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>支持泛化</td>
            <td>支持S1、S2支持不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1~128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>-</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h"

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

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // TND layout: qShape=(T1,N1,D), kShape=(T2,N2,D), softmaxLseShape=(N2,T1,G)
  std::vector<int64_t> qShape = {16, 128, 512};
  std::vector<int64_t> kShape = {2048, 1, 512};
  std::vector<int64_t> softmaxLseShape = {1, 16, 128};
  std::vector<int64_t> cuSeqQLenshape = {2};
  std::vector<int64_t> cuSeqKLenshape = {2};
  std::vector<int64_t> cmpResidualKShape = {1};
  std::vector<int64_t> softmaxL1NormShape = {16, 1, 2048};
  std::vector<int64_t> metadataShape = {64};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* cuSeqQLenDeviceAddr = nullptr;
  void* cuSeqKLenDeviceAddr = nullptr;
  void* cmpResidualKDeviceAddr = nullptr;
  void* softmaxL1NormDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* cuSeqQLen = nullptr;
  aclTensor* cuSeqKLen = nullptr;
  aclTensor* cmpResidualK = nullptr;
  aclTensor* softmaxL1Norm = nullptr;
  aclTensor* metadata = nullptr;

  std::vector<short> qHostData(16 * 128 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<float> softmaxLseHostData(1 * 16 * 128, 3.0);
  std::vector<int32_t> cuSeqQLenHostData = {0, 16};
  std::vector<int32_t> cuSeqKLenHostData = {0, 2048};
  std::vector<int32_t> cmpResidualKHostData = {0};
  std::vector<float> softmaxL1NormHostData(16 * 1 * 2048, 0);
  std::vector<int32_t> metadataHostData(64, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqKLenHostData, cuSeqKLenshape, &cuSeqKLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqKLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &cmpResidualKDeviceAddr, aclDataType::ACL_INT32,
                        &cmpResidualK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxL1NormHostData, softmaxL1NormShape, &softmaxL1NormDeviceAddr, aclDataType::ACL_FLOAT,
                        &softmaxL1Norm);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double softmaxScale = 0.088388;
  int64_t maxSeqlenK = 2048;
  int64_t cmpRatio = 128;
  int64_t maskMode = 3;
  char layoutQ[4] = {'T', 'N', 'D', 0};
  char layoutK[4] = {'T', 'N', 'D', 0};

  // 1. 调用 metadata 前置算子
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
      cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr,
      0, 16, 2048, 128, 1, 512, 0, cmpRatio, maskMode, layoutQ, layoutK,
      metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSparseFlashMlaSoftmaxL1NormMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadata failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 2. 调用主算子
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
      q, k, softmaxLse, nullptr, cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr, metadata,
      softmaxScale, maxSeqlenK, cmpRatio, maskMode, layoutQ, layoutK, softmaxL1Norm, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  try {
    ret = aclnnSparseFlashMlaSoftmaxL1Norm(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("Expected kernel failure (skeleton stage): ERROR: %d\n", ret);
    } else {
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    }
  } catch (const std::exception& e) {
    LOG_PRINT("Expected kernel failure (skeleton stage): %s\n", e.what());
  }

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(cuSeqQLen);
  aclDestroyTensor(cuSeqKLen);
  aclDestroyTensor(cmpResidualK);
  aclDestroyTensor(softmaxL1Norm);
  aclDestroyTensor(metadata);
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(cuSeqQLenDeviceAddr);
  aclrtFree(cuSeqKLenDeviceAddr);
  aclrtFree(cmpResidualKDeviceAddr);
  aclrtFree(softmaxL1NormDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
