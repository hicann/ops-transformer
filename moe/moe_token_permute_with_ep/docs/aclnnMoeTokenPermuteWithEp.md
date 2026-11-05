# aclnnMoeTokenPermuteWithEp

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。
- **计算公式**：
  - paddedMode为`false`时，公式如下。topK表示每个token选择的专家数量。如果indices为2维，则topK等于indices最后一维的大小。如果indices为1维，则topK为1。下文参数表与约束说明中的topK_num即topK。

    $$
    sortedIndicesFirst=argSort(\text{flatten}(indices))
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    当rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]时:

    $$
    permuteTokensOut[sortedIndicesOut[i] - rangeOptional[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndicesOut[i] - rangeOptional[0]]=probsOptional[i]
    $$

  - paddedMode为`true`时（暂不支持）：

    $$
    permuteTokensOut[i]=tokens[indices[i]]
    $$

    $$
    sortedIndicesOut=indices
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMoeTokenPermuteWithEpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteWithEp”接口执行计算。

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(
    const aclTensor    *tokens,
    const aclTensor    *indices,
    const aclTensor    *probsOptional,
    const aclIntArray  *rangeOptional,
    int64_t             numOutTokens,
    bool                paddedMode,
    const aclTensor    *permuteTokensOut,
    const aclTensor    *sortedIndicesOut,
    const aclTensor    *permuteProbsOut,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEp(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMoeTokenPermuteWithEpGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 160px">
    <col style="width: 100px">
    <col style="width: 280px">
    <col style="width: 520px">
    <col style="width: 200px">
    <col style="width: 90px">
    <col style="width: 120px">
    <col style="width: 80px">
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
      <td>tokens（aclTensor*）</td>
      <td>输入</td>
      <td>表示permute中的输入tokens。</td>
      <td><ul><li>不支持空tensor。</li><li>要求为一个2D的Tensor，shape为(num_tokens, hidden_size)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入tokens对应的专家索引。</td>
      <td><ul><li>不支持空tensor。</li><li>要求shape为1D或2D。</li><li>paddedMode为false时表示每一个输入token对应的topK个处理专家索引，shape为(num_tokens, topK_num)或(num_tokens)。</li><li>paddedMode为true时表示每个专家选中的token索引（暂不支持）。</li><li>topK_num（二维时取最后一维大小，一维时为1）取值范围为[1, 512]。</li><li>元素个数（num_tokens * topK_num）取值小于16777215，即不超过16777214。</li><li>元素值取值范围为[0, 16777214]（大于等于0且小于16777215）。除该值域外，仅单点支持本数据类型（INT32或INT64）的最大值或最小值，用作无效位置标记；其余越界值的排序结果不正确。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1或2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示输入tokens对应的专家概率。</td>
      <td><ul><li>可选输入，不传入时不输出permuteProbsOut，此时permuteProbsOut不参与计算与结果比对，但仍须传入合法的aclTensor（不得传空指针）。</li><li>传入时shape与indices的shape相同，数据类型需与tokens相同，不得为空tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1或2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rangeOptional（aclIntArray*）</td>
      <td>可选输入</td>
      <td>ep切分的有效范围。</td>
      <td><ul><li>支持空。</li><li>size为2，元素类型为INT64。</li><li>非空时两个元素start、end均要求在[0, num_tokens*topK]范围内，否则报错。</li><li>为空时，忽略probsOptional和permuteProbsOut，执行逻辑回退到aclnnMoeTokenPermute。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numOutTokens（int64_t）</td>
      <td>输入</td>
      <td>输出token数量相关参数。</td>
      <td><ul><li>仅在rangeOptional为空时生效，此时切片起点为0、终点为该参数值。</li><li>取值为0时表示不删除任何token，输出num_tokens*topK行；取值为正整数时保留排序结果的前numOutTokens行。</li><li>取值范围为[0, num_tokens*topK]，不支持负数，也不支持大于num_tokens*topK的取值，越界返回失败。</li><li>rangeOptional非空时该参数被忽略，输出token数量由rangeOptional决定。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode（bool）</td>
      <td>输入</td>
      <td>表示是否为填充模式。</td>
      <td>取值为false和true。<ul><li>false：表示非填充模式，对indices进行排序。</li><li>true：表示填充模式，indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序（暂不支持）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut（aclTensor*）</td>
      <td>输出</td>
      <td>根据indices进行扩展并排序过的tokens。</td>
      <td><ul><li>要求是一个2D的Tensor：rangeOptional非空时shape为(rangeOptional[1] - rangeOptional[0], hidden_size)；rangeOptional为空时shape为(numOutTokens, hidden_size)，numOutTokens取0时第一维为num_tokens*topK_num。</li><li>数据类型同tokens。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sortedIndicesOut（aclTensor*）</td>
      <td>输出</td>
      <td>permuteTokensOut和tokens的映射关系。</td>
      <td><ul><li>要求是一个1D的Tensor，Shape为(num_tokens * topK_num)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>permuteProbsOut（aclTensor*）</td>
      <td>输出</td>
      <td>根据indices进行扩展并排序过的probs。</td>
      <td><ul><li>要求是一个1D的Tensor：rangeOptional非空时shape为(rangeOptional[1] - rangeOptional[0])，rangeOptional为空时shape为(numOutTokens,)，numOutTokens取0时为(num_tokens*topK_num,)。</li><li>数据类型同tokens；probsOptional不传入时该输出不参与计算与比对。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
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
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 120px">
      <col style="width: 630px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>输入和输出的Tensor是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入和输出的数据类型或数据格式不在支持的范围内。</td>
      </tr>
      <tr>
        <td rowspan="11">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="11">561002</td>
        <td>tokens的shape维度不为2。</td>
      </tr>
      <tr>
        <td>tokens或indices的shape第0维为0（不支持空tensor输入）。</td>
      </tr>
      <tr>
        <td>tokens的hidden_size（第1维）不大于0，或hidden_size过大导致整数溢出。</td>
      </tr>
      <tr>
        <td>indices的shape不为1D或2D，或者paddedMode为false时indices的shape第一维与tokens的第一维不相等。</td>
      </tr>
      <tr>
        <td>indices的topK（indices最后一维的大小）不在[1, 512]范围内。</td>
      </tr>
      <tr>
        <td>indices的元素个数（num_tokens * topK）大于等于16777215。</td>
      </tr>
      <tr>
        <td>rangeOptional非空时其size不为2。</td>
      </tr>
      <tr>
        <td>rangeOptional（或rangeOptional为空时的numOutTokens）不满足start、end均在[0, num_tokens*topK]范围内。</td>
      </tr>
      <tr>
        <td>传入probsOptional时，其shape维度与indices不一致，或第一维与tokens的第一维不一致。</td>
      </tr>
      <tr>
        <td>输出permuteTokensOut、sortedIndicesOut、permuteProbsOut的shape与推导结果不一致。</td>
      </tr>
      <tr>
        <td>paddedMode为true（暂不支持）。</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeTokenPermuteWithEp

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 170px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithEpGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- tokens要求为2D，shape为(num_tokens, hidden_size)，且第0维num_tokens与indices的第0维相同；indices要求为1D或2D。
- 支持边界：
  - indices元素个数（num_tokens * topK_num）小于16777215（不超过16777214），元素值在[0, 16777214]范围内，单个int32/int64最大值或最小值可作无效位置标记。
  - topK_num取值范围为[1, 512]。
  - rangeOptional非空时size为2，两个元素start、end均在[0, num_tokens*topK_num]内；rangeOptional为空时numOutTokens取值范围为[0, num_tokens*topK_num]，0表示不删除任何token。
  - 输入tokens、indices不支持空tensor：num_tokens与hidden_size均需大于0，num_tokens为0的调用不在支持范围内；probsOptional为可选输入，可不传入，传入时其shape与indices相同且不得为空tensor。
  - probsOptional不传入时，permuteProbsOut不参与计算与比对，但仍须传入合法的aclTensor（不得传空指针），建议其shape与参数表中permuteProbsOut的推导shape保持一致。
  - paddedMode仅支持false。
- 确定性计算：
  - aclnnMoeTokenPermuteWithEp默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = hostData.size() * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
    // 1. 固定写法，device/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> probsShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {4, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expandedProbsOutShape = {4};

    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    void* expandedProbsOutDeviceAddr = nullptr;
    void* rangeDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* probs = nullptr;
    aclIntArray* range = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    aclTensor* expandedProbsOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 3, 1, 2, 3};
    std::vector<float> probsHostData = {0.5, 0.3, 0.4, 0.2, 0.5, 0.4};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<float> expandedProbsOutHostData = {0, 0, 0, 0};
    std::vector<int64_t> rangeHostData = {1, 5};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedProbsOutHostData, expandedProbsOutShape, &expandedProbsOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedProbsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建相关attr
    ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenPermuteWithEp第一段接口
    ret = aclnnMoeTokenPermuteWithEpGetWorkspaceSize(x, indices, probs, range, numTokenOut, padMode, expandedXOut, sortedIndicesOut, expandedProbsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMoeTokenPermuteWithEp第二段接口
    ret = aclnnMoeTokenPermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEp failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    auto expandedProbsSize = GetShapeSize(expandedProbsOutShape);
    std::vector<float> expandedProbsData(expandedProbsSize, 0);
    ret = aclrtMemcpy(expandedProbsData.data(), expandedProbsData.size() * sizeof(expandedProbsData[0]), expandedProbsOutDeviceAddr, expandedProbsSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedProbsSize; i++) {
        LOG_PRINT("expandedProbsData[%ld] is: %f\n", i, expandedProbsData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(probs);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);
    aclDestroyTensor(expandedProbsOut);
    aclDestroyIntArray(range);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    aclrtFree(expandedProbsOutDeviceAddr);
    aclrtFree(rangeDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
