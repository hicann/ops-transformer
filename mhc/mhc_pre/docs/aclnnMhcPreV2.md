# aclnnMhcPreV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_pre)

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

- 接口功能：基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。

- 相较于[aclnnMhcPre](aclnnMhcPre.md)接口，此接口新增`opImplMode`参数，用于指定MhcPre算子的计算模式：
  - `opImplMode=0`：在Cube中使用FP32模式计算。
  - `opImplMode=1`：在Cube中使用HF32模式计算。

- 计算公式：

  其中，xFlat表示将x的最后两维n和D视作长度为nD的向量，gammaFlat表示将gamma视作长度为nD的向量，@表示矩阵乘法，$\odot$表示逐元素乘法。

  $$
  \begin{aligned}
  invRms &= \left(mean(xFlat^{2}) + normEps\right)^{-\frac{1}{2}}\\
  xGamma &= \begin{cases}
      xFlat \odot gammaFlat, & gamma \ne null \\
      xFlat, & gamma = null
  \end{cases}\\
  hMix &= xGamma @ phi^{T}\\
  w &= hMix \odot invRms\\
  [pPre, pPost, pRes] &= \begin{cases}
      split(w, [n, n, n^{2}]), & alpha.shape=[3] \\
      [split(w, [n, n]), 0], & alpha.shape=[2]
  \end{cases}\\
  hPre &= \sigma(pPre \odot alpha0 + bias0) + hcEps\\
  hPost &= \begin{cases}
      2\sigma(pPost \odot alpha1 + bias1), & alpha.shape=[3] \\
      2\sigma(pPost \odot alpha1 + bias1) + hcEps, & alpha.shape=[2]
  \end{cases}\\
  hRes &= \begin{cases}
      pRes \odot alpha2 + bias2, & alpha.shape=[3] \\
      0, & alpha.shape=[2]
  \end{cases}\\
  hIn_{d} &= \sum_{i=0}^{n-1} hPre_{i} x_{i,d}
  \end{aligned}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMhcPreV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcPreV2"接口执行计算。

```Cpp
aclnnStatus aclnnMhcPreV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *gammaOptional,
    double           normEps,
    double           hcEps,
    int64_t          opImplMode,
    aclTensor       *hIn,
    aclTensor       *hPost,
    aclTensor       *hRes,
    aclTensor       *invRmsOptional,
    aclTensor       *hMixOptional,
    aclTensor       *hPreOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMhcPreV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcPreV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
      <td>x</td>
      <td>输入</td>
      <td>待计算数据，表示网络中mHC层的输入数据，对应公式中的x。</td>
      <td>不能为空Tensor。</td>
      <td>BFLOAT16或FLOAT16</td>
      <td>ND</td>
      <td>[B,S,n,D] 或 [T,n,D]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>phi</td>
      <td>输入</td>
      <td>mHC的参数矩阵，对应公式中的phi。</td>
      <td>不能为空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n^2+2n, nD]或[2n, nD]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>mHC的缩放参数，对应公式中的alpha。</td>
      <td>不能为空Tensor。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>[3]或[2]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>bias</td>
          <td>输入</td>
      <td>mHC的bias参数，对应公式中的bias。</td>
      <td>不能为空Tensor。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>[n^2+2n]或[2n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>gammaOptional</td>
      <td>可选输入</td>
      <td>表示进行RmsNorm计算的缩放因子，对应公式中的gamma。</td>
      <td>对于推理场景，可传空指针。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[n, D]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>normEps</td>
      <td>可选输入</td>
      <td>RmsNorm的防除零参数，对应公式中的normEps。</td>
      <td>建议值：1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hcEps</td>
      <td>可选输入</td>
      <td>h_pre的sigmoid后的eps参数，以及当alpha.shape=[2]时h_post的sigmoid后的eps参数，对应公式中的hcEps。</td>
      <td>建议值：1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>opImplMode</td>
      <td>输入</td>
      <td>指定MhcPre算子的计算模式。</td>
      <td>支持的枚举值如下：<ul>
        <li>0：建议值，在Cube中使用FP32模式计算。</li>
        <li>1：在Cube中使用HF32模式计算。</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hIn</td>
      <td>输出</td>
      <td>输出的h_in作为Attention/MLP层的输入，对应公式中的hIn。</td>
      <td>不能为空Tensor。</td>
      <td>BFLOAT16或FLOAT16</td>
      <td>ND</td>
      <td>[B,S,D] 或 [T,D]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hPost</td>
      <td>输出</td>
      <td>输出的mHC的h_post变换矩阵，对应公式中的hPost。</td>
      <td>不能为空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n] 或 [T,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hRes</td>
      <td>输出</td>
      <td>输出的mHC的h_res变换矩阵（未做sinkhorn变换），对应公式中的hRes。</td>
      <td>当alpha=[3]时不能为空Tensor；当alpha=[2]时可传空指针。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n,n] 或 [T,n,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>invRmsOptional</td>
      <td>可选输出</td>
      <td>RmsNorm计算得到的1/r，对应公式中的invRms。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S] 或 [T]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hMixOptional</td>
      <td>可选输出</td>
      <td>xGamma与phi矩阵乘的结果，对应公式中的hMix。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n^2+2n] 或 [T,n^2+2n] 或 [B,S,2n] 或 [T, 2n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hPreOptional</td>
      <td>可选输出</td>
      <td>做完sigmoid计算之后的h_pre矩阵，对应公式中的hPre。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,n] 或 [T,n]</td>
      <td>-</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>x、phi、alpha、bias、hIn、hPost、hRes存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="3"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="3"> 161002 </td>
    <td>x、phi、alpha、bias、gammaOptional、hIn、hPost、hRes、invRmsOptional、hMixOptional、hPreOptional的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
    <td>x、phi、alpha、bias、gammaOptional、hIn、hPost、hRes、invRmsOptional、hMixOptional、hPreOptional的shape维度不在支持的范围内。</td>
    </tr>
    <tr>
    <td>opImplMode不为0或1。</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_RUNTIME_ERROR </td>
    <td> 361001 </td>
    <td>API内部调用npu runtime的接口异常。</td>
    </tr>
  </tbody></table>

## aclnnMhcPreV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreV2GetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMhcPreV2默认确定性实现。

- Batch一致性说明：
  - 默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。

- 规格约束：
  - n目前支持4、6、8。
  - D支持1~16384范围以内，需满足D为16对齐。
  - 当alpha=[3]时，支持hRes输出，必须满足以下条件：输入phi=[n^2+2n, nD],bias=[n^2+2n]，输出hMixOptional=[B,S,n^2+2n] 或 [T,n^2+2n]；当alpha=[2]时，hRes输出为0，必须满足以下条件：输入phi=[2n, nD],bias=[2n]，输出hMixOptional=[B,S,2n] 或 [T,2n]。
  - 可选输出 invRmsOptional、hMixOptional、hPreOptional 为互存关系，需同时输出或全部不输出，不支持仅返回其中部分，输出条件：invRmsOptional!=nullptr && hMixOptional!=nullptr && hPreOptional!=nullptr。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_v2.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float类型）
void PrintTensorDataFloat(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float16类型）
void PrintTensorDataFloat16(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_fp16(size);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_fp16.data(), size * sizeof(aclFloat16), device_addr, size * sizeof(aclFloat16),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    for (int64_t i = 0; i < size; ++i) {
        host_data[i] = aclFloat16ToFloat(host_fp16[i]);
    }

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

// 创建FLOAT32类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat32(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT16类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<aclFloat16> host_data_fp16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_fp16[i] = aclFloat16(host_data[i]);
    }

    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_fp16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT16类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(aclFloat16);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建FLOAT32类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat32Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr, *gamma_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr, *inv_rms_addr = nullptr;
    void *h_mix_addr = nullptr, *h_pre_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr, *inv_rms = nullptr;
    aclTensor *h_mix = nullptr, *h_pre = nullptr;
};

int CreateInputTensors(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &phi_shape,
                       const std::vector<int64_t> &alpha_shape, const std::vector<int64_t> &bias_shape,
                       const std::vector<int64_t> &gamma_shape, Tensors &tensors)
{
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(3, 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);

    int ret = CreateAclTensorFloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(gamma_host_data, gamma_shape, tensors.gamma_addr, tensors.gamma);
    CHECK_RET(ret == 0, LOG_PRINT("Create gamma_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &hin_shape, const std::vector<int64_t> &h_post_shape,
                        const std::vector<int64_t> &h_res_shape, const std::vector<int64_t> &inv_rms_shape,
                        const std::vector<int64_t> &h_mix_shape, const std::vector<int64_t> &h_pre_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorFloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_mix_shape, tensors.h_mix_addr, tensors.h_mix);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_mix_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.gamma);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.h_mix);
    aclDestroyTensor(tensors.h_pre);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.gamma_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.h_mix_addr);
    aclrtFree(tensors.h_pre_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;
    int B = 1, S = 2048, n = 4, D = 2560;
    std::vector<int64_t> x_shape = {B * S, n, D}, phi_shape = {n * n + 2 * n, n * D}, alpha_shape = {3},
                         bias_shape = {n * n + 2 * n}, gamma_shape = {n, D};
    std::vector<int64_t> hin_shape = {B * S, D}, h_post_shape = {B * S, n}, h_res_shape = {B * S, n, n},
                         inv_rms_shape = {B * S}, h_mix_shape = {B * S, n * n + 2 * n}, h_pre_shape = {B * S, n};
    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);
    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, gamma_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, inv_rms_shape, h_mix_shape, h_pre_shape, tensors);
    CHECK_RET(ret == 0, return -1);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    int64_t opImplMode = 1;
    aclnnStatus aclnn_ret = aclnnMhcPreV2GetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.gamma, 1e-6, 1e-6, opImplMode, tensors.hin,
        tensors.h_post, tensors.h_res, tensors.inv_rms, tensors.h_mix, tensors.h_pre, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);
    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }
    aclnn_ret = aclnnMhcPreV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreV2 failed, error: %d\n", aclnn_ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);
    LOG_PRINT("MhcPre compute success!\nOutput tensor data: \n");
    PrintTensorDataFloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(inv_rms_shape, tensors.inv_rms_addr);
    PrintTensorDataFloat(h_mix_shape, tensors.h_mix_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);
    DestroyTensors(tensors);
    FreeDeviceMemory(tensors);
    if (workspace_size > 0)
        aclrtFree(workspace_addr);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
```
