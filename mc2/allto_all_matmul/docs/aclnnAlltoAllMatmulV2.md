# aclnnAlltoAllMatmulV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/allto_all_matmul)

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

- 接口功能：完成AlltoAll通信、Permute（保证通信后地址连续）和Matmul计算的融合，**先通信后计算**。

- 计算公式：假设x1输入shape为(BS, H)，rankSize为NPU卡数

  $$
  commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
  permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
  output = permutedOut @ x2 + bias \\
  $$

相较于`aclnnAlltoAllMatmul`接口，该接口变更如下：

- 新增`commMode`参数，用户根据该参数指定芯片使用的通信引擎。

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：`commMode`支持`ai_cpu`和`ccu`。
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：无新增特性，`commMode`仅支持`ai_cpu`。
  <!-- end id8 -->
  <!-- npu="910b" id9 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：无新增特性，`commMode`仅支持`aiv`。

  <!-- end id9 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAlltoAllMatmulV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAlltoAllMatmulV2”接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllMatmulV2GetWorkspaceSize(
  const aclTensor*   x1,
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char*        group,
  const char*        commMode,
  bool               transposeX1,
  bool               transposeX2,
  const aclTensor*   output,
  const aclTensor*   alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllMatmulV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllMatmulV2GetWorkspaceSize

- ​**参数说明**​

    <table style="undefined;table-layout: fixed; width: 1556px"> <colgroup>
    <col style="width: 154px">
    <col style="width: 123px">
    <col style="width: 270px">
    <col style="width: 325px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 116px">
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
    <th>非连续tensor</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>输入</td>
    <td>融合算子的左矩阵输入，对应公式中的x1。</td>
    <td>该输入进行AlltoAll通信与Permute操作后结果作为MatMul计算的左矩阵输入。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵。</td>
    <td>直接作为MatMul计算的右矩阵输入。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(H*rankSize, N)</td>
    <td>不同设备型号支持情况不同，参见<a href="#约束说明">约束说明</a>。</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>输入</td>
    <td>矩阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>支持传入空指针场景，根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>1维，shape为(N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>输入</td>
    <td>AlltoAll和Permute数据交换的方向。</td>
    <td>支持配置空或者[-2,-1]，传入空时默认按[-2,-1]处理，表示将输入由(BS, H)转为(BS/rankSize, rankSize*H)。</td>
    <td>aclIntArray*(元素类型INT64)</td>
    <td>-</td>
    <td>1维，shape为(2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
    <td>字符串长度要求(0, 128)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commMode</td>
    <td>输入</td>
    <td>指定当前通信类型。</td>
    <td>支持输入"aiv"、"ai_cpu"或"ccu"。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>输入</td>
    <td>标识左矩阵是否转置过。</td>
    <td>暂不支持配为True。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>输入</td>
    <td>标识右矩阵是否转置过。</td>
    <td>配置为True时右矩阵Shape为(N, rankSize*H)。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>输出</td>
    <td>最终的计算结果。</td>
    <td>数据类型与输入x1保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>输出</td>
    <td>接收AlltoAll和Permute后的内容。</td>
    <td>传入nullptr时表示不输出通信输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, H*rankSize)</td>
    <td>x</td>
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
    <td>返回op执行器，包含了算子的计算流程。</td>
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

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:282px">
    <col style="width:120px">
    <col style="width:747px">
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
      <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
        <td>输入Tensor为空Tensor。</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional非法。</td>
    </tr>
    <tr>
        <td>transposeX1为true。</td>
    </tr>
    <tr>
        <td>通信域长度非法。</td>
    </tr>
    <tr>
        <td>输入输出Tensor维度不合法。</td>
    </tr>
    <tr>
        <td>输入输出format为私有格式。</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllMatmulV2

- **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllMatmulV2GetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

* aclnnAlltoAllMatmulV2默认支持确定性计算。
* NPU卡数（rankSize），根据设备型号有不同限制：

  <!-- npu="910b" id10 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡。
  <!-- end id10 -->
  <!-- npu="A3" id11 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：支持2、4、8、16卡。
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16卡。
  <!-- end id12 -->

* 参数说明中shape使用的变量BS必须整除NPU卡数。
* BS和N的值不得超过2147483647（INT32_MAX），BS的值不得小于0，N的值不得小于1。
* H*rankSize范围，根据设备型号有不同限制：

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持[1, 35000]。
  <!-- end id13 -->
  <!-- npu="950,A3" id14 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  、<term>Ascend 950PR/Ascend 950DT</term>：支持[2, 65535]。
  <!-- end id14 -->

* 空tensor的支持度根据不同设备型号有不同的限制：

  <!-- npu="910b" id15 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持任何空tensor。
  <!-- end id15 -->
  <!-- npu="950,A3" id16 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  、<term>Ascend 950PR/Ascend 950DT</term>：仅支持输入x1的第一维度（BS）为0的空tensor，其它空tensor均不支持。
  <!-- end id16 -->

* 非连续tensor的支持度根据不同设备型号有不同的限制：

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：不支持任何非连续tensor。
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持x2为非连续tensor，其它非连续tensor均不支持。
  <!-- end id18 -->

* x1、x2计算输入的数据类型要和output、alltoAllOutOptional计算输出的数据类型一致，传入的x1、x2与output均不为空指针。
* biasOptional的数据类型根据不同设备型号有不同的限制：

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：x1/x2计算输入的数据类型为FLOAT16时，biasOptional计算输入的数据类型支持FLOAT16；x1/x2计算输入的数据类型为BFLOAT16时，biasOptional计算输入的数据类型支持FLOAT32。
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：x1/x2计算输入的数据类型为FLOAT16时，biasOptional计算输入的数据类型支持FLOAT16和FLOAT32；x1/x2计算输入的数据类型为BFLOAT16时，biasOptional计算输入的数据类型支持BFLOAT16和FLOAT32。
  <!-- end id20 -->

* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。
* 通信引擎参数commMode根据不同设备型号有不同的限制：

   <!-- npu="910b" id21 -->
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持输入`aiv`。
   <!-- end id21 -->
   <!-- npu="A3" id22 -->
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持输入`ai_cpu`。
   <!-- end id22 -->
   <!-- npu="950" id23 -->
   - <term>Ascend 950PR/Ascend 950DT</term>：支持输入`ai_cpu`和`ccu`。
   <!-- end id23 -->

* 通信引擎约束：

   <!-- npu="910b" id24 -->
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持MTE通信。
   <!-- end id24 -->
   <!-- npu="A3" id25 -->
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：支持AI_CPU通信。
   <!-- end id25 -->
   <!-- npu="950" id26 -->
   - <term>Ascend 950PR/Ascend 950DT</term>：
      - 支持CCU通信和AI_CPU通信。
      - CCU通信仅支持单机UB域内互联，AI_CPU可支持跨机UB域内互联。
      - 通信域约束：同一个通信域内只能使用同一种通信方式。

   <!-- end id26 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy,请参考[《HCCL API (C)》](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

<!-- npu="910b" id27 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_matmul_v2.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadAlltoAllMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> alltoalloutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *alltoalloutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *alltoallout = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long alltoalloutShapeSize = GetShapeSize(alltoalloutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<op::fp16_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> alltoalloutHostData(alltoalloutShapeSize, 0);
        // 创建tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(alltoalloutHostData, alltoalloutShape, &alltoalloutDeviceAddr, aclDataType::ACL_FLOAT16, &alltoallout);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnAlltoAllMatmulV2GetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, "aiv", false, false,
                                                out, alltoallout, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnAlltoAllMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllMatmulV2 failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllMatmulV2 execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (alltoallout != nullptr) {
            aclDestroyTensor(alltoallout);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (alltoalloutDeviceAddr != nullptr) {
            aclrtFree(alltoalloutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // 本样例基于Atlas A2实现，必须在Atlas A2上运行
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id27 -->
<!-- npu="A3" id28 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_matmul_v2.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadAlltoAllMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> alltoalloutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *alltoalloutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *alltoallout = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long alltoalloutShapeSize = GetShapeSize(alltoalloutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<op::fp16_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> alltoalloutHostData(alltoalloutShapeSize, 0);
        // 创建tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(alltoalloutHostData, alltoalloutShape, &alltoalloutDeviceAddr, aclDataType::ACL_FLOAT16, &alltoallout);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnAlltoAllMatmulV2GetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, "ai_cpu", false, false,
                                                out, alltoallout, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnAlltoAllMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllMatmulV2 failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllMatmulV2 execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (alltoallout != nullptr) {
            aclDestroyTensor(alltoallout);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (alltoalloutDeviceAddr != nullptr) {
            aclrtFree(alltoalloutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // 本样例基于Atlas A2实现，必须在Atlas A2上运行
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id28 -->
<!-- npu="950" id29 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_allto_all_matmul_v2.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadAlltoAllMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> alltoalloutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *alltoalloutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *alltoallout = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long alltoalloutShapeSize = GetShapeSize(alltoalloutShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> alltoalloutHostData(alltoalloutShapeSize, 0);
        // 创建tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(alltoalloutHostData, alltoalloutShape, &alltoalloutDeviceAddr, aclDataType::ACL_FLOAT16, &alltoallout);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnAlltoAllMatmulV2GetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, "ai_cpu", false, false,
                                                out, alltoallout, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnAlltoAllMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllMatmulV2 failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllMatmulV2 execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (alltoallout != nullptr) {
            aclDestroyTensor(alltoallout);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (alltoalloutDeviceAddr != nullptr) {
            aclrtFree(alltoalloutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // 本样例基于<term>Ascend 950PR/Ascend 950DT</term>实现，必须在<term>Ascend 950PR/Ascend 950DT</term>上运行
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id29 -->
