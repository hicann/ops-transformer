# aclnnGroupedMatMulAllReduce

**This API will be deprecated in later versions. Do not use this API.**

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Implements the multi-rank parallel AllReduce function based on grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The following four scenarios are supported based on the tensor count of $x$, $weight$, and $y$:
  - The number of tensors for `x`, `weight`, and `y` equals the group count. That is, the tensors of each group are independent.
  - `x` has one tensor, while `weight` and `y` have tensors equal to the group count. In this case, use the optional parameter `group_list` to define the row-wise grouping of `x`. For example, `group_list[0]=10` indicates that the first 10 rows of `x` participate in the multiplication of the first group of matrices.
  - The number of tensors for `x` and `weight` equals the group count, but `y` has only one tensor. In this case, products of each matrix group multiplication are stored contiguously within a single tensor.
  - `x` and `y` each have one tensor, while `weight` has tensors equal to the group count. This is a hybrid configuration combining the preceding two cases.

- Formula:

  Non-quantization scenario:

  $$
  y_i=x_i\times weight_i + bias_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatMulAllReduceGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatMulAllReduce` is called to perform computation.

```cpp
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x,
    const aclTensorList* weight,
    const aclTensorList* bias,
    const aclIntArray* groupListOptional,
    int64_t splitItem,
    const char* group,
    const char* reduceOp,
    int64_t commTurn,
    int64_t streamMode,
    const aclTensorList* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnGroupedMatMulAllReduce(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnGroupedMatMulAllReduceGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1600px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 140px">
    <col style="width: 320px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 180px">
    </colgroup>
    <thead>
      <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>x(aclTensorList*)</td>
      <td>Input</td>
      <td>x in the formula.</td>
      <td>The number of tensors in the list cannot exceed 64. When splitItem is 0, 2 to 6 dimensions are supported. When splitItem is 1, 2, or 3, only 2 dimensions are supported.</td>
      <td>FLOAT16, BFLOAT16 (data type of tensors in the list)</td>
      <td>ND</td>
      <td>2-6</td>
      <td>×</td>
      </tr>
      <tr>
      <td>weight(aclTensorList*)</td>
      <td>Input</td>
      <td>weight in the formula.</td>
      <td>The number of tensors in the list cannot exceed 64. Only 2 dimensions are supported.</td>
      <td>FLOAT16, BFLOAT16 (data type of tensors in the list)</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
      </tr>
      <tr>
      <td>bias(aclTensorList*)</td>
      <td>Optional input</td>
      <td>Bias in the formula.</td>
      <td>The number of tensors in the list cannot exceed 64. Only 1 dimension is supported. Null pointers can be passed.</td>
      <td>FLOAT16, FLOAT32 (data type of tensors in the list)</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
      </tr>
      <tr>
      <td>groupListOptional(aclIntArray*)</td>
      <td>Optional input</td>
      <td>aclIntArray type on the host, indicating the size distribution of the input and output matmul in the M direction.</td>
      <td>The array length cannot exceed 64. This parameter is mandatory when splitItem is 1 or 3, and must be left empty when splitItem is 0 or 2.</td>
      <td>INT64 (data type of array elements)</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>splitItem (uint64_t)</td>
      <td>Input</td>
      <td>Optional attribute, indicating whether to split the input and output tensors.</td>
      <td>The value can be 0, 1, 2, or 3. 0 indicates that neither the input nor the output is split. 1 indicates that the input is split and the output is not split. 2 indicates that the input is not split and the output is split. 3 indicates that both the input and output are split.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>group(char*)</td>
      <td>Input</td>
      <td>A string on the host identifying the communication domain name. The `commName` obtained via the HcclGetCommName API is used as the value for this parameter.</td>
      <td>The string length must be greater than 0.</td>
      <td>STRING</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>reduceOp(char*)</td>
      <td>Input</td>
      <td><code>reduce</code> operation type.</td>
      <td>Only "sum" is supported.</td>
      <td>STRING</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>commTurn (uint64_t)</td>
      <td>Input</td>
      <td>Number of times that communication data is split (Total data volume/Single communication volume).</td>
      <td>Only 0 is supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>streamMode (uint64_t)</td>
      <td>Input</td>
      <td>Enumeration of the ACL stream mode.</td>
      <td>Only 1 is supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>y(aclTensorList*)</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>The number of tensors in the list cannot exceed 64. When splitItem is 0, 2 to 6 dimensions are supported. When splitItem is 1, 2, or 3, only 2 dimensions are supported.</td>
      <td>FLOAT16, BFLOAT16 (data type of tensors in the list)</td>
      <td>ND</td>
      <td>2-6</td>
      <td>×</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>aclOpExecutor*</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
    <th>Return</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>1. A null pointer is passed in instead of the required input, output, or attribute.</td>
    </tr>
    <tr>
    <td class="merged-cell" rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
    <td class="merged-cell" rowspan="9">161002</td>
    <td>1. The data types and formats of x, weight, bias, groupListOptional, splitItem, and y are not supported.</td>
    </tr>
    <tr>
    <td>2. The length of x is neither 1 nor the length of weight.</td>
    </tr>
    <tr>
    <td>3. If bias is not empty, the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
    <td>4. When splitItem is 1, the length of x is not 1.</td>
    </tr>
    <tr>
    <td>5. When splitItem is 2, the length of y is not 1.</td>
    </tr>
    <tr>
    <td>6. When splitItem is 1 or 3, the lengths of x and y are not 1.</td>
    </tr>
    <tr>
    <td>7. reduceOp is not "sum".</td>
    </tr>
    <tr>
    <td>8. commTurn is not 0.</td>
    </tr>
    <tr>
    <td>9. streamMode is not 1.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatMulAllReduce

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGroupedMatMulAllReduceGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatMulAllReduce` defaults to a deterministic implementation.

- Constraints on data type combinations: The supported data type combinations for `x`, `weight`, and `bias` are as follows:
  - `x`: FLOAT16; `weight`: FLOAT16; `bias`: FLOAT16
  - `x`: BFLOAT16; `weight`: BFLOAT16; `bias`: FLOAT32
- Dimension constraints:
  - When `splitItem` is set to `0`, `x` and `y` both support 2D to 6D.
  - When `splitItem` is set to `1`, `2`, or `3`, both `x` and `y` support 2D.
  - Regardless of the value of `splitItem`, `weight` supports 2D.
- Device quantity: 2, 4, or 8 devices can be deployed.
- Tensor dimension size requirements:
  - The size of the last dimension for each tensor in `x` and `weight` must be less than 65536. (For `x`, the last dimension refers to the K-axis when `transpose_x` is false, and the M-axis when it is true; for `weight`, the last dimension refers to the N-axis when `transpose_weight` is false, and the K-axis when it is true.)
  - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).

## Calling Example

The following is sample code and is for reference only. For the detailed compilation and execution process, see the compilation and execution example.

```Cpp
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <iomanip>
#include <cassert>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include "hccl/hccl.h"
#include <hccl/hccl_types.h>

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "aclnnop/aclnn_grouped_mat_mul_all_reduce.h"
using namespace std;

#ifdef __aarch64__
    #define gettid() syscall(SYS_gettid)
#endif
#define N 128

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::string data_file;
    std::vector<int64_t> shape;
};

struct TensorsInfo {
    std::vector<TensorInfo> x_array;
    std::vector<TensorInfo> w_array;
    std::vector<TensorInfo> b_array;
    std::vector<TensorInfo> grouplist_array;
    std::vector<TensorInfo> y_array;
};

std::map<string, aclDataType> DtypeMap = {
    {"float16", ACL_FLOAT16},
    {"bfloat16", ACL_BF16},
    {"int64", ACL_INT64},
    {"uint64", ACL_UINT64}
};
thread_local int g_dev_id;
int32_t g_ndev = 0;
string input_dir = "./golden";
string output_dir = "./output";
string json_path;
vector<int> device_list;

int32_t loop = 1;

void set_device_id(int id)
{
    g_dev_id = id;
}
int get_device_id(void)
{
    return g_dev_id;
}

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define ACL_CHECK(ret)                                                                                     \
    do {                                                                                                   \
        auto retcode = ret;                                                                                \
        if (retcode != ACL_SUCCESS) {                                                                      \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
            return retcode;                                                                                \
        }                                                                                                  \
    } while (0)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)


struct DataSize {
    u64 min_bytes;
    u64 max_bytes;
    u64 step_bytes = 0;
    double step_factor;
    u64 count;
    u64 data_size;
    u64 type_size;
    int op_flag;
    void Print()
    {
        printf("acl min_bytes : %lld, max_bytes: %lld, step_bytes: %lld, step_factor: %f, count: %lld, data_size: "
               "%lld,  type_size: %lld\n",
            min_bytes,
            max_bytes,
            step_bytes,
            step_factor,
            count,
            data_size,
            type_size);
    }
};

struct Resource {
    aclrtStream rtStream;
    aclrtEvent startEvent;
    aclrtEvent endEvent;
    aclrtContext context;
  };

struct Args {
    int ndev;
    int rankId;
    uint32_t logicDeviceId;
    uint32_t rootRank;
    HcclComm hcclComm;
    Resource resources;
    DataSize dataPara;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    std::string dtype;
    std::string bin_path;
    std::string run_type;
    uint32_t loop_cnt;
    int bias_flag;
    std::map<std::string, int64_t> addrMap;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // (Boilerplate) Initialize resources.
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

int ReleaseAddr(vector<TensorInfo>& tensorList, map<string, int64_t> &addrMap) {
    int len = tensorList.size();
    if (len == 0 || (len == 1 && tensorList[0].shape.size() == 0)) {
        return 0;
    }
    for (size_t j=0; j < len; ++j) {
        string name = tensorList[j].name;
        if (addrMap.find(name) != addrMap.end()) {
            aclrtFree((void*)addrMap.at(name));
        }
    }
    return 0;
}

int DestroyTensor(void* items[], int item_type[], int num) {
  for (size_t i = 0; i < num; ++i) {
    if (items[i] != nullptr) {
      continue;
    }
    if (item_type[i] == 0) {
      aclDestroyTensor((aclTensor*)items[i]);
    } else if (item_type[i] == 1) {
      aclDestroyTensorList((aclTensorList*)items[i]);
    } else {
      aclDestroyIntArray((aclIntArray*)items[i]);
    }
  }
  return 0;
}

int CreateAclTensor(std::vector<int64_t>& shape,  std::string& name, std::string& dataTypeStr,
                    void*& deviceAddr, void*& tensor, bool int_array, int rankId=0) {
  uint32_t axis = 0;
  uint32_t byteBlock = 0;
  auto dataType = DtypeMap[dataTypeStr];
  uint32_t dataTypeSize = aclDataTypeSize(dataType);
  if (name.compare(0, 1, "x") == 0) {
    if (shape[1] % g_ndev != 0) {
      printf("[ERROR] X_k cannot be divided by ndev!\n");
      return -1;
    } 
    shape[1] /= g_ndev;
    axis = 1;
    byteBlock = shape[1] * dataTypeSize;
  } else if (name.compare(0, 6, "weight") == 0) {
    if (shape[0] % g_ndev != 0) {
      printf("[ERROR] Weight_k cannot be divided by ndev!\n");
      return -1;
    } 
    shape[0] /= g_ndev;
    axis = 0;
    byteBlock = shape[0] * dataTypeSize;
  } 
  auto size = GetShapeSize(shape) * dataTypeSize;
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(&deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  uint8_t* hostData = new (std::nothrow) uint8_t[size];
  if (int_array) {
    tensor = (void*)aclCreateIntArray((int64_t*)hostData, size / sizeof(int64_t));
    delete[] hostData;
    return 0; 
  }
  ret = aclrtMemcpy(deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  tensor = (void*)aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,shape.data(), shape.size(), deviceAddr);
  return 0;
}

int CreateTensor_(int i, vector<TensorInfo>& tensorList, map<string, int64_t> &addrMap, void* items[], int item_type[], bool copy, int rankId) {
    int len = tensorList.size();
    if (len == 0 || (len == 1 && tensorList[0].shape.size() == 0)) {
        items[i] = nullptr;
        item_type[i] = 1;
        return 0;
    }
    void* tensors[len];
    for (int j = 0; j < len; ++j) {
        string name = tensorList[j].name;
        vector<int64_t> shape = tensorList[j].shape;
        string dtype = tensorList[j].dtype;
        void* deviceAddr = nullptr;
        auto ret = CreateAclTensor(shape, name, dtype, deviceAddr, tensors[j], false, rankId);
        addrMap[name] = (int64_t)deviceAddr;
        if (ret != 0) {return -1;}
    }
    items[i] = (void*)aclCreateTensorList((aclTensor**)tensors, len);
    item_type[i] = 1;
    return 0;
}


int CreateTensor(TensorsInfo& config, map<string, int64_t> &addrMap, void* items[], int item_type[], bool copy, int rankId) {
    int ret = 0;
    if (!copy) {
        ret = CreateTensor_(0, config.y_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
    } else {
        ret = CreateTensor_(0, config.x_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
        ret = CreateTensor_(1, config.w_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
        ret = CreateTensor_(2, config.b_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}

        bool int_array = true;
        void* deviceAddr = nullptr;
        TensorInfo& grouplist = config.grouplist_array[0];
        ret = CreateAclTensor(grouplist.shape, grouplist.name, grouplist.dtype, deviceAddr, items[3], int_array, rankId);
        addrMap[grouplist.name] = (int64_t)deviceAddr;
        item_type[3] = 2;
        if (ret != 0) {return -1;}
    }
    return 0;
}

int PrepareDeviceMemAndData(TensorsInfo& config, std::map<std::string, int64_t>& addrMap, void** inputs, void** outputs, int* input_type, int* output_type, int rankId) {
  int ret = 0;
  ret = CreateTensor(config, addrMap, inputs, input_type, true, rankId);
  CHECK_RET(ret == ACL_SUCCESS, printf("CreateTensor Failed\n"); return -1);
  ret = CreateTensor(config, addrMap, outputs, output_type, false, rankId);
  CHECK_RET(ret == ACL_SUCCESS, printf("CreateTensor Failed\n"); return -1);
  return 0;
}

typedef int32_t rtError_t;
typedef void *rtNotify_t;
typedef void *rtStream_t;

// init hccl args end
extern "C" rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm);
extern "C" rtError_t rtNotifyRecord(rtNotify_t notify, rtStream_t stm);
extern "C" rtError_t rtNotifyCreate(int32_t deviceId, rtNotify_t *notify);
extern "C" HcclResult HcclCreateComResource(char *commName, uint32_t streamMode, void **commContext);
extern "C" HcclResult HcclGetAicpuOpStreamNotify(char *commName, rtStream_t *Opstream, void **aicpuNotify);

int callMatmulAndAicpu(HcclComm hcclComm, void **inputs, void **outputs, int* input_type, int* output_type, DataSize &dataPara, aclrtStream rtStream, aclrtEvent startEvent, aclrtEvent endEvent, int checkErr, int ndev, int rankId, int deviceId, int rootRank, std::string dtype, std::map<std::string, int64_t>& addrMap)  // Main function
{
    struct timespec tp1, tp2;
    long cost;
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    set_device_id(deviceId);

    // **********Add Hcommname start********
    char hcom_name[N];
    auto hccl_name_ret = HcclGetCommName(hcclComm, hcom_name);
    CHECK_RET(
        hccl_name_ret == ACL_SUCCESS, printf("[ERROR] HcclGetCommName failed. hccl_name_ret = %d \n", hccl_name_ret);
        return -1);
    printf("rank %d hcom name is %s\n", rankId, hcom_name);
    // **********Add Hcommname end*********

    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    int64_t split_item = 0;
    inputs[3] = nullptr;
    int64_t comm_turn = 0;
    
    auto aclnnRet = aclnnGroupedMatMulAllReduceGetWorkspaceSize(
        (aclTensorList*)inputs[0], (aclTensorList*)inputs[1], (aclTensorList*)inputs[2], (aclIntArray*)inputs[3], split_item,  hcom_name, "sum", comm_turn, 1, (aclTensorList*)outputs[0], &workspaceSize, &executor);
    CHECK_RET(aclnnRet == ACL_SUCCESS,
                printf("[ERROR] aclnnGroupedMatMulAllReduceGetWorkspaceSize failed. aclnnRet = %d \n", aclnnRet);
                return -1);
    printf("[INFO] gmm_allreduce workspaceSize = %lu\n", workspaceSize);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    aclnnRet = aclnnGroupedMatMulAllReduce(workspaceAddr, workspaceSize, executor, rtStream);
    CHECK_RET(aclnnRet == ACL_SUCCESS, printf("[ERROR] aclnnGroupedMatMulAllReduce failed. aclnnRet = %d \n", aclnnRet);
                return -1);
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(rtStream, 10000));
    if (workspaceAddr != nullptr) {
      auto ret = aclrtFree(workspaceAddr);
      CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnFree workspaceAddr failed. ret = %d \n", ret);
                return -1);
    }

    clock_gettime(CLOCK_MONOTONIC, &tp2);
    if (tp2.tv_sec != tp1.tv_sec) {
        cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
    } else {
        cost = tp2.tv_nsec - tp1.tv_nsec;
    }
    printf("[INFO] mc2 costtime = %lu, deviceid = %d\n", cost, rankId);

    return 0;
}

int launchOneThread(Args &args, TensorsInfo& config)
{
    std::string thread_name = "test" + std::to_string(args.logicDeviceId);
    prctl(PR_SET_NAME, thread_name.c_str());
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    int numa_node = args.logicDeviceId;
    int cpu_num = 8;
    int cpu_start = cpu_num * args.rankId;
    for (int i = cpu_start; i < cpu_start + cpu_num; i++) {
        CPU_SET(i, &cpu_set);
    }
    int ret = sched_setaffinity(gettid(), sizeof(cpu_set), &cpu_set);
    if (ret) {
        printf("failed to set cpu affinity, errno = %d\n", errno);
        return -1;
    }
    printf("0 rankId: %d, stream: %p, context : %p\n",
        args.logicDeviceId,
        args.resources.rtStream,
        args.resources.context);

    DataSize dataPara = args.dataPara;
    int checkErr = 0;
    ACL_CHECK(aclrtSetDevice(args.logicDeviceId));
    ACL_CHECK(aclrtSetCurrentContext(args.resources.context));
    
    constexpr int input_num = 4;
    void* inputs[input_num];
    int input_type[input_num];
    constexpr int output_num = 1;
    void* outputs[output_num];
    int output_type[output_num];
    args.addrMap.clear();
    ret = PrepareDeviceMemAndData(config, args.addrMap, inputs, outputs, input_type, output_type, args.rankId);
    CHECK_RET(ret == 0, return ret);

    for (int i = 0; i < args.loop_cnt; i++) {
        printf("======Startloop: %d / %d ============\n", i, args.loop_cnt);

        ret = callMatmulAndAicpu(args.hcclComm,
            inputs,
            outputs,
            input_type,
            output_type,
            dataPara,
            args.resources.rtStream,
            args.resources.startEvent,
            args.resources.endEvent,
            checkErr,
            args.ndev,
            args.logicDeviceId,
            args.rankId,
            args.rootRank,
            args.run_type,
            args.addrMap);
        if (ret != 0) {
            printf("TestCall execute AICPU failed, Detailed logs are stored in path: /root/ascend/log/");
            return ret;
        }
    }
    // save output
    // if (args.logicDeviceId == args.rootRank) {
    //   SaveTensor(config["outputs"], args.addrMap, output_dir);
    // }
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.resources.rtStream, 10000));
    DestroyTensor(inputs, input_type, input_num);
    DestroyTensor(outputs, output_type, output_num);
    ReleaseAddr(config.x_array, args.addrMap);
    ReleaseAddr(config.w_array, args.addrMap);
    ReleaseAddr(config.b_array, args.addrMap);
    ReleaseAddr(config.grouplist_array, args.addrMap);
    ReleaseAddr(config.y_array, args.addrMap);

    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.resources.rtStream, 10000));
    ACL_CHECK(aclrtDestroyStreamForce(args.resources.rtStream));
    // Destroy the collective communication domain.
    //  HCCLCHECK(HcclCommDestroy(args.hcclComm));
    HcclCommDestroy(args.hcclComm);
    // Reset devices.
    ACL_CHECK(aclrtResetDevice(args.logicDeviceId));
    return 0;
}

int launchMultiThread(Args &input_args, int32_t *devices, HcclComm *comms, Resource *resources)
{
    DataSize dataPara;
    dataPara.data_size = 10485760;
    dataPara.type_size = sizeof(uint16_t);
    dataPara.count = (dataPara.data_size + sizeof(uint16_t) - 1) / sizeof(uint16_t);  // data->count;
    dataPara.step_factor = 1;                                                         // data->step_factor;
    dataPara.step_bytes = 1;                                                          // data->step_bytes;
    dataPara.max_bytes = 10485760;                                                    // data->max_bytes;
    dataPara.min_bytes = 10485760;                                                    // data->min_bytes;
    dataPara.op_flag = 0;                                                             // op_flag;
    input_args.bin_path = "";
    input_args.dtype = "float16";

    TensorsInfo config;
    TensorInfo x0_json;
    x0_json.name = "x0";
    x0_json.dtype = "float16";
    x0_json.data_file = "x_0.bin";
    x0_json.shape = vector<int64_t>{256,256};
    TensorInfo x1_json;
    x1_json.name = "x1";
    x1_json.dtype = "float16";
    x1_json.data_file = "x_1.bin";
    x1_json.shape = vector<int64_t>{1024,256};
    config.x_array.push_back(x0_json);
    config.x_array.push_back(x1_json);
    
    TensorInfo w0_json;
    w0_json.name = "weight0";
    w0_json.dtype = "float16";
    w0_json.data_file = "w_0.bin";
    w0_json.shape = vector<int64_t>{256,256};
    TensorInfo w1_json;
    w1_json.name = "weight1";
    w1_json.dtype = "float16";
    w1_json.data_file = "w_1.bin";
    w1_json.shape = vector<int64_t>{256, 1024};
    config.w_array.push_back(w0_json);
    config.w_array.push_back(w1_json);

    TensorInfo b0_json;
    b0_json.name = "b0";
    b0_json.dtype = "float16";
    b0_json.data_file = "b_0.bin";
    b0_json.shape = vector<int64_t>{256};
    TensorInfo b1_json;
    b1_json.name = "b1";
    b1_json.dtype = "float16";
    b1_json.data_file = "b_1.bin";
    b1_json.shape = vector<int64_t>{1024};
    config.b_array.push_back(b0_json);
    config.b_array.push_back(b1_json);
    
    
    TensorInfo group_list_json;
    group_list_json.name = "group_list";
    group_list_json.dtype = "int64";
    group_list_json.data_file = "";
    group_list_json.shape = vector<int64_t>{};
    config.grouplist_array.push_back(group_list_json);
    
    TensorInfo y0_json;
    y0_json.name = "y0";
    y0_json.dtype = "float16";
    y0_json.data_file = "y_0.bin";
    y0_json.shape = vector<int64_t>{256, 256};
    TensorInfo y1_json;
    y1_json.name = "y1";
    y1_json.dtype = "float16";
    y1_json.data_file = "y_1.bin";
    y1_json.shape = vector<int64_t>{1024, 1024};
    config.y_array.push_back(y0_json);
    config.y_array.push_back(y1_json);

    uint32_t ndev = input_args.ndev;
    Args args[ndev];
    aclrtStream rtStream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ACL_CHECK(aclrtSetDevice(devices[rankId]));
        ACL_CHECK(aclrtGetCurrentContext(&context[rankId]));
        ACL_CHECK(aclrtCreateStream(&rtStream[rankId]));
        // printf("1 rankId: %d, stream: %p\n", rankId,  rtStream[rankId]);
    }

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rootRank = devices[0];
        args[rankId].ndev = ndev;
        args[rankId].logicDeviceId = devices[rankId];
        args[rankId].hcclComm = comms[rankId];
        args[rankId].resources = resources[rankId];
        args[rankId].dataPara = dataPara;
        args[rankId].resources.rtStream = rtStream[rankId];
        args[rankId].resources.context = context[rankId];
        args[rankId].m = input_args.m;
        args[rankId].k = input_args.k;
        args[rankId].n = input_args.n;
        args[rankId].dtype = input_args.dtype;
        args[rankId].bin_path = input_args.bin_path;
        args[rankId].run_type = input_args.run_type;
        args[rankId].loop_cnt = input_args.loop_cnt;
        args[rankId].bias_flag = input_args.bias_flag;
        threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId]), std::ref(config)));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    return 0;
}

int InitParams(int argc, char** argv) {
  if (argc < 4) {
    LOG_PRINT("miss paramaters\n");
  }
  g_ndev = std::stoi(argv[1]);
  loop = std::stoi(argv[2]);
  std::stringstream str = std::stringstream(argv[3]);
  std::string seg;
  char c = ',';
  while(std::getline(str, seg, c)) {
    device_list.push_back(std::stoi(seg));
  }
  assert (g_ndev == device_list.size());
  return 0;
}


int main(int argc, char *argv[])
{
    // This example is implemented based on Atlas A3 and can only run on Atlas A3.
    // usage: ./main 2 1 0,1
    int ret = 0;
    CHECK_RET(InitParams(argc, argv) == 0, LOG_PRINT("init params failed.\n"); return 0);
    Args input_args;
    input_args.ndev = g_ndev;
    input_args.run_type = string("reduce");
    input_args.loop_cnt = loop;

    if (input_args.ndev != 1 && input_args.ndev != 2 && input_args.ndev != 4 && input_args.ndev != 8) {
        printf("device_num input error, only support 8,4,2,1.\n");
        return -1;
    }
    int32_t devices[input_args.ndev];
    for (int i = 0; i < input_args.ndev; i++) {
        devices[i] = device_list[i];
    }

    HcclComm comms[N];
    Resource resources[N];

    ACL_CHECK(aclInit(NULL));
    for (int i = 0; i < 1; i++) {
        // Initialize the collective communication domain.
        for (int j = 0; j < input_args.ndev; j++) {
            ACL_CHECK(aclrtSetDevice(devices[j]));
        }
        ret = HcclCommInitAll(input_args.ndev, devices, comms);
        if (ret != 0) {
            printf("This is an error in init_hcclComm.\n");
            return -1;
        }

        // Start the test.
        ret = launchMultiThread(input_args, devices, comms, resources);
        if (ret != 0) {
            printf("This is an error in opbase_test_by_data_size.\n");
            // ACL_CHECK(aclFinalize());
            return -1;
        }
    }
    ACL_CHECK(aclFinalize());
    return ret;
}
```
