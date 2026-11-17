# aclnnEngramHash

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：Engram n-gram hash 索引计算。对每层 `l`、每个 token `t`、每个输出列 `c`，按 n-gram 长度逐级计算 int64 多项式 hash 并取模 vocab，加上 embedding 表偏移，输出 INT32 索引。

- 计算公式（对每个输出元素 `[l, t, c]`，`last = c / tables + 1`，`tb = c % tables`）：

  $$
  \begin{aligned}
  hash &= \mathrm{int64}(ids[t, 0]) \times mult[l, 0] \\
  hash &\mathrel{\oplus}= \mathrm{int64}(ids[t, k]) \times mult[l, k], \quad k = 1 \dots last \\
  r &= hash \bmod \mathrm{int64}(vocab[l, last-1, tb]) \\
  out[l, t, c] &= \mathrm{int32}(r) + offs[l, c]
  \end{aligned}
  $$

  hash 中间量为 int64（乘积可达 2^62），`%` 为截断取模（truncmod），余数为负时加 `V` 修正。

- 示例：

  ```python
  ngram_token_ids shape: [4096, 3]           # num_tokens=4096, max_ngram_size=3
  multipliers shape:      [2, 3]             # num_ngram_layers=2
  vocab_sizes shape:     [2, 2, 8]           # tables=8
  offsets shape:          [2, 16]            # num_out_cols=(3-1)*8=16
  hash_indices shape: [2, 4096, 16]          # INT32 输出
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnEngramHashGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEngramHash”接口执行计算。

```cpp
aclnnStatus aclnnEngramHashGetWorkspaceSize(
  const aclTensor *ngramTokenIds,
  const aclTensor *multipliers,
  const aclTensor *vocabSizes,
  const aclTensor *offsets,
  aclTensor       *hashIndices,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnEngramHash(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnEngramHashGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |---|---|---|---|---|---|---|---|
  | ngramTokenIds（const aclTensor*） | 输入 | N-gram token ids。 | <li>支持空Tensor</li><li>不为空Tensor时必须为二维且 max_ngram_size ≥ 2。</li> | INT32 | ND | [num_tokens, max_ngram_size] | √ |
  | multipliers（const aclTensor*） | 输入 | 每层 hash 乘子。 | <li>支持空Tensor</li><li>不为空Tensor时 shape 必须为 [num_ngram_layers, max_ngram_size] 且 num_ngram_layers ≥ 1。</li> | INT64 | ND | [num_ngram_layers, max_ngram_size] | √ |
  | vocabSizes（const aclTensor*） | 输入 | 每个 n-gram 长度每张 embedding 表的大小。 | <li>支持空Tensor</li><li>不为空Tensor时必须为三维且 dim1 == max_ngram_size - 1，num_embed_table_per_ngram ≥ 1。</li> | INT32 | ND | [num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram] | √ |
  | offsets（const aclTensor*） | 输入 | 每层 embedding 表偏移（词表前缀和）。 | <li>支持空Tensor</li><li>不为空Tensor时 shape 必须为 [num_ngram_layers, num_out_cols] 且 num_out_cols == (max_ngram_size - 1) * num_embed_table_per_ngram。</li> | INT32 | ND | [num_ngram_layers, num_out_cols] | √ |
  | hashIndices（aclTensor*） | 输出 | hash 后的 embedding 索引。 | <li>支持空Tensor</li><li>不为空Tensor时 shape 为 [num_ngram_layers, num_tokens, num_out_cols]。</li> | INT32 | ND | [num_ngram_layers, num_tokens, num_out_cols] | × |
  | workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
  | executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的ngramTokenIds、multipliers、vocabSizes、offsets、hashIndices是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入或输出的shape不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnEngramHash

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  |---|---|---|
  | workspace | 输入 | 在Device侧申请的workspace内存地址。 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnEngramHashGetWorkspaceSize获取。 |
  | executor | 输入 | op执行器，包含了算子计算流程。 |
  | stream | 输入 | 指定执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：默认确定性实现。
- dtype 固定：ngram_token_ids INT32、multipliers INT64、vocab_sizes INT32、offsets INT32，输出 INT32。
- ngram_token_ids 必须为二维 `[num_tokens, max_ngram_size]` 且 max_ngram_size ≥ 2；vocab_sizes 必须为三维 `[num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram]` 且 num_ngram_layers ≥ 1、num_embed_table_per_ngram ≥ 1；multipliers shape 须为 `[num_ngram_layers, max_ngram_size]`；offsets shape 须为 `[num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram]`。
- 无 num_tokens 规模上限：tileTokens 随规模自适应。
- max_ngram_size 有上限，需要能满足全载 SIMT 的 UB 窗口。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)，完整代码参见 [test_aclnn_engram_hash.cpp](../examples/test_aclnn_engram_hash.cpp)。

```cpp
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_engram_hash.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法, 资源初始化
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
                    aclDataType dataType, aclTensor** tensor)
{
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

// 测试配置: num_tokens / max_ngram_size / num_ngram_layers / num_embed_table_per_ngram
struct HashCase {
    int64_t numTokens;
    int64_t maxNgramSize;
    int64_t numNgramLayers;
    int64_t numTables;
    const char* desc;
};

// CPU 参考实现, 与 tilelang engram_hash_kernel 语义位级一致。
static void EngramHashRef(const std::vector<int32_t>& ids, const std::vector<int64_t>& mult,
                          const std::vector<int32_t>& vocab, const std::vector<int32_t>& offs, const HashCase& c,
                          std::vector<int32_t>& out)
{
    const int64_t T = c.numTokens;
    const int64_t N = c.maxNgramSize;
    const int64_t L = c.numNgramLayers;
    const int64_t tables = c.numTables;
    const int64_t C = (N - 1) * tables;
    out.assign(static_cast<size_t>(L * T * C), 0);
    for (int64_t l = 0; l < L; ++l) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t col = 0; col < C; ++col) {
                const int64_t last = col / tables + 1;
                const int64_t tb = col % tables;
                // hash 中间量为 int64 (乘积可达 2^62), 逐级 XOR。
                int64_t hashValue = static_cast<int64_t>(ids[t * N]) * mult[l * N];
                for (int64_t k = 1; k <= last; ++k) {
                    hashValue ^= static_cast<int64_t>(ids[t * N + k]) * mult[l * N + k];
                }
                const int64_t vocabSize = static_cast<int64_t>(vocab[(l * (N - 1) + (last - 1)) * tables + tb]);
                int64_t remainder = hashValue % vocabSize; // truncmod
                if (remainder < 0) {
                    remainder += vocabSize;
                }
                out[(l * T + t) * C + col] = static_cast<int32_t>(remainder) + offs[l * C + col];
            }
        }
    }
}

static int RunCase(const HashCase& c, aclrtStream stream)
{
    const int64_t T = c.numTokens;
    const int64_t N = c.maxNgramSize;
    const int64_t L = c.numNgramLayers;
    const int64_t tables = c.numTables;
    const int64_t C = (N - 1) * tables;

    // 1.host 侧生成随机数据: ids/mult ∈ [0, 100000), vocab ∈ [100000, 1000000), offsets 为 vocab 前缀和
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int32_t> idDist(0, 99999);
    std::uniform_int_distribution<int32_t> vocabDist(100000, 999999);
    std::vector<int32_t> idsHost(static_cast<size_t>(T * N));
    for (auto& v : idsHost) {
        v = idDist(rng);
    }
    std::vector<int64_t> multHost(static_cast<size_t>(L * N));
    for (auto& v : multHost) {
        v = idDist(rng);
    }
    std::vector<int32_t> vocabHost(static_cast<size_t>(L * (N - 1) * tables));
    for (auto& v : vocabHost) {
        v = vocabDist(rng);
    }
    std::vector<int32_t> offsHost(static_cast<size_t>(L * C), 0);
    for (int64_t l = 0; l < L; ++l) {
        int32_t prefix = 0;
        for (int64_t col = 0; col < C; ++col) {
            offsHost[l * C + col] = prefix;
            prefix += vocabHost[l * C + col];
        }
    }
    std::vector<int32_t> outHost(static_cast<size_t>(L * T * C), 0);

    // 2.构造输入与输出 aclTensor
    void* idsAddr = nullptr;
    void* multAddr = nullptr;
    void* vocabAddr = nullptr;
    void* offsAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* idsTensor = nullptr;
    aclTensor* multTensor = nullptr;
    aclTensor* vocabTensor = nullptr;
    aclTensor* offsTensor = nullptr;
    aclTensor* outTensor = nullptr;
    auto ret = CreateAclTensor(idsHost, {T, N}, &idsAddr, aclDataType::ACL_INT32, &idsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(multHost, {L, N}, &multAddr, aclDataType::ACL_INT64, &multTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vocabHost, {L, N - 1, tables}, &vocabAddr, aclDataType::ACL_INT32, &vocabTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsHost, {L, C}, &offsAddr, aclDataType::ACL_INT32, &offsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, {L, T, C}, &outAddr, aclDataType::ACL_INT32, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API: aclnn 两段式接口 (无属性)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnEngramHashGetWorkspaceSize(idsTensor, multTensor, vocabTensor, offsTensor, outTensor, &workspaceSize,
                                          &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramHashGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnEngramHash(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramHash failed. ERROR: %d\n", ret); return ret);

    // 4.同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.拷回 host 与 CPU 参考实现位级比对
    ret = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(int32_t), outAddr, outHost.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    std::vector<int32_t> expected;
    EngramHashRef(idsHost, multHost, vocabHost, offsHost, c, expected);
    int64_t mismatch = 0;
    int64_t maxErr = 0;
    int64_t firstBad = -1;
    for (size_t i = 0; i < expected.size(); ++i) {
        int64_t err = static_cast<int64_t>(outHost[i]) - static_cast<int64_t>(expected[i]);
        if (err < 0) {
            err = -err;
        }
        if (err != 0) {
            ++mismatch;
            if (firstBad < 0) {
                firstBad = static_cast<int64_t>(i);
            }
        }
        if (err > maxErr) {
            maxErr = err;
        }
    }
    LOG_PRINT("case %-16s T=%-5ld N=%ld L=%ld tables=%-2ld C=%-3ld | mismatch=%ld / %zu, maxErr=%ld, firstBad=%ld, "
              "%s\n",
              c.desc, T, N, L, tables, C, mismatch, expected.size(), maxErr, firstBad, mismatch == 0 ? "PASS" : "FAIL");

    // 6.释放aclTensor与device资源
    aclDestroyTensor(idsTensor);
    aclDestroyTensor(multTensor);
    aclDestroyTensor(vocabTensor);
    aclDestroyTensor(offsTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(idsAddr);
    aclrtFree(multAddr);
    aclrtFree(vocabAddr);
    aclrtFree(offsAddr);
    aclrtFree(outAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return mismatch == 0 ? 0 : 1;
}

int main()
{
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.多组配置验证 (含模型默认 shape N=3/L=2/tables=8 与 C=24 的 N=4 配置)
    const std::vector<HashCase> cases = {
        {4096, 3, 2, 8, "model_default"}, {42, 3, 2, 8, "small_tokens"}, {1, 3, 2, 8, "single_token"},
        {2048, 4, 2, 8, "ngram4_c24"},    {128, 2, 1, 16, "ngram2_l1"},  {100, 7, 5, 3, "ngram7_l5"},
    };
    int failed = 0;
    for (const auto& c : cases) {
        if (RunCase(c, stream) != 0) {
            ++failed;
        }
    }
    LOG_PRINT("%s: %zu cases, %d failed\n", failed == 0 ? "ALL PASS" : "FAILED", cases.size(), failed);

    // 3.释放device资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
```
