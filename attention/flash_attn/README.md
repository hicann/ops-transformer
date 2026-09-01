# FlashAttn

## 功能说明

- 算子功能：基于FlashAttention算法实现self-attention（自注意力）计算，训练推理归一化，仅支持非量化场景。

- 计算公式：

  自注意力的正向计算公式如下：

  $$
  Attention(Q,K,V)=Softmax(SoftmaxScale * (Q \cdot K^T))V
  $$

  其中$Q \in (Q_S, Q_N, D)$，$K, V \in (KV_S, KV_N, D)$，$D$为head dim。默认$SoftmaxScale=1.0$，标准实现中通常取$1/\sqrt{D}$作为缩放因子。

  FlashAttention算法将$Q_S \times KV_S$的注意力矩阵按块（tile）分块计算，避免实例化完整注意力矩阵，显存复杂度由$O(Q_S \cdot KV_S)$降至$O(tile_M \cdot tile_N)$。

## Quick Start

### 1. custom 包编译与安装

完整脚本（整合编译与安装步骤；`CANN_DIR` 为 CANN 安装根目录，按实际调整）：

```bash
# 前置：加载 CANN 环境
source ${CANN_DIR}/cann/set_env.sh

# 清理历史构建产物（避免残留影响增量编译）
rm -rf ./build ./build_out
rm -rf ${CANN_DIR}/vendors

# 编译（flash_attn 与 flash_attn_metadata 需同时编译，生成所有 tiling key 的 kernel 变体）
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata

# 安装
cd build_out
./cann-ops-transformer-*.run --install-path=${CANN_DIR}
```

编译产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`

**可选参数说明**：

- **`-j`（限制并行线程数）**：默认按机器核数并行。当机器内存不足、或 cgroup 实际限制核数小于 `/proc/cpuinfo` 报告值（如本机报告 128 核但 cgroup 限制 32 核）导致编译 OOM 或失败时，需显式指定较小值：

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata -j16
  ```

- **`--op_debug_config dump_cce`（保留内核编译中间产物）**：kernel 调试（如上板查汇编）时需要保留 `.cce` 中间产物，可在编译命令后追加：

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata -j16 --op_debug_config dump_cce
  ```

- **`--tiling_key`（只编译指定 tiling key 变体）**：默认全量编译会生成所有 tiling key 的 kernel 变体，耗时较长。当只需要调试/验证特定布局与配置组合（如仅 BSND/BNSD/TND × 无 mask × D=128）时，可只编译对应变体以加速：

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --tiling_key="2279866368;2279866369;2279866370"
  ```

  各 tiling key 的编码含义见[开发者指南 §5](#5-tilingkey)。

### 2. torch 扩展包构建与安装

使用 torch 接口前必做。在仓根目录构建 torch 扩展 whl 并安装：

```bash
# 前置：加载 CANN 环境
source ${CANN_DIR}/cann/set_env.sh

# 清理 torch 扩展缓存（~ 为当前用户 home，需与安装/运行 torch 的用户一致，避免加载过期编译产物）
rm -rf ~/.cache/torch_extensions/*

# 构建 torch 扩展 whl（全量包，包名 cann_ops_transformer 保持不变，whl 输出至 build_out/）
bash build.sh --torch_extension --soc=ascend950

# 安装
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

**安装后验证**：

```bash
python3 -c "from cann_ops_transformer.ops import flash_attn, flash_attn_metadata; print('ok')"
```

### 3. 接口调用

- **torch 接口**（`flash_attn_metadata` + `flash_attn` 的函数原型、参数说明、返回值说明）：[flash_attn.md](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md)

调用分两步：先用`flash_attn_metadata`生成负载均衡metadata，再调用`flash_attn`主算子。完整调用示例（含代码）见接口文档的[调用示例](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md#调用示例)章节。

导入路径与安装包名一致（按步骤 2 构建的全量包）：

```python
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
```

### 4. NPU上板运行算子精度测试

前置：加载 CANN 与已安装 custom 包的环境：

```bash
source ${CANN_DIR}/cann/set_env.sh
source ${CANN_DIR}/vendors/custom_transformer/bin/set_env.bash
```

执行仓库内 pytests（用例在 `tests/pytests/`，npu/gpu/cpu 三后端）：

```bash
cd attention/flash_attn/tests/pytests
python3 test_flash_attn.py --case_files functional_redline_infer --case_id Decode_Unquant_BF16_BF16_L0_1_5_1_1_2048_BNSD_BSND_000004
python3 test_flash_attn.py --device_id 0            # 跑全部functional用例
python3 test_flash_attn.py --device_id 0 --case_id <id>   # 跑单个用例
python3 test_flash_attn.py --device_id 0 --use_gpu --gpu_device 0   # 用GPU端flash_attn库对照
```

pytests 框架说明见 [tests/pytests/readme.md](./tests/pytests/readme.md)（functional/redline/perf 用例矩阵、结果CSV输出）。

### 5. 性能测试

用例文件：`tests/pytests/test_cases/performance_redline_train.py`、`performance_redline_infer.py`（train/infer 各 220+ 性能用例）。

```bash
cd attention/flash_attn/tests/pytests

# 批量性能采集（train 性能用例集，输出到指定目录）
python3 test_flash_attn.py --case_files performance_redline_train --perf_mode --perf_output ./perf_out

# 逐个执行（--one_by_one 逐个 case 执行，便于观察与中断）
python3 test_flash_attn.py --perf_mode --one_by_one
```

性能相关参数：`--perf_runs`（采集轮数，默认 5）、`--perf_cold_thr`（冷/热 case 分界 S1，默认 16）、`--perf_output`（输出目录，默认 `./perf_output`）。结果解析与报告生成见 `tests/pytests/utils/perf_runner.py`、`perf_parser.py`。

### 6. 运行 tiling 单元测试

```bash
bash build.sh --ops=flash_attn,flash_attn_metadata --soc=ascend950
cd build && ctest -R flash_attn_tiling --output-on-failure
```

用例定义：`tests/ut/op_host/arch35/test_flash_attn_tiling.csv`。另有 inferShape 与 inferDtype 单测：`tests/ut/op_host/test_flash_attn_shape_infershape.cpp`、`test_flash_attn_dtype_infershape.cpp`。

## 开发者指南

### 1. 基础概念

#### 1.1 MHA 与 GQA

- **MHA（Multi-Head Attention）**：Q、K、V 各有一组 head，Q_N = KV_N，每个 Q head 各自与对应的 KV head 做注意力。
- **GQA（Grouped Query Attention）**：Q_N 是 KV_N 的整数倍，即 g = Q_N / KV_N，同一组 g 个 Q head 共享一份 KV。用于减少 KV 的显存与计算量（Q_N 远大 KV_N 时，KV cache 显著缩小）。
- 本算子要求 Q_N % KV_N == 0（GQA 场景）。kernel 中 `gSize = g`，任务按 (bN2, gS1, s2) 三层循环展开，其中 bN2 = batch × KV_N 扁平索引，gS1 = Q 序列 × g 的扁平索引。

#### 1.2 PagedAttention 场景及 layout_kv 取值

- **PagedAttention**：将 KV cache 按 block（block_size 个 token）分页存储，通过 `block_table` 索引物理 block，避免 KV cache 预留过大的连续显存，支持动态分配与复用。推理场景（长序列 decode 且 KV cache 会增长）通常使用。
- 启用方式：`layout_kv` 取 PA 布局，同时传入 `block_table`（INT32，shape (B, max_blocks_per_seq)）。block_size 由 KV 的 PA 布局推断，取值需 ∈ [16, 1024] 且为 16 的整数倍。
- `layout_kv` 取值及对应 KV shape：

| layout_kv | KV shape | 说明 |
|---|---|---|
| BSND / BNSD / TND | (B, S, N, D) / (B, N, S, D) / (T, N, D) | 连续 KV，非分页 |
| PA_BBND | (num_blocks, block_size, KV_N, D) | block 内按 BSND 排布 |
| PA_BNBD | (num_blocks, KV_N, block_size, D) | block 内按 BNSD 排布 |
| PA_NZ | (num_blocks, KV_N, D/16, block_size, 16) | NZ 分块格式 |

- PA 场景约束：PA 时 `seqused_kv` 必填；非 PA 场景传入 `block_table` 会直接报错。`layout_kv` 为 PA 时，`layout_q`/`layout_out` 维持 BSND/BNSD/TND 不变。

#### 1.3 变长序列（TND）

- **TND 布局**：将 batch 内所有序列的 token 拼接为一个长序列（总长度 Q_T / KV_T），shape 记为 (Q_T, Q_N, D) 与 (KV_T, KV_N, D)，消除 padding 浪费。用于 batch 内序列长度不等的场景。
- 启用方式：`layout_q`（及 `layout_kv`）取 TND，同时传入 `cu_seqlens_q`/`cu_seqlens_kv` 描述每个序列的起始位置。
- TND 场景 `max_seqlen` 属性不参与实际切分（以 cu_seqlens 为准），`layout_out` 必须为 TND。
- 功能说明中的"Packed Sequence"即 TND 场景。

#### 1.4 cu_seqlens 与 seqused

- **cu_seqlens_q / cu_seqlens_kv**：INT32 一维，shape (B+1,)，单调不减且首元素为 0，表示每个序列的累积起点/终点（TND 布局用于恢复各序列的边界）。仅 TND 布局允许传入。
- **seqused_q / seqused_kv**：INT32 一维，shape (B,)，表示每个 batch 实际生效的序列长度（截断冗余 padding）。PA 场景下 `seqused_kv` 必填，用于确定每个 batch 实际 KV 长度。
- 这两个字段的意义：cu_seqlens 定位序列边界（TND），seqused 做长度截断（PA/变长）；两者同时传入时以实际较短者生效。

#### 1.5 Mask（mask_mode 与 attn_mask）

- **mask_mode** 属性选择掩码模式：0 = 全计算（无掩码）、3 = causal（下三角）、4 = window（band，前后各 win_left/win_right 个 token）。
- **attn_mask 输入**：INT8 模板矩阵，固定 shape (2048, 2048)，kernel 按 tile 从中扣取所需子块，适配任意 S1/S2。mask_mode=3/4 时必须传入 attn_mask；mask_mode=0 时禁止传入 attn_mask 且 win_left/win_right 必须为 -1。
- mask_mode=4（window）时 win_left/win_right 定义窗口宽度，kernel 据此跳过无效 tile。

#### 1.6 LSE（softmax_lse）

- **LSE（Log-Sum-Exp）**：`return_softmax_lse=True` 时输出每行 softmax 的 log 分母：lse = log(Σ e^{s-m}) + m，其中 m 为行内 max 值。
- 输出 shape：BSND/BNSD → (B, Q_N, Q_S)；TND → (Q_T, Q_N)；dtype 为 FLOAT32。
- 典型用途：跨 batch 的 KV cache 拼接（如 speculative decoding 的概率融合）、断点续训时的 softmax 状态重建。

#### 1.7 行无效（RowInvalid）

- **概念**：当某一行（一个 Q head 的一行 Q token）的所有 token 均被掩码排除（如 window 窗口外、seqused 截断、mask 全 1），该行没有任何有效参与计算的元素时，称为**无效行**。kernel 以该行 softmax max 是否仍为 `-inf`（softmaxMax == -inf，即没有任何有效 s 值）作为判定依据。
- **行无效时输出**：
  - **attn_out**：无效行输出**清零**（写 0）。`RowInvalidUpdateVF` 用 `Select` 将 max == -inf 的行整体写 0（`vf_flashupdate_new.h`），FD 阶段由 `InvalidRows`/`InvalidMaskRows` 清零后写出。
  - **softmax_lse**：无效行输出 **+inf**（3e+99）。代码不依赖数学推导（log(sum) + max 会得 -inf），而是显式处理：`ComputeLseOutputVF` 先算 `lse = log(sum) + max`，再用 `Select` 将 max == -inf 的行置为 `infValue = 3e+99`（`vf_flashupdate_new.h:522-523`）；FD 阶段 `ComputeScaleValue` 同样处理（`vf_flash_decode_arch35.h:310-311`）。

#### 1.8 Sink（learnable sink）

- **Sink**：在 softmax 分母中额外加入一项 `e^{sink - m}`（sink 同时参与 m 的 max 计算），用于提升部分模型（如 Qwen/GLM 系列）的数值稳定性。
- 本算子接口已预留 `sinks` 输入（FLOAT32，shape (Q_N,)），但**当前版本尚未支持**：传入会因 checker 拦截报错（"sinks is currently not supported"），需要等后续版本开放。

#### 1.9 FlashDecode

- **FlashDecode**：decode 阶段 S1 很小（如 1）、S2 很长时的优化——将 KV 方向 split-K 切分到多个 AIV 核并行计算，再跨核归约，避免单核串行扫描全部 KV。
- 分两个阶段：FA 阶段每个 split 独立算局部 softmax max/sum 与部分输出（写入 workspace）；FD 阶段跨 split 归约全局 max/sum、重算 scale、加权求和得最终输出。
- 自动启用（由 metadata 的 `enableFlashDecode` 决定），无需用户配置；`return_softmax_lse` 与 FlashDecode 可同时使用（LSE 从 FD 归约结果直接产出）。

### 2. 代码分层

算子从调用入口到执行引擎由四部分构成：**torch API 层**（用户 Python 入口）、**AICPU 侧**（配套 `flash_attn_metadata` 负载均衡）、**Host 侧**（主算子的接口/定义/tiling）、**Kernel 侧**（AIC+AIV 异构执行），共享 `attention/common/` 公共设施。

#### 2.1 torch API 层（用户入口）

| 文件 | 重点内容 |
|---|---|
| `torch_extension/flash_attn.py` | PyTorch 算子 schema（`flash_attn_metadata` + `flash_attn` 两个接口）、FakeTensor meta 实现（shape/dtype 推导）、`PrivateUse1` 分发的 NPU 实现；内含 metadata 尺寸公式 `((36+72)*B*KV_N+1)*16`（4096B 对齐）与 batch 推导 |
| `torch_extension/graph_convert_flash_attn.py` | torchair 图模式（graph mode）下发转换 |
| `torch_extension/csrc/` | C++ 绑定层，承载 `OpBuilder` 编译与底层调用 |

对外函数原型、参数/返回值说明及调用示例见接口文档 `flash_attn.md`（不在此重复）。

#### 2.2 AICPU 侧（flash_attn_metadata）

| 位置 | 重点内容 |
|---|---|
| `attention/flash_attn_metadata/op_kernel_aicpu/` | AICPU 算子实体：读 cu_seqlens/seqused/mask 等信息，调 SectionStreamK 生成任务切分 metadata |
| `attention/common/op_kernel/load_balance/section_stream_k/` | SectionStreamK 算法库（与 quant_flash_attn 共用）：按 96MB L2 预算切 section、成本模型 `6*ceil(m/16)+10*ceil(s2/64)`、FA 段（按核分配 bN2/gS1/s2 区间）与 FD 段（跨核 split-K 行分配） |
| 输入/输出 | 输入 shape/头数/mask 等参数；输出 metadata 张量（header + FA 段 + FD 段，由主算子 kernel 消费） |

AICPU 侧与 Host 侧 tiling 共享 `fa_adjust_sinner_souter.h` 的档位选择，保证两侧切分口径一致。

#### 2.3 Host 侧（3 层）

**① 接口层（op_api）**

| 文件 | 重点内容 |
|---|---|
| `op_api/aclnn_flash_attn.h` / `op_api/aclnn_flash_attn.cpp` | 对外 `aclnnFlashAttn` 两段式接口（GetWorkspaceSize + 执行），薄壳转发 |
| `op_api/aclnn_flash_attn_inner.h` / `op_api/aclnn_flash_attn_inner.cpp` | inner 实现：对可选输入（sinks/softmax_lse）造占位 tensor |
| `op_api/flash_attn.h` / `op_api/flash_attn.cpp` | L0 API 层（`namespace l0op`）：注册 launcher-list，由框架据此自动生成 inner workspace 计算 |

**② 算子定义与 shape 推导**

| 文件 | 重点内容 |
|---|---|
| `op_host/flash_attn_def.cpp` | 算子定义：9 输入（q/k/v/block_table/cu_seqlens/seqused/sinks/attn_mask/metadata）+ 2 输出（attn_out/softmax_lse）+ 10 属性，含 dtype/格式/默认值与 ascend950 配置 |
| `op_host/flash_attn_infershape.cpp` | 输出 shape 推导：attn_out 按 layout；softmax_lse 按 (B, Q_N, Q_S) 或 TND 的 (Q_T, Q_N)；**不查值域**（留待 tiling 的 checker） |

**③ tiling 层**（重点：入口三层调用）

**入口**：`TilingFlashAttn`（`op_host/flash_attn_tiling.cpp:40-66`，经 `IMPL_OP_OPTILING(FlashAttn).Tiling(TilingFlashAttn)` 注册）。其函数体就是 **parser → check → doTiling 三层调用的骨架**：

```cpp
ge::graphStatus TilingFlashAttn(gert::TilingContext *context)
{
    ...
    FaTilingInfo faInfo;
    FaInfoParser faInfoParser(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {      // 第1层：parser
        return ge::GRAPH_FAILED;
    }
    FAChecker faChecker;
    faChecker.Init(faInfo);
    if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {       // 第2层：check
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);  // 第3层：doTiling
}
```

**第 1 层：parser（参数解析）** — `FaInfoParser::Parse`（`op_host/flash_attn_tiling_info_parser.cpp`）

```
GetOpParaInfo     取输入/属性（q/k/v/block_table/cu_seqlens/seqused/…）
布局解析           决定 BSND/BNSD/TND/PA 与 stride；GraphStatus 失败即报错
GetKvStorageMode   KV 存储模式（连续/PA_BBND/PA_BNBD/PA_NZ）
ParseAxisInfo      GetN1/N2 → GetQueryTSize → GetQkHeadDim/ValueHeadDim
                   → GetBSize → GetS1Size → GetS2Size → GetGSize
ParseFeatureInfo   mask/sinks/lse/maxSeq
产出：faInfo（统一信息结构体），后续 tiling 只读它
```

**第 2 层：check（合法性校验）** — `FAChecker::Process`（`op_host/checkers/fa_checker.cpp:85-108`）

```
对 faInfo 执行 9 组校验器 × 四轮：单参数 → 存在性 → 特性 → 组合
任一校验失败即返回报错，阻断后续 tiling
（base/common/mask/metadata/paged_attention/seq_len/sinks/softmax_lse 各自负责一组约束）
```

**第 3 层：doTiling（切分计算）** — `FiaTilingRegistry::DoTilingImpl`（common）→ `FlashAttnTilingImpl`（arch35）

```
FiaTilingRegistry::DoTilingImpl（common/op_host/fia_tiling_templates_registry.h:94-126）
  按 NpuArch 查表分发 → FlashAttnTilingImpl::DoOpTiling（arch35/flash_attn_tiling.cpp:51-69，DAV_3510，优先级1）
    SetPlatMemoryInfo   平台信息：aiv/aic 核数、CV_RATIO=2、UB/L0/L1 容量、sys workspace
    InitImplParam       cu_seqlens/seqused 存在性判断
    SplitPolicy         win处理 + AdjustSinnerAndSouter + CalcNumBlocks（coreNum）
    FillTiling          ComputeTilingData：tiling data 与 metadata 所需尺寸
    CalcWorkspaceSize   sys区 + D>128额外BMM2暂存 + PA对齐 + FD区（见§10）
    GenTilingKey        UpdateTilingKeyInfo → GET_TPL_TILING_KEY
产出：tiling data / workspace size / tiling key / 核数
```

**产出下发**（运行期，经 TilingContext 交给 kernel）：`SetBlockDim`（核数）、`SetTilingKey`（kernel 变体选择）、`SetWorkspaceSize`、`SetTilingData`（raw 结构）。

三层的职责边界：**parser 只做参数解析**（把 GE/Ascend 参数映射为算子内部 `faInfo`），**check 只做合法性校验**（不产出 tiling 结果，用检测非输入组合），**doTiling 只做切分计算**（信任前两层的结果）。三层先后严格有序、缺一不可——解析完成后、计算前由 check 拦截非法输入。

#### 2.4 Kernel 侧：入口 + 4 层

| 层 | 位置 | 重点内容 |
|---|---|---|
| ① 入口层 | `op_kernel/flash_attn.cpp` | 唯一的 `__global__` 入口：按 tiling key 路由 Dn/Nd（`EnableSoftmaxDn`），并用 `__DAV_C310_CUBE__` 宏区分 AIC/AIV 双编译（`KERNEL_TYPE_MIX_AIC_1_2`），同一份源码编译出 Cube 侧与 Vector 侧两个变体 |
| ② 调度框架层 | `arch35/flash_attn_kernel_dn.h` / `arch35/flash_attn_kernel_nd.h` | 任务级流水框架：`Process()` 按 metadata section 循环 `FlashAttention`（(bN2, gS1, s2) 三重循环 + `CreateTask/ExecuteTask`，PRELOAD_N=2 预取）与 `FlashDecode`（AIV 做跨核归约，两端 `SyncAll`） |
| ③ 计算 block 层（AIC/AIV 分离） | `arch35/flash_attn_block_cube_dn.h` / `arch35/flash_attn_block_cube_nd.h`（AIC 侧）、`flash_attn_block_vec_dn.h` / `flash_attn_block_vec_nd.h`（AIV 侧）、`flash_attn_block_vec_flashdecode.h`（FD 归约） | AIC：BMM1/BMM2（L0A/L0B/L0C 多级 buffer、MTE2→MTE1→M→FIX 四级流水）；AIV：softmax VF（ProcessVec1）与 output 累加（ProcessVec2）、FD 用的跨 split 归约 |
| ④ 公共 API 层 | `attention/common/op_kernel/arch35/`（`flash_attention_score_common_regbase_arch35.h` 等）、`utils/`（`flash_attn_type.h`、`flash_attn_common_def.h`、`attenmask_gs1.h`） | 指令级 VF 算子库（`ProcessVec1VfDn`/`FusedExpSub`/`FlashUpdateNew` 等）与类型/布局/掩码工具；与 flash_attention_score 等算子共享 |

> 更细的 kernel 内部划分见下文各章：kernel 层调度框架（§7）、block cube 层（§8）、block vector 层（§9）、FD block（§10）、common 层（§11）。

#### 2.5 层间关系

```
torch API 层（flash_attn.py，用户入口）
  → AICPU 侧（flash_attn_metadata：SectionStreamK 生成 metadata）
  → Host 侧：接口层（aclnn/L0）
      → def/infershape（定义与 shape）
      → tiling 层（parser → check → doTiling）
          ↓ TilingContext 下发：tiling data / workspace size / tiling key / 核数
      → Kernel 侧：flash_attn.cpp 入口 → flash_attn_kernel_dn/flash_attn_kernel_nd 调度框架 → flash_attn_block_cube_* + flash_attn_block_vec_*
          → attention/common VF 指令库（指令级）
```

关键点：kernel 不感知全局 shape，只按 **metadata 任务区间**执行；编译期由 **tiling key**（模板参数）决定变体，运行期由 **tiling data** 决定切分，两者分别来自 doTiling 的编译清单与切分计算。

### 3. 基本块

FA 计算的核心是把注意力矩阵按块切分、逐块 online 计算。基本块参数由编译期 config 与运行时 shape 共同决定。

**config 映射**（编译期模板参数之一，见 §5 TilingKey）：

| config | sOuter | sInner | D | DV |
|---|---|---|---|---|
| 0 | 64 | 128 | 64 | 64 |
| 1 | 32 | 256 | 64 | 64 |
| 2 | 64 | 128 | 128 | 128 |
| 3 | 32 | 256 | 128 | 128 |
| 4 | 32 | 256 | 256 | 256 |

- **sOuter**：M 方向（Q 序列）每核每次迭代的块行数。
- **sInner**：N 方向（KV 序列）的块大小，即 s2BaseSize。

**运行时基本块参数**：

- **mBaseSize = sOuter × CV_RATIO**（CV_RATIO=2，AIC:AIV=1:2）：一个 AIC 核承担 mBaseSize 行 Q，对应 2 个 AIV 核各处理 mBaseSize/2 行。
- **s2BaseSize = sInner**：KV 序列方向的块大小。
- host 侧 `AdjustSinnerAndSouter` 按 D（及 maxSeq/window 条件）选择 sOuter/sInner，再映射到 config（D=64→config0/1，D=128→config2/3，D=256→config4）。

**gS1 合轴**：

- **含义**：GQA 下 g = Q_N / KV_N 个 Q head 共享同一份 KV。为避免任务循环中分别遍历 g 与 S1 两个维度，将两轴合并为一个 **gS1 轴**：`gS1Size = actSeqLensQ × gSize`（`flash_attn_kernel_dn.h:290`）。
- **排布差异**（合轴内部 g 与 S1 的先后，影响 mask/RowInvalid 的行换算）：
  - **S1G 排布**（BSND/TND 布局）：S1 在外、g 在内，`s1Idx = gS1Idx / gSize`（`flash_attn_kernel_dn.h:473`）。
  - **GS1 排布**（BNSD 布局）：g 在外、S1 在内，`s1Idx = gS1Idx % actSeqLensQ`（`flash_attn_kernel_dn.h:476`）。
- **合轴收益**：bN2 × gS1 二维任务空间按 mBaseSize 统一分块（`gS1LoopTimes = ceil(gS1Size / mBaseSize)`）；共享同一份 KV 的 g 个 Q head 落在相邻行，便于 KV 复用与负载均衡切分。

**任务循环结构**（kernel 内三重循环，详见 §7）：

```
for bN2 in [bN2Start, bN2End):        # batch × KV head 维度
    for gS1 in [gS1Start, gS1End):    # Q 序列 × GQA group 维度
        for s2 in [s2Start, s2End):   # KV 序列维度
            CreateTask / ExecuteTask  # BMM1 + Softmax + BMM2
```

每个 tile 做一次 BMM1(Q×K^T) → Softmax → BMM2(attn×V)，跨 s2 份在线累加 softmax 与输出。

### 4. 负载均衡（SectionStreamK）

**算法位置**：`attention/common/op_kernel/load_balance/section_stream_k/`，由 `flash_attn_metadata`（AICPU）调用，输出 FA段 + FD段 任务切分 metadata。

**SectionStreamK 切分要点**：

- **section 划分**：按 L2 预算（96MB token 成本）逐 batch×head 累加切 section，这样可以确保一个section内的kv数据不会被挤出L2 cache，从而提升L2 cache的命中率，降低KV数据的拷贝时间；`maxGS1Size ≤ mBaseSize` 或单 head 成本 ≤ 预算/AIC数 时不切，因为此时在整个计算过程中，kv数据不需要重复拷贝，也就不存在L2复用。
- **成本模型**（非 token 加权）：块成本 = `6*ceil(m/16) + 10*ceil(s2/64)`。
- **每 section 核数**：`minCore = sqrt(块数)` 下界启发，maxCore = min(36, 块数)。
- **FA 任务分配**：三档贪心 —— 整 batch → 整行（gS1G）→ 行内按 s2 块，逐核按 `成本/剩余核数` 配额分配。
- **FD 任务分配**（跨核行）：行被多个核切分时记 FD 任务，`s2SplitNum` = 切分段数，按 `Σ s2Split×m` 平摊到 ≤72 个 AIV。
- **FD 启用判定**（`CheckChooseWithFd`）：sectionNum>1 必用 FD；否则无 FD 方案收益优于 10 块成本时不用 FD。

**section 内分核（metadata 的 FA/FD 段布局）**：

metadata 为每个 section 独立记录两种任务段，kernel 按核号直接读取自己的区间执行：

```
header（16 字段）：sectionNum | isFd | mBaseSize | s2BaseSize | ...
FA 段  [section][aicIdx(36)][16 字段]：bN2Start/gS1Start/s2Start, bN2End/gS1End/s2End, firstFdDataWorkspaceIdx
FD 段  [section][aivIdx(72)][16 字段]：bN2Idx, mIdx(gS1Idx), workspaceIdx, s2SplitNum, mStart, mLen
总大小 = 16 + sectionNum×(36+72)×16 个 uint32
```

- **FA 段分核**：每个 AIC 核（aicIdx）在 section 内被分到一段 `(bN2, gS1, s2)` 区间，kernel 只执行自己区间内的任务；区间边界由三档贪心分配确定（batch→行→块）。
- **FD 段分核**：仅当该 section 内存在跨核行（同一 gS1 行被多个核切分）时生成，每个 AIV 核（aivIdx）分到一行中的一段 `mStart + mLen`，并按 `s2SplitNum` 对应到 workspace 槽位。
- **section 与核数的关系**：sectionNum 由 L2 预算决定，每个 section 独立分核——同一 AIC 核在不同 section 有不同任务区间，`Process()` 按 section 循环执行（见 §7）。

**metadata 输出**：`16 + sectionNum×(36 AIC + 72 AIV)×16` 个 uint32（header + FA 段 + FD 段），尺寸计算见 `flash_attn_metadata.h:57-128` 与 `flash_attn_metadata_aicpu.cpp:63-68`。

**对 NumBlock（核数）设置的影响**：

- 因为Host侧Tiling在计算NumBlock时，并不知道实际需要使用的核数（运行在AICPU上的metadata接口计算的），所以将NumBlock设置为全部核数。

### 5. TilingKey

**作用**：编译期决定生成哪些 kernel 变体，运行期选择变体。4 个模板参数按 8+8+1+3 位从 bit0 起拼成 tiling key（`flash_attn_template_tiling_key.h`）：

| bits | 参数 | 取值 |
|---|---|---|
| 0-7 | InOutLayoutType | 0=BSND, 1=BNSD, 2=TND, 3=BNSD_BSND |
| 8-15 | KvLayoutType | 0=连续, 1=PA_BBND, 2=PA_BNBD, 3=PA_NZ |
| 16 | HasAttenMask | false/true |
| 17-19 | Config | 0~4（sOuter×sInner×D×DV 组合，见 §3 基本块） |

**生成**：host tiling 的 `GenTilingKey` 一步 —— `UpdateTilingKeyInfo`（按布局/mask/config 填 tilingKeyInfo）→ `GET_TPL_TILING_KEY`（`arch35/flash_attn_tiling.cpp:193-203`）。

**下发与选择**：tiling 期 `SetTilingKey` 写入 TilingContext；运行时 kernel 按 key 选择变体执行。编译期可用 `--tiling_key` 指定只生成部分变体（见 Quick Start 编译）。

**示例解码**：tiling key `2279866368` = 0x87E40000 → InOutLayoutType=0(BSND)、KvLayout=0(连续)、无mask、config=2(D=128, sOuter=64, sInner=128)。

**与 Dn/Nd 的关系**：`useDn = !hasAttenMask && (config==0||config==2)`，见 §6。

### 6. kernel 入口与模板范围

kernel 按 tiling key 的 4 个模板参数编译出不同变体；运行时入口 `op_kernel/flash_attn.cpp` 同时编译 AIC 与 AIV（`KERNEL_TYPE_MIX_AIC_1_2`，`__DAV_C310_CUBE__` 区分 Cube 编译/Vector 编译），每个 AIC 配 2 个 AIV（CV_RATIO=2）。

Dn/Nd 路由（`EnableSoftmaxDn`）：

```cpp
useDn = !hasAttenMask && (config == 0 || config == 2)
```

| 路径 | 条件 | 调度框架 | softmax VF | 适用 |
|---|---|---|---|---|
| Dn | 无mask 且 config∈{0,2} | flash_attn_kernel_dn.h | ProcessVec1VfDn（无mask专用优化） | 无mask、D≤128 |
| Nd | 其余（有mask 或 config∈{1,3,4}） | flash_attn_kernel_nd.h | ProcessVec1Vf（按 actS2 四档通用） | 有mask或 D=256 |

加新模板组合（如新 config、新布局）的改动点：`flash_attn_template_tiling_key.h`（参数声明）→ host `UpdateTilingKeyConfig`（映射）→ `flash_attn.cpp` 的路由与 kernel 模板实例化处。

### 7. kernel 层：调度框架（flash_attn_kernel_dn.h / flash_attn_kernel_nd.h）

**整体流程**（`Process()`）：

```text
Process():
  InitBuffers + InitCrossCoreSync + AllocEventID   # 各核初始化
  for sectionIdx in [0, sectionNum):
      FlashAttention(sectionIdx)                   # AIC+AIV 协同计算（仅 aicIdx < coreNum 的核）
      FlashDecode(sectionIdx)                      # 仅 AIV：两端 SyncAll 包裹
  FreeEventID + UnInitCrossCoreSync
```

**① InitOutput**（`flash_attn_block_vec_dn.h` / `flash_attn_block_vec_nd.h` 的 `ClearOutput`，`needInitOutput=true` 时执行）：计算 attn_out 与 softmax_lse 的总大小（TND 取 T×g×N×DV），由 `2×coreNum` 个 AIV 按 32KB POP buffer 并行写 GM：attn_out 清零（0），LSE 预置 `3e+99`；写完后 `SyncAll` 保证后续计算可见（`flash_attn_block_vec_dn.h:269-294`）。

**② FlashAttention（FA）——核心 while 循环**（`flash_attn_kernel_dn.h:215-268`）：

```text
bN2Cur, gS1Cur, s2Cur ← metadata 段起点；createdTaskCount / executedTaskCount = 0
while (shouldDispatchTask || validTaskCount):
  ├─ shouldDispatchTask = ShouldDispatchTask(...)          # 是否还有任务空间
  ├─ if dispatch:
  │     taskDealMode = GetTaskDealMode(...)                # 判此 (bN2,gS1,s2) 如何处置
  │     ├─ CREATE_TASK → CreateTask() → CalcParams() 填 RunInfoX（bpIdx/n2Idx/gS1Idx/loop/mloop）
  │     │                → createdTaskCount++、validTaskCount++ → UpdateAxisInfo() 推进游标
  │     └─ DEAL_ZERO / 其他（整行跳过 / 快进）→ UpdateAxisInfo() 后 continue
  └─ if validTaskCount > 0:
        ExecuteTask(executedTaskCount)                     # 见下
        if executedTaskCount > PRELOAD_N: validTaskCount--
```

- `GetTaskDealMode`：按 seqused/cu_seqlens 算 `actSeqLensQ/Kv`，行长为 0 时整行跳过（DEAL_ZERO），窗口模式下 s2 游标未到 `curS2Start` 时快进（NOT_START），mask 有效时 `CalcCurS2StartEndWithSparse` 决定跳过无效 tile。
- **while 循环的本质**：创建任务（游标推进）与执行任务（流水消费）分离，`validTaskCount` 追踪未执行任务数，保证创建慢于执行 2 轮（PRELOAD_N）以形成流水。

**③ 核间 PRELOAD 流水（ExecuteTask）**（`flash_attn_kernel_dn.h:424-446`）：

```text
ExecuteTask(loop, taskRunInfo):
  runInfo0   = taskRunInfo[loop & 3]              # 本轮任务
  if isValid: AIC→ComputeMm1（BMM1）｜AIV→ComputeVec1（softmax）
  if loop ≥ PRELOAD_N:
    runInfoNegN = taskRunInfo[(loop-PRELOAD_N) & 3]   # 上 PRELOAD_N=2 轮任务
    if isValid: AIC→ComputeMm2（BMM2）｜AIV→ComputeVec2（output 累加）
                runInfoNegN.isValid = false
```

即同一时刻 AIC 上 BMM1(本轮) 与 BMM2(2 轮前) 并行，AIV 上 Vec1(本轮) 与 Vec2(2 轮前) 并行；数据经核间 flag 接力（BMM1 结果 UB → Vec1 → L1 → BMM2），`PRELOAD_TASK_CACHE_SIZE=4` 用 `loop & 3` 代替取模。

**④ FlashDecode（FD 调度）**（`flash_attn_kernel_dn.h:557-568`）：读 FD 段 metadata（bN2Idx/mIdx/workspaceIdx/s2SplitNum/mStart）→ `vecFdBlock_.InitBuffers()` → `ICachePreLoad` → `SyncAll()`（等 FA 全部完成）→ `vecFdBlock_.FlashDecode()` → `SyncAll()`（等 FD 完成再进下一 section）。

**⑤ FA 的 CV 计算流程（CV 配比 1:2 的数据流向）**：`KERNEL_TYPE_MIX_AIC_1_2` 下 1 个 AIC 配 2 个 AIV，AIC 承担两个 BMM，AIV 承担 softmax/output——双 AIV 通过 flag + `AIV0_AIV1_OFFSET=16` 各自独立同步，行方向各处理 `mBaseSize/2` 行（`actVecMSize`）：

```text
            AIC（Cube）                              AIV0（前半 mBaseSize/2 行）      AIV1（后半 mBaseSize/2 行）
──────────────────────────────────────────────────────────────────────────────────────────────────────
BMM1:      Q×K^T → L0C → Fixpipe → UB[bufId]
           CrossCoreSetFlag(CC_BMM1_0+bufId) ──────────────► CrossCoreWaitFlag 读 UB 前半
           CrossCoreSetFlag(CC_BMM1_0+bufId+16) ─────────────────────────────► CrossCoreWaitFlag 读 UB 后半
           CrossCoreWaitFlag(同 flag×2，等两 AIV 读完) ◄─ CrossCoreSetFlag ×2
                                                     Vec1 softmax → cast → L1(P) ──► AIV 各自
           AIC CrossCoreWaitFlag 等 L1(P) 就绪（CC_L1P，双路）◄─ CrossCoreSetFlag ×2

BMM2:      P×V → L0C → Fixpipe → UB[bufId]
           CrossCoreSetFlag(CC_BMM2_0+bufId) ──────────────► Vec2 前半行累加 → 写 GM
           CrossCoreSetFlag(CC_BMM2_0+bufId+16) ────────────────────────────► Vec2 后半行累加 → 写 GM
```

- **flag 分工**：BMM1/BMM2 各用一对 flag（`CC_BMM1_0/1`、`CC_BMM2_0/1`，UB 双 buffer 轮转），`+16` 偏移让同一 AIC 的两个 AIV 使用独立 flag 位，互不串扰。
- **L1(P) 交接**：Vec1 结果由两个 AIV 各自写 L1(P) 后 CrossCoreSetFlag(CC_L1P)（3 buffer 对应三路 flag），AIC 在 BMM2 前等待两路都就绪。
- **行分担**：两个 AIV 各算 `mBaseSize/2` 行（Dn 按 `align32(actM)>>1`、Nd 按 `(actM+1)>>1` 切分），与 BMM1 fixpipe 的 `dualDstCtl=2` 双目标写、BMM2 fixpipe 的 `dualDstCtl=1` 配合，保证每 AIV 恰好拿到自己那半。


### 8. block cube 层

#### 8.1 buffer 分配
AIC 侧按存储层级分配 buffer，覆盖 BMM1（Q×K^T）与 BMM2（P×V）两级 Matmul 的数据流。flash_attn_block_cube_nd.h的总体分配如下：

| 存储层 | Buffer | 数量×大小 | 用途 |
|---|---|---|---|
| L1 | `l1PBuffers_` | 3×（mBaseSize×s2BaseSize×sizeof(INPUT_T)） | softmax 结果（P 矩阵），供 BMM2 读 |
| L1 | `l1QBuffers_` | 2×（mBaseSize×dBaseSize×sizeof(Q_T)） | Q 矩阵（BMM1 输入），gS1 行内复用 |
| L1 | `l1KvBuffers_` | 4×64KB（Nd D=256：2×128KB 大缓冲） | K/V 矩阵，BMM1 用 K、BMM2 用 V，共享 |
| L0A | `l0aBufferManager_` | 2×32KB（BufferManager 初始 64KB） | BMM1 的 K / BMM2 的 P |
| L0B | `l0bBufferManager_` | 2×32KB（BufferManager 初始 64KB） | BMM1 的 Q / BMM2 的 V |
| L0C | `l0CBuffers_` | 4×64KB | 两个 MM 的 Matmul 结果 |
| UB | `ubMm1ResBuffers_` | 2×（mBaseSize/2×s2BaseSize×sizeof(MM_T)） | BMM1 fixpipe 结果 → AIV softmax |
| UB | `ubMm2ResBuffers_` | 2×（mBaseSize/2×dVBaseSize×sizeof(MM_T)） | BMM2 fixpipe 结果 → AIV output 累加 |

**L1 buffer 分配策略解释**：

- **P 3 个**：P 由 AIV 写（softmax 结果）、AIC 读（BMM2 输入），读写分属不同核。PRELOAD_N=2 下 AIV 写第 N 轮 P 与 AIC 读第 N-2 轮 P 并行，3 buffer 轮转恰好容纳「写入中/待读取/已释放」三个状态，避免核间 flag 等待阻塞流水。
- **Q 2 个**：同一 gS1 行内多个 s2 循环复用同一份 Q，仅在 `isFirstS2Loop` 加载、`isLastS2Loop` 释放；2 buffer 用于前后 gS1 行切换时新旧 Q 交替，MTE2 加载与计算重叠。
- **KV 4×64KB**：K 与 V 分别被 BMM1/BMM2 消费，二者在 PRELOAD_N=2 流水下时间错开——加载 K 的同时可加载下一轮的 V。4 buffer 让 K/V 交替加载保持 MTE2 连续工作；64KB 容量覆盖 mBaseSize×s2BaseSize×2B 单矩阵。Nd D=256 时 s2BaseSize=256，单矩阵超过 64KB，改用 2×128KB 大缓冲。

**L0A/L0B 分配策略解释**：L0A/L0B 是 Matmul 的 A/B 操作数缓存，2×32KB 双 buffer 让 MTE1 搬运（L1→L0）与 PIPE_M 计算重叠——搬运第 i+1 轮数据时第 i 轮正在计算；由 `BuffersPolicyDB`（LOCK_UNLOCK + 外部 eventID）管理锁依赖。

**L0C 分配策略解释**：L0C 是 Matmul 结果的累加器，4×64KB 双倍覆盖「上一轮 Fixpipe 读 + 本轮 Matmul 写」的重叠窗口。两个 MM 共用同一组 L0C（时间错开复用），4 buffer 使 MTE1→M→FIX 三级流水连续推进。

**UB 分配策略解释**：`ubMm1Res/ubMm2ResBuffers_` 是 AIC→AIV 的 CV 通信 buffer（fixpipe 写、AIV 读）。2 buffer 轮转配合核间 flag（`CC_BMM1_0/1`、`CC_BMM2_0/1`）实现双缓冲：AIC fixpipe 写 buf A 时 AIV 正在读 buf B，flag 置位/等待交替，隐藏 CV 传输延迟。大小为 `mBaseSize/2` 行切片（双 AIV 各处理一半，`CV_RATIO=2`）。需要与block_vec上对应UB buffer的地址和大小完全对齐。

### 9. block vector 层

#### 9.1 buffer 分配
**UB buffer 分配（以 Nd 为例，`UbLayout` 结构体定义，`flash_attn_block_vec_nd.h:205-220`）**：Nd 是功能最全的路径（含 mask），其布局覆盖 Dn 全部 buffer 并多出 mask/tmp 两项；按地址从低到高排布，`static_assert(sizeof(UbLayout) <= 248KB)` 限制总占用；UB总size为256KB，但是由于需要为AscendC和VF预留8K，所以业务只能使用248K。

| Buffer | 数量×大小 | 用途 | 分配策略 |
|---|---|---|---|
| `ubMm2ResBuffers_` | 2 × (mBaseSize/2 × dVBaseSize × sizeof(FP32)) ≈ 2×32KB | CV 通信 BUF：AIC BMM2 fixpipe 结果（Vec2 输入） | **双 buffer**：AIC fixpipe 写与 AIV Vec2 读并行，配合 `CC_BMM2_0/1` flag 轮转 |
| `ubMm1ResBuffers_` | 2 × (mBaseSize/2 × s2BaseSize × sizeof(FP32)) ≈ 2×32KB | CV 通信 BUF：AIC BMM1 fixpipe 结果（Vec1 输入） | **双 buffer**：同上，`CC_BMM1_0/1` flag 轮转；大小为 mBaseSize/2 行切片（双 AIV 各处理一半） |
| `ubMaskBuffers_` | 2 × (mBaseSize/2 × s2BaseSize × sizeof(uint8)) | mask 从 GM 拷入的输入 BUF（仅 Nd） | **双 buffer 预取**：当前轮 `AttenMaskCopyIn` 计算的同时 MTE3 预取下一轮 mask，规避 GM 访存延迟。真实需求 = 每个 AIV 的 mBaseSize/2 行 × s2BaseSize 个 INT8（1B），即 2×(mBaseSize/2 × s2BaseSize × 1B)（当前代码写死 2×8KB，仅等于 s2BaseSize=128 时的需求） |
| `ubVec2Res_` | 1 × (mBaseSize/2 × dVBaseSize × sizeof(FP32)) | Vec2 中间结果 + attn_out 输出 buffer | **单 buffer**：与 PRELOAD_N=2 流水配合（Vec2 与 2 轮前 Vec1 同时执行），首次 tile 用 `DataCopy` 分支跳过脏值（见 §7），无需双缓冲。真实需求 = 每个 AIV 的 mBaseSize/2 行 × dVBaseSize × FP32(4B)（当前代码写死 32KB，仅等于 dVBaseSize=128 时的需求） |
| `ubVec1ResBuffers_` | 2 × ((mBaseSize/2 + 1) × dVBaseSize × sizeof(BF16/FP16)) | softmax 结果 cast 后拷至 L1(P) 的过渡 BUF | **双 buffer**：V pipe 写与 MTE3 拷 L1 并行，2 buffer 使两段传输交替。**行大小取 (mBaseSize/2 + 1) 而非 mBaseSize/2**：每行多留 1 个元素作为地址偏移，使相邻行的起始地址错开（不再同余于 UB bank 数），避免多行并发访问命中同一 bank 造成 bank 冲突；×2B 为 cast 回 FP16 的元素大小 |
| `softmaxSumBuf_` / `softmaxMaxBuf_` | 各 3 × (mBaseSize/2 × sizeof(FP32)) | online softmax 的 sum/max 累加状态 | **3 buffer，`mloop % 3`**：每个 gS1 行（mloop）独立一组状态，3 轮转容纳 PRELOAD_N=2 下同时活跃的 mloop（当前行 + 2 行在途），需要注意的是由于VF的处理约束，单个buffer的大小需要256B对齐 |
| `softmaxExpBuf_` | 3 × (mBaseSize/2 × sizeof(FP32)) | exp 中间结果 | **3 buffer，`loop % 3`**：与 sum/max 配合同步轮转，需要注意的是由于VF的处理约束，单个buffer的大小需要256B对齐 |
| `ubLseOutBuffers_` | 2 × (mBaseSize/2 × 32B) | FD 中间结果（sum/max）拷出 GM，或 LSE 结果拷出 | **双 buffer**：MTE3 拷出与 V pipe 计算重叠，`UB_OUT_LSE_OUT_EVENT0/1` 轮转，32B是因为输出时单行的max和sum需要扩展为32B后输出 |
| `softmaxTmpBuf_` | 1 × 512B | softmax 通用 VF 的中间结果缓存（仅 Nd） | **单 buffer 常驻**：VF 调用内临时量，无跨轮依赖，512B 足够 |

**L1 buffer 分配（Nd 同 Dn）**：`l1PBuffers_` 3 × (mBaseSize×s2BaseSize×sizeof(INPUT_T))，softmax 结果（P）写 L1 供 BMM2 读取——**3 buffer 与 PRELOAD_N=2 配合**：AIV 写第 N 轮 P、AIC 读第 N-2 轮 P，中间始终有一个 buffer 处于「写入中/待读取/已释放」之外的缓冲态，避免跨核 flag 等待。需要与block_cube上对应L1 buffer的地址和大小完全对齐。

### 10. FD block：flash_attn_block_vec_flashdecode.h

**数学原理**（跨 split-K 归约的 online softmax 反向展开）：

FA 阶段 KV 方向被切成 K 份（`fdS2SplitNum`），每份独立算出局部统计量：max $m_i$、sum $\ell_i = \sum e^{s-m_i}$、部分输出 $o_i = \sum e^{s-m_i} v$。FD 阶段跨 K 份归约：

$$
m_{global} = \max_i m_i, \qquad
\ell_{global} = \sum_i \ell_i \cdot e^{m_i - m_{global}}, \qquad
o_{final} = \frac{\sum_i o_i \cdot e^{m_i - m_{global}}}{\ell_{global}}
$$

- `ComputeScaleValue`：先 Max 归约得全局 max；再对每份做 `Sub(m_i, m_global) → Exp → Mul(l_i, ·) → Add 累加` 得全局 sum；LSE 输出 `log(sum) + max`，max==-inf（无效行）时置 +inf（`vf_flash_decode_arch35.h:299-315`）。
- `ReduceFinalRes`：用 scale $e^{m_i - m_{global}}$ 重算每份 accumOut 权重，加权累加后再除 $\ell_{global}$（`ReduceFinalRes_VF`）。

**计算流程**：

```text
FlashDecode(fdParams)：
  ├─ fdCoreEnable 判断；fdBalanceMBaseSize=8 把 mLen 拆成 8 行一组的小任务
  ├─ 每小任务：combineTaskPrefixSum = fdWorkspaceIdx（前缀和 → workspace 槽号）
  │     Load：accumOut、lseSum、lseMax 从 workspace 按槽号定位（splitKVIndex×mBaseSize×DV 偏移）
  ├─ ComputeScaleValue（跨 split-K 归约 softmax，见上）
  ├─ ReduceFinalRes（跨 split-K 归约 output，见上）
  └─ DealInvalidRows / DealInvalidMaskRows（清零无效行）→ Cast → Bmm2DataCopyOutTrans 写 GM
```

**UB 分配**（`InitBuffers`，`flash_attn_block_vec_flashdecode.h:175-233`；与 FA block **共享同一块 UB 布局**，按绝对字节偏移手动管理）：

**BASE 定义**（`flash_attn_block_vec_flashdecode.h:179-183`）：

```cpp
constexpr uint32_t mm1Sz = mBaseSize / 2U * s2BaseSize * sizeof(T);   // FA ubMm1Res 单 buffer 大小
constexpr uint32_t mm2Sz = mBaseSize / 2U * dVBaseSize * sizeof(T);   // FA ubMm2Res 单 buffer 大小
constexpr uint32_t BASE  = mm1Sz * 2U + mm2Sz * 2U;                   // FD 业务区起始字节偏移
```

FD 复用 FA 的 UB 布局（同一 AIV），因此必须从 FA block 已占用的 buffer **之后**起址。`BASE = ubMm1Res×2 + ubMm2Res×2` 正是跳过 FA 的 **CV 通信双 buffer 区**（BMM1/BMM2 fixpipe 结果区），FD 业务区从此偏移开始；其后的 `SharedBuffer1/2/3` 区在 FA 阶段由 `attenMaskBuf`/`stage1OutBuf`/`stage2OutBuf` 使用，FA 结束、FD 开始时前者已不再需要，故可分时复用（详见下方并行性说明）。

| Buffer | 数量×大小 | 用途 |
|---|---|---|
| `fdMm2ResBuf1_/2_` | 2 × 16KB | BMM2 部分结果（每 split 的加权输出 o_i）双 buffer，流水预取 |
| `fdReduceBuf_` | 1 × 16KB | 加权累加中间结果（Σ o_i·scale） |
| `fdOutputBuf_` | 1 × 16KB | 最终输出 buffer（供 Cast+拷出 GM） |
| `fdSumBuf1_/2_` | 2 × 6KB | lseSum 双 buffer（每份局部 sum） |
| `fdMaxBuf1_/2_` | 2 × 6KB | lseMax 双 buffer（每份局部 max） |
| `fdLseExpBuf_` | 1 × 6KB | rescale 因子 exp(m_i - m_global) 暂存 |
| `fdLseMaxUbBuf1_/2_` | 2 × 256B | 归约后的行 max 小 buffer（无效行判定） |
| `fdLseUbBuf_` | 1 × 256B | LSE 计算暂存 |

> 注：FD 各 buffer 与 FA 的 `attenMaskBuf`/`stage1OutBuf`/`stage2OutBuf` **复用同一物理 UB 区**（`InitBuffers` 注释：SharedBuffer1/2/3 区 FA 与 FD 分时复用），因此不增加额外 UB 总量。

**FD 与 block_cube 的并行性**（`flash_attn_kernel_dn.h:603-621`）：

```text
Process():       AIC（block_cube）                     AIV（block_vec / FD）
  FA(section)     BMM1/BMM2 fixpipe → UB(mm区)          softmax/累加（读 mm 区）
  FD(section)     ×  不参与                             SyncAll → FD 归约（读 workspace） → SyncAll
  下 section      FA 提前开始（不等 FD）                 FD 完成后进入 FA
```

- **同步划分**：`FlashDecode` 两端各一次 `SyncAll`（全 AIV 同步）——首端保证所有 AIV 的 FA 完成（workspace 已写全），末端保证所有 AIV 的 FD 完成后再进入下一 section；**AIC 不参与 SyncAll**。
- **BV 并行窗口**：AIC 完成本 section 的 BMM2 后无需等待 FD，直接提前开始下一 section 的 FA（`Process()` 中 AIC 分支只执行 `FlashAttention`）；AIC 写下一 section 的 `[0, BASE)` 的 mm 区，FD 读 `[BASE, …)` 的归约区，**两者 UB 区域不重叠**——这正是 BASE 将 FD 业务区与 CV 通信区隔离的目的：FD 执行期间 AIC 可继续搬运/计算下一 section 的 BMM，两引擎不互相阻塞。
- **边界**：AIV 的 FA 下一 section 必须等其自身 FD 完成（SyncAll）才开始，因此 AIC 的提前量受限于下一 section 的 mm1 结果还需 AIV 消费（flag 等待），实际重叠窗口为「AIC 的 BMM 计算/搬运」与「AIV 的 FD 归约」并行。

FD 与 FA 通过 **workspace 槽位**衔接：FA 阶段每个跨核行的部分结果写入 `workspaceIdx` 槽（accumOut + lseSum + lseMax 三段布局），FD 阶段按 `fdWorkspaceIdx` 前缀和读回归约（详见 §10 的 workspace 槽位与 §4 的 section 内分核）。

### 11. common 层：共享设施

common 层位于 `attention/common/op_kernel/`，按功能分四类基础设施：

**① buffer 管理（`buffer.h`、`buffer_manager.h`、`buffers_policy.h`）**：`fa_base_matmul` 命名空间下与 buffer 生命周期/同步策略相关的基础设施：
- `BufferManager`（`buffer_manager.h`）：L0A/L0B/L0C buffer 的分配与管理（初始化、大小划分）。
- `BuffersPolicy*`（`buffers_policy.h`）：buffer 使用策略——`BuffersPolicySingleBuffer`（单 buffer）、`BuffersPolicyDB`（双 buffer，LOCK_UNLOCK/外部 eventID，block_cube 的 L0A/L0B 即用此策略）、`BuffersPolicy3buff`（三 buffer）、`BuffersPolicy4buff`（四 buffer），决定 MTE1 搬运与 Matmul 计算的重叠方式。
- 辅助工具：`buffer.h` 中与 buffer 访问/同步相关的通用函数。

**② Matmul 封装（`matmul.h`）**：`fa_base_matmul` 命名空间的 Matmul 指令封装，供 block_cube 层调用：
- 指令族：`MatmulBase`（K 维固定 128）、`MatmulK`（K 方向 128 分块循环）、`MatmulN`（N 方向 128 分块）、`MatmulFull`（全量一次算完）。
- 配套：`ABLayout`（MK/KN 排布枚举）、`LoadDataToL0A/L0B`、`MakeMMParam`、MM 参数（M/N/K、转置标志），与 `BuffersPolicyDB` 配合完成 L1→L0 搬运与计算。

**③ 数据搬移（`memcopy/`，含 `copy_gm_to_l1.h`、`copy_gm_to_ub.h`、`copy_ub_to_gm.h`）**：GM 与 L1/UB 间的拷贝封装：
- `copy_gm_to_l1.h`：GM→L1 的 `DataCopy` 封装（含 ND→NZ 变换，配合 `offset_calculator_v2.h` 的 GM 坐标定位）。
- `copy_gm_to_ub.h` / `copy_ub_to_gm.h`：GM↔UB 拷贝。
- 配套的 GM/UB 张量描述：`fa_gm_tensor.h`、`fa_l1_tensor.h`、`fa_ub_tensor.h`、`gm_layout.h`、`gm_coord.h`。

**④ VF 指令库（`arch35/` 与 `arch35/vf/`）**：指令级 VF 算子集合，被 block vector 层调用：
- 入口与通用 API：`flash_attention_score_common_regbase_arch35.h`（`FaVectorApi`：`ProcessVec1VfDn`、`FusedExpSub`、`FlashUpdateNew`、`ComputeLseOutputVF`、`RowInvalidUpdateVF` 等）。
- `vf/` 下的 VF 算子族：softmax/update 系列（`vf_basic_block_*`）、flash_decode（`vf_flash_decode_arch35.h`）、antiquant（`vf_antiquant_*`）等，按 D/对齐/更新策略拆分多个变体。
- 与 flash_attention_score 等算子共享，**改 VF 影响所有使用方**。

此外还有：
- **类型/布局工具**（flash_attn 侧 `op_kernel/utils/`）：`flash_attn_type.h`（FAType/布局映射）、`flash_attn_common_def.h`（常量/枚举）、`attenmask_gs1.h`（掩码计算）。
- **负载均衡算法**（`common/op_kernel/load_balance/section_stream_k/`）：host 侧 metadata 生成的算法库（见 §4）。

### 12. 给贡献者的建议

- **从哪读起**：`flash_attn_def.cpp` → `checkers/` → `arch35/flash_attn_tiling.cpp` → `op_kernel/flash_attn.cpp` → `flash_attn_kernel_dn.h`/`flash_attn_kernel_nd.h`。
- **改接口**：def.cpp 增删参数 → 同步 checkers → infershape → 接口文档（torch_extension 侧）。
- **改切分策略**：`fa_adjust_sinner_souter.h`（主算子与 metadata 共用，必须保持一致）+ `common/op_kernel/load_balance/section_stream_k/`。
- **改 kernel 计算**：先在 `flash_attn_template_tiling_key.h` 确认模板范围，AIC 侧重 `flash_attn_block_cube_*`，AIV 侧重 `flash_attn_block_vec_*`；共享 VF API 在 `common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`。
- **新增测试**：tiling 用例加进 `tests/ut/op_host/arch35/test_flash_attn_tiling.csv`；e2e 用例加进 `tests/pytests/test_cases/`。
- **注意事项**：kernel 侧无独立单测（依赖 pytests 真机验证）；tiling CSV 中 TND/BF16 部分正例当前标记 FAIL（按实际结果反填），新增用例时以 `expectResult` 表述期望。

## 参考资源

- 接口文档：`../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md`
- e2e 测试框架：`tests/pytests/readme.md`
- SectionStreamK 算法：`../common/op_kernel/load_balance/section_stream_k/`
- 共享 VF 公共 API：`../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`
