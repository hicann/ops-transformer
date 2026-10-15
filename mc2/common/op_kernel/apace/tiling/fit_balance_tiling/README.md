# fit-balance tiling 引擎——接入指南

> 通信-计算流水线配平切分引擎。
> 本文档面向接入本引擎的算子开发者：接口、示例与建模公式。

---

## 一、对外接口

全部符号为 global namespace，include 这一个头文件即可用：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"
#include "apace/tiling/core_occupancy_first_tiling.h"    // 标定 miss 的兜底: 核利用率优先切分
```

### 主入口

```cpp
CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, const FitBalanceCalibration &calib);
```

单入参 + 标定。内部顺序完成 四件构造（数据流形态 flow / mm 输入事实 mmInput / mm 模型参数 mmModel / 通信模型参数 commModel）→ 切分器构造（MakeCutPlanner）→ 键控求解（Schedule，四件入参）。
求解后与核利用率优先的波量几何候选（`DeriveCommCut`，见同目录 `core_occupancy_first_tiling.h`）
在同一估时器下同台比时，择优返回。
返回"1 短 + N 长"的 M 轴切分方案（或退化的等分/单块）。
通信引擎（AICPU/CCU/AIV_URMA）经 `args.commEngine` 传入——op 侧 `ParseCommEngine(comm_mode)` 解析。

### 分派判定

```cpp
bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration &calib);
```

通信标定表是否命中当前形态（通信引擎 + 卡数二维匹配）。算子用它做切分策略分派——
**命中 → FitBalanceCut；miss → 核利用率优先切分兜底**（同目录公共引擎
`core_occupancy_first_tiling.h`，纯几何、无标定依赖，见接入示例的 miss 分支）；
算子有更合适的策略时也可替换兜底。

```cpp
TilingCommEngine ParseCommEngine(const char *commMode);   // "aiv_urma"→AIV_URMA, "ccu"→CCU, 其余(含"ai_cpu")→AICPU
```

---

## 二、接入示例（新算子视角）

新算子直接使用引擎的 args/cut 类型，无需任何旧类型转换：

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

// ── ① 标定数据: 每算子一份, 算子接入唯一编辑面 ─────────────────────────────────
// profile 按本算子流水线结构填写; rows 填实测拟合值; mmFit 按 dtype 档填
static const FitBalanceCalibration CALIBRATION = {
    /* dataflow: 数据流形态，根据通信原语以及kernel实现决定 */
    {
        CommComputeOrder::COMM_FIRST, // 通信在前: 沿着K轴通信，通信在后则沿N轴
        LocalMMPosition::FRONT,       // 本卡 MM 块提前计算 → 引擎推导通信数据量倍数为 rankDim−1
        true,                         // mValue 为每卡份: 本卡实算聚合后的全量(= mValue × rankDim); false 则 mValue 已是全量(×1)
    },
    /* commRows: 通信拟合行——{通信引擎, 卡数, k 带宽斜率(us/MB), b 每轮启动开销(us), c 一次性固定开销(us)};
     * 一行 = 一个已标定 (引擎, 卡数) 形态; 未标定形态不写行(IsCommCalibrated 返回 false, 走核利用率切分)。
     * 标定假设: 引擎按 dtype 推导通信字节与计算算术档; 算子侧通信量化模式(如 pertile fp8/int8
     * 压缩通信等属性开关)与 dtype 正交——模式固定为单值的算子其字节差会被拟合斜率 k 吸收,
     * 多模式算子需按模式拆分标定行(键无模式维度, 无法同表共存) */
    {
        { {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58}, // 950 标卡 4P 实测(在网行)
          // ↓ 两行新增标定填法演示(数值为演示数——上线前替换为实测值, 未标定则删行留空):
          {TilingCommEngine::AICPU, 8, 22.7301, 33.17, 20.58}, // 新卡档: 同引擎追加一行(8P 实测后照此)
          {TilingCommEngine::AIV_URMA, 4, 3.2, 4.45, 1.8} },   // 新引擎: 引擎维度在行上, 同表加行即可
    },
    /* mmFit: matmul 基本块 + 平台规格; 算力默认零填写 */
    {
        {256, 256, 128},   // baseBlock
        // 算力档默认按公式 总算力/算术字节 自动推导(950: fp16 4096 / fp8 8192 / fp4 16384);
        // 特殊调度形态实测有效算力偏离公式时, 填 computesPerCycleOverride 覆盖——选择依据
        // 不限 dtype, 由算子侧按自己的运行时属性分派(见"多数据流形态算子"节)
        // 平台规格可不填(默认 950: 总算力 8192, 频率 1689.6 cycle/us); 新芯片按规格覆盖
    },
};

// ── ② tiling 主流程: 分派 + 一行调用 ─────────────────────────────────────
ge::graphStatus MyOpTiling::DoTiling()
{
    const auto commEngine = ParseCommEngine(commMode_);   // "ai_cpu"/"ccu" → 枚举

    CommCutResult cut;
    if (IsCommCalibrated(rankDim_, commEngine, CALIBRATION)) {
        // 命中: 构造引擎入参并调用(存量算子也可用 MakeFitBalanceArgs 从旧 TilingArgs 转换)
        FitBalanceTilingArgs args = {mValue_, nValue_, kValue_, rankDim_, aicCoreNum_,
                                     geAType_, inputADtypeSize_, outputDtypeSize_,
                                     ParseCommEngine(commMode_)};
        cut = FitBalanceCut(args, CALIBRATION);
    } else {
        // miss 兜底: 核利用率优先切分(纯几何、通信跟随 matmul 最优分核, 无标定依赖)。
        // SWAT 形态一行完成; 结果经 CutResultFromCof 转为本引擎切分形态, 输出统一
        QuantMatmulTilingData mmTiling;
        CommTilingData dataTiling, scaleTiling;
        GetCoreOccupancyFirstTiling(platformInfo_, mValue_, nValue_, kValue_, isFp4_, hasBias_,
                                    mmTiling, dataTiling, scaleTiling);
        cut = CutResultFromCof(dataTiling);
    }
    // 直接消费 CommCutResult: kernel 侧按 shortTileAtBack 指示的顺序执行 1 短 + N 长
    UseCutResult(cut);
    return ge::GRAPH_SUCCESS;
}
```

接入要素就一个：**填 CALIBRATION**——结构声明（dataflow 三元组）与实测数据（拟合行/算力档）
都在这一处；调用侧构造 `FitBalanceTilingArgs`（存量算子经 `MakeFitBalanceArgs` 转换）后单入参调用。
多形态算子不整份维护 CALIBRATION，按轴拆表、调用处经 `MakeFitBalanceCalibration` 一行组装（见
"多数据流形态算子"节）。

### 建模公式——标定数据参与的计算

**通信模型**（commRows 的 k/b/c）：

```
通信耗时   y = k·x + 轮数×b + c
通信数据量 x = 行数 × 通信轴长 × 每元素字节数（MB）
```

k = 带宽斜率（us/MB），b = 每轮启动开销（us，切几轮付几份），c = 一次性固定开销（us，
不随轮数放大）。通信轴长取 K 还是 N 由计算和通信的前后顺序决定。

**计算模型**（mmFit 的算力档 + 平台规格）：

```
每行耗时  gradient = max(baseN, n) × max(baseK, k) / (computesPerCycle × 频率 × 核数 × util)
块耗时    = 总计算量 × gradient
利用率    util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^0.7)
调和平均  = (m × shares × n) / (m × shares + n)
```

- **总计算量**：块的计算量折合行数（= 块行数 × 倍数）；倍数由输入 M 的口径决定——
  M 为每卡份（perRankM = true）→ ×rankDim，M 已是全量（perRankM = false）→ ×1
  （即下节推导规则的总数据量倍数 `rowMul`）；
- **shares**：通算互相依赖的数据量倍数，推导见下节；
- **指数 0.7 / 0.6**：流水线正常重叠运转的主体计算取 0.7；FRONT/BACK 的本地块单独估时（本地窗口）取 0.6，更保守。

computesPerCycle 即有效算力（MACs/cycle，默认 = 总算力/算术字节，特殊形态覆盖）；频率为 mmFit
平台规格；核数为运行时
入参 aicCoreNum。引擎用这两个模型做流水线配平（长块的通信时长盖住相邻块的计算时长）与切分收益
判定（轮次启动开销 vs 重叠收益）——**标定数据失真会直接反映为切分决策失真**。

**多芯片算子**：标定是普通入参，引擎不含芯片分支——算子侧为每个芯片各备一份 CALIBRATION
（仓内惯例是 tiling 按芯片分目录各持一份），检测平台后选用：

```cpp
static const FitBalanceCalibration CALIB_950 = {...};
static const FitBalanceCalibration CALIB_NEW = {...};    // 频率/L2/有效算力/通信行全按新芯片
// DoTiling 内: PlatformAscendC platform(context_->GetPlatformInfo()); 按其结果选芯片
const FitBalanceCalibration &calib = IsNewChip(platform) ? CALIB_NEW : CALIB_950;
if (!IsCommCalibrated(rankDim_, commEngine, calib)) { /* miss → 核利用率优先兜底(见上) */ }
cut = FitBalanceCut(args, calib);
```


### 数据流形态（CALIBRATION.dataflow 三元组）

算子接入时定死的**结构声明**，引擎据此推导其余全部结构量：

| 字段 | 回答的问题 | 取值含义 |
|---|---|---|
| `order` | 流水线**哪段先行**？ | `COMM_FIRST` 通信在前（先收输入再算）；`COMPUTE_FIRST` 计算在前（先算再发输出） |
| `localMMPosition` | kernel 侧有没有实现"本地计算块的特殊调度"？ | `FRONT` 通信在前时，不参与通信的本地计算块不依赖通信、提前开始计算（kernel 已实现该调度）；`BACK` 通信在后时，先算被通信依赖的块以求尽早开始通信、本地块后置计算（kernel 已实现该调度）；`NONE` kernel 无此特殊实现 |
| `perRankM` | mValue 的口径：**每卡份**还是**本卡实算全量**？ | `true`：mValue 是每卡份——实算 mm 总行数与单次通信字节均为 mValue × rankDim（每卡实算聚合后的全量，通信聚合/分发类算子）；`false`：mValue 已是本卡实算全量，两者 ×1 |

**推导规则**（引擎内部，接入方无需再填任何结构量）：

| 推导量 | 规则 |
|---|---|
| 通信承载侧 `axis` | 通信在前 → K 轴（传 A 侧数据，另一维 = kValue）；通信在后 → N 轴（传 C 侧数据，另一维 = nValue） |
| 总数据量倍数 `rowMul`（= commVolumeMul） | mValue 为每卡份（perRankM = true）→ rankDim；mValue 已是全量（perRankM = false）→ 1 |
| 通算互相依赖的数据量倍数 `shares` | 本地块前置（FRONT）→ rankDim − 1（本地计算块不参与通信，单独按单份建模）；其余 → 同 rowMul |

选错的后果：order/localMM 错 → 配平锚点错位、短块位置错、总时公式形态错（切分整体劣化）；
perRankM 错 → 通信量与 mm 总量估计整体偏移。

判定方法（只看本算子自己的通信方向与 kernel 实现，不考虑其他因素）：
1. **order**：通信的输入是 matmul 的输入 → `COMM_FIRST`；通信的输入是 matmul 的输出 →
   `COMPUTE_FIRST`
2. **localMMPosition**：与通算顺序和 kernel 侧实现绑定——
   - 通信在前，且存在不参与通信的本地计算块可以不依赖通信直接提前开始计算，且 kernel 侧
     实现了这一调度 → `FRONT`
   - 通信在后，存在不被通信依赖的本地计算块，先算被通信依赖的块以求尽早开始通信、把本地
     块后置计算，且 kernel 侧实现了这一调度 → `BACK`
   - kernel 侧没有对本地计算块的特殊调度实现 → `NONE`（即使数据流上存在本地块）
3. **perRankM**：mValue 传进引擎的是每卡份（实算量 = mValue × rankDim）→ `true`；
   mValue 本身就是本卡实算全量 → `false`

### 多数据流形态 / 多算力形态算子（分轴标定 + 运行时组装）

dataflow 三元组按算子接入定死，但 `FitBalanceCut` 的 `calib` 参数是**逐调用传入的普通参数**
（非全局单例）。形态多于一种（运行时属性切换 localMMPosition 等）或标定需运行时拼装的算子，
用 `MakeFitBalanceCalibration` 分轴维护、调用处一行组装：

```cpp
// 标定头: 三轴各一张表——轴间解耦, 每张表只存本轴的独立变体, 组合不在表上放大
constexpr FitBalanceCalibration::DataflowShape DATAFLOWS[] = {
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::NONE,  true},   // 形态0
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, true},   // 形态1
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, true},   // 形态2
};
constexpr FitBalanceCalibration::CommFitRow COMM_ROWS[] = {         // 通信不随轴变: 全局共享
    {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58},
};
constexpr FitBalanceCalibration::MatmulFit MM_FITS[] = {            // 每元素 = 一个形态的 mmFit, 不填的项走默认值
    {{256, 256, 128}, 0.0, 8192.0, 1.65 * BYTES_PER_KB},            // 全显式(常规形态, 算力 0 = 走公式)
    {{128, 128, 128}, 7000.0},                                      // 特殊调度: 基本块/算力不同, 平台规格默认
};

// 存在动态选择 DATAFLOWS 和 MM_FITS 的场景，根据算子自己的实际需要来选择排列组合
const FitBalanceCalibration calib = MakeFitBalanceCalibration(
    DATAFLOWS[dataFlowIdx], COMM_ROWS, MM_FITS[mmFitsIdx]);
if (IsCommCalibrated(rankDim_, commEngine, calib)) {
    cut = FitBalanceCut(args, calib);
}
```
