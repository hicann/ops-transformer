# QuantAllReduce

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT系列产品</term>                             |    √    |
| <term>Atlas A3系列产品</term>       |    ×    |
| <term>Atlas A2系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                               |    ×    |
| <term>Atlas训练系列产品</term>                              |    ×    |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的9.2.0版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- 算子功能：QuantAllReduce算子实现低比特量化数据的AllReduce通信与反量化计算的融合：各Rank的量化数据与缩放因子经一步AllGather分发到通信域内全部Rank后，在读取通信数据的过程中逐元素反量化并跨Rank归约求和，每个Rank均得到与输入形状相同的全量通信结果，实现通信与计算的掩盖。

- 计算公式：

  - 输入：
    - $\mathbf{X}_r \in \mathbb{R}^{\text{BS} \times \text{H}}$：Rank $r$ 的量化输入数据，对应入参 `x`。$\text{BS}$ 为token数（`x`为3维时按 $\text{BS}=\text{b}\times\text{s}$ 合轴），$\text{H}$ 为隐藏层维度。
    - $\mathbf{S}_r$：Rank $r$ 的量化缩放因子，对应入参 `scales`，形状与量化模式相关，详见参数说明。
  - 输出：
    - $\mathbf{Y} \in \mathbb{R}^{\text{BS} \times \text{H}}$：每个Rank的最终输出，对应出参 `output`，shape与入参 `x` 相同。
  - 约定：
    - $R$ 表示通信域内rank总数，对应入参 `world_size`，$r \in \{0, 1, \dots, R-1\}$。
    - $⊙$ 表示逐元素乘法，缩放因子按量化分组在 $\text{H}$ 轴广播。
    - $\lfloor z \rfloor$ 表示将 $z$ 向下取整。
    - $\mathrm{DQ}(\cdot)$ 表示按量化模式执行的逐元素反量化操作。
  - 整体计算：

    $$
    \mathbf{Y} = \sum_{q=0}^{R-1} \mathrm{DQ}\left(\mathbf{X}_q \odot \mathbf{S}_q\right)
    $$

    即先对各Rank的量化数据与缩放因子完成AllGather分发（$\text{AllGatherData}=\text{AllGather}(\mathbf{x})$、$\text{AllGatherScales}=\text{AllGather}(\text{scales})$），再逐元素反量化并跨Rank求和，每个Rank均得到完整结果 $\mathbf{Y}$。`reduce_op` 当前仅支持"sum"。

- 计算过程

    当前版本采用MTE一步AllGather（one-shot）方式实现：每个Rank将本卡完整数据一次性写入本地通信窗口，其余Rank直接从各对端窗口读取，通信与反量化归约计算按数据块流水并行。各量化场景的输入类型与反量化分组粒度如下：

    | 量化模式 | x | scales | 分组粒度 | 输出 |
    | --- | :---: | :---: | :---: | :---: |
    | pertoken-pergroup（KG）量化 | INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT32 | 每token每128个连续元素共享1个scale | BFLOAT16、FLOAT16、FLOAT32 |
    | MX量化 | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E8M0 | 每32个连续元素共享1个scale | BFLOAT16、FLOAT16、FLOAT32 |

    <details>
    <summary> pertoken-pergroup（KG）量化场景</summary>

    - **数据分发（一步AllGather通信）**

        每个Rank $r$ 将本卡完整的 $\mathbf{X}_r$ 与 $\mathbf{S}_r$ 一次性写入本地通信窗口（Win区）的数据区，并在状态区写入同步标志位；通信域内其余Rank通过UB-Memory直接读取该窗口数据，无需中转聚合，一步完成AllGather。

    - **反量化与Reduce求和**

        本Rank从各Rank $q$ 的Win区读取数据块。首先做类型转换：INT8 经 FLOAT16 转为 FLOAT32，HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2 直接转为 FLOAT32；缩放因子为 FLOAT32，按每128个连续元素广播一次，得到逐元素缩放视图。随后逐元素反量化并跨Rank累加：

        $$
        \mathbf{D}_q[i,j] = \mathbf{X}_q[i,j] \cdot \mathbf{S}_q\left[i, \left\lfloor j / 128 \right\rfloor\right]
        $$

        $$
        \mathbf{Y}[i,j] = \sum_{q=0}^{R-1} \mathbf{D}_q[i,j]
        $$

        累加在 FLOAT32 下完成，最终按 `output_dtype` 舍入写出，每个Rank得到与 `x` 形状相同的完整结果。
    </details>

    <details>
    <summary> MX量化场景</summary>

    - **数据分发（一步AllGather通信）**

        与KG场景一致，每个Rank将本卡的 $\mathbf{X}_r$（FLOAT8_E4M3FN或FLOAT8_E5M2）与 $\mathbf{S}_r$（FLOAT8_E8M0）一次性写入本地Win区并同步状态标志，其余Rank直接读取。

    - **反量化与Reduce求和**

        本Rank从各Rank $q$ 的Win区读取数据块。$\mathbf{S}_r$ 的shape为 $(\text{BS}, \text{H}/64, 2)$（`x`为3维时为 $(\text{b}, \text{s}, \text{H}/64, 2)$），共 $\text{BS} \times \text{H}/32$ 个缩放因子，即每32个连续元素共享1个 FLOAT8_E8M0 scale。缩放因子先经 BFLOAT16 转为 FLOAT32，再按每32个元素广播为逐元素缩放视图：

        $$
        \mathbf{D}_q[i,j] = \mathbf{X}_q[i,j] \cdot \widetilde{\mathbf{S}}_q\left[i, \left\lfloor j / 32 \right\rfloor\right]
        $$

        其中 $\widetilde{\mathbf{S}}_q$ 表示将上述打包布局按行展开后的逐组缩放因子。随后与KG场景一致地在 FLOAT32 下跨Rank累加，并按 `output_dtype` 舍入写出全量结果。
    </details>

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 140px">
 <col style="width: 120px">
 <col style="width: 500px">
 <col style="width: 220px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>输入</td>
   <td>本卡MTE通信上下文数据，包含通信域地址表与通信窗口信息。aclnn接口直接传入该张量；PyTorch接口通过<code>hcom</code>通信域名由框架自动构建，用户无需显式准备。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>本Rank待通信的量化数据。shape支持2维(bs, h)和3维(b, s, h)，h满足128对齐且范围[1024, 8192]。该参数不支持空tensor。各量化场景支持的数据类型详见约束说明。</td>
   <td>INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>输入</td>
   <td>通信数据的量化缩放因子。pertoken-pergroup（KG）量化模式下数据类型为FLOAT32且维数与x相同：x为2维时shape为(bs, h/128)，x为3维时shape为(b, s, h/128)；MX量化模式下数据类型为FLOAT8_E8M0：x为2维时shape为(bs, h/64, 2)，x为3维时shape为(b, s, h/64, 2)。该参数不支持空tensor。</td>
   <td>FLOAT32、FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>world_size</td>
   <td>属性</td>
   <td>通信域内rank总数，支持2、4、8。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>hccl_buffer_size</td>
   <td>属性</td>
   <td>单个通信域占用的通信缓冲区大小，单位MB。PyTorch接口由框架根据通信域配置自动获取并传入；直接调用aclnn接口时需保证满足约束说明中HCCL_BUFFSIZE的要求。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>reduce_op</td>
   <td>可选属性</td>
   <td>reduce操作类型。默认值为"sum"。当前版本仅支持"sum"。</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>output_dtype</td>
   <td>可选属性</td>
   <td>指定输出参数的数据类型。当前取值为bfloat16、float16、float32，默认值为bfloat16。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>output</td>
   <td>输出</td>
   <td>最终通信结果，shape与输入x相同：x为2维(bs, h)时output为(bs, h)，x为3维(b, s, h)时output为(b, s, h)。数据类型由output_dtype指定，默认bfloat16。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

**说明：** PyTorch接口还提供 `x_dtype`、`scales_dtype` 可选参数，用于在以uint8张量承载的自定义数据类型（如HIFLOAT8、FLOAT8_E8M0）上标注真实的数据类型；`hcom` 参数表示通信域handle名，通过get_hccl_comm_name接口获取。

## 约束说明

- **使用场景约束**：
  - 该接口支持训练场景下使用，当前仅<term>Ascend 950DT系列产品</term>支持。

- **通信引擎约束**：
  - <term>Ascend 950DT系列产品</term>：仅支持UB-Memory通信。

- **通信域约束**：
  - 通信域大小支持2、4、8。
  - 同一通信域内仅允许连续执行`aclnnQuantAllReduce`和`aclnnQuantReduceScatter`算子，且该通信域中不允许有其他通信算子。
  - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求满足`HCCL_BUFFSIZE` >= 2 * CeilDiv(`xDataSize` + `scalesDataSize` + `reserveSize`, 1024 * 1024)，其中CeilDiv为向上取整除法，等式右侧结果单位转换为MB。各项均以字节(Byte)为计算单位：`xDataSize`为输入`x`的数据大小，`xDataSize = BS * H * 1 (Byte)`；`scalesDataSize`为`scales`的数据大小，pertoken-pergroup量化下为`BS * H / 128 * 4 (Byte)`，MX量化下为`BS * H / 32 * 1 (Byte)`；`reserveSize = 1 * 1024 * 1024 (Byte)`，为状态区预留。

- **参数约束**：
  - `x`的BS轴（3维时为B*S合轴）必须能被`world_size`整除。
  - `x`的h轴范围仅支持[1024, 8192]，且要求128对齐。
  - `x`、`scales`均不支持空tensor。
  - `output`的shape与`x`完全一致（2维或3维）。
  - `x`、`scales`、`output`的数据格式仅支持ND。

- **数据类型约束**：

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 256px">
    <col style="width: 384px">
    <col style="width: 128px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>场景</th>
        <th>x</th>
        <th>scales</th>
        <th>output</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>hifloat8、int8、float8_e4m3fn、float8_e5m2</td>
        <td>float32</td>
        <td>float16、bfloat16、float32</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>float8_e4m3fn、float8_e5m2</td>
        <td>float8_e8m0</td>
        <td>float16、bfloat16、float32</td>
    </tr>
    </tbody></table>

- **shape约束**：

    <table style="undefined;table-layout: fixed; width: 1024px"><colgroup>
    <col style="width: 256px">
    <col style="width: 256px">
    <col style="width: 256px">
    </colgroup>
    <thead>
    <tr>
        <th>场景</th>
        <th>x</th>
        <th>scales</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>2维tensor，shape为(bs, h)</td>
        <td>2维tensor，shape为(bs, h/128)</td>
    </tr>
    <tr>
        <td>pertoken-pergroup量化</td>
        <td>3维tensor，shape为(b, s, h)</td>
        <td>3维tensor，shape为(b, s, h/128)</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>2维tensor，shape为(bs, h)</td>
        <td>3维tensor，shape为(bs, h/64, 2)</td>
    </tr>
    <tr>
        <td>MX量化</td>
        <td>3维tensor，shape为(b, s, h)</td>
        <td>4维tensor，shape为(b, s, h/64, 2)</td>
    </tr>
    </tbody></table>

- **确定性计算**：
  - 默认支持确定性计算。

- **预留和非对外参数说明**：
  - 参数表格中的部分参数、部分数据类型暂未对外提供，为预留或内部实现使用。接口参数的介绍及其约束在接口文档[quant_all_reduce算子接口文档](./docs/torchapi_quant_all_reduce.md)中详细说明。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | - | 通过[quant_all_reduce](./docs/torchapi_quant_all_reduce.md) PyTorch接口方式调用quant_all_reduce算子，支持单算子模式与torchair图模式。 |
