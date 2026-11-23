# MoeFinalizeRoutingV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                      |     √    |
| <term>Atlas A3系列产品</term>      |    √    |
| <term>Atlas A2系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    ×     |
| <term>Kirin X90处理器系列产品</term> | √ |
| <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- **算子功能**：MoE(Mixture of Experts)计算中，最后处理合并MoE(Mixture of Experts)FFN(Feed-Forward Network)的输出结果。
- **计算公式**：

  输入为 FLOAT16、BFLOAT16、FLOAT32 时：

  $$
  expertid=expertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

输入为 `FLOAT4_E2M1` 时，普通专家使用以下公式反量化，其中 r 为 expandedX 的物理行，j 为逻辑列，E2M1 和 E8M0 表示按对应格式解码。

  $$
  \mathrm{dequant}(r,j)=\mathrm{E2M1}(\mathrm{expandedX}[r,j])\cdot
  \mathrm{E8M0}(\mathrm{expandedXScaleOptional}[r,\lfloor j/64\rfloor,\lfloor(j\bmod 64)/32\rfloor])
  $$

输入为 `FLOAT4_E2M1` 时，ACLNN V5 使用 `expandedXScaleOptional` 对 `expandedX` 反量化。传入 `scalesOptional` 时，每路在叠加可选 bias 后乘以对应的路由权重。路由贡献和共享专家输出使用 FP32 累加，结果转换为 BF16。

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>expandedX</td>
        <td>输入</td>
        <td>公式中的`expandedX`，MoE(Mixture of Experts)的FFN(Feed-Forward Network)输出。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT4_E2M1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>输入</td>
        <td>公式中的`expandedRowIdx`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x1Optional</td>
        <td>输入</td>
        <td>公式中的`x1`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32；输入为 FLOAT4_E2M1 时仅支持 BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2Optional</td>
        <td>输入</td>
        <td>公式中的`x2Optional`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32；输入为 FLOAT4_E2M1 时仅支持 BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>公式中的`bias`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32；输入为 FLOAT4_E2M1 时仅支持 BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>公式中的`scales`，表示可选的路由权重。当 expandedX 输入为 FLOAT4_E2M1 时，仅支持 FLOAT32。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOptional</td>
        <td>输入</td>
        <td>公式中的`expertIdx`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedXScaleOptional</td>
        <td>输入</td>
        <td>expandedX 的块反量化系数。输入为 FLOAT4_E2M1 时必传。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>属性</td>
        <td>表示是否支持丢弃模式，并说明expandedRowIdx的排列方式。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>k</td>
        <td>属性</td>
        <td>表示每个token选出的top-K专家个数。expandedX为非量化类型且scalesOptional不为空时，若k>1则校验k必须与scalesOptional的shape第二维一致；scalesOptional为空时，直接使用k推导输出shape。默认值为1。 输入为 FLOAT4_E2M1 时，使用 k 和索引长度推导 NUM_ROWS，不从反量化系数的 shape 推导 K。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的`out`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32；输入为 FLOAT4_E2M1 时仅支持 BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>: 不支持BFLOAT16。

## 约束说明

- ACLNN 接口中，`FLOAT4_E2M1` 输入仅支持 V5。
- 输入为 `FLOAT4_E2M1` 时，H 为逻辑宽度，必须为正偶数；R 为实际物理行数。dropPadMode 为 0/2 时 expandedX 为 `[R,H]`，为 1/3 时为 `[E,C,H]`，R=E*C。expandedXScaleOptional 的 shape 为 `[R,ceil(H/64),2]`，与 expandedX 的物理行对齐，每个反量化系数对应 32 个元素。
- 输入为 `FLOAT4_E2M1` 时，expandedRowIdx 为 `[NUM_ROWS*K]`，k 必须大于 0，索引长度必须能被 k 整除；NUM_ROWS 由索引长度除以 k 得到，k 不从 scale 形状推导。x1/x2/bias 仅支持 BF16；x2 依赖 x1，bias 依赖 expertIdx。不支持输出再量化。
- MXFP4 输入当前不支持零计算专家、拷贝专家和常量专家。x、alpha1、alpha2、v 和三个特殊专家区间必须传空。
- PyTorch API 中，`expanded_x` 使用 `torch.float4_e2m1fn_x2` 或存储 E2M1 编码的 `torch.uint8`，最后一维为打包宽度 H/2。`expanded_x_scale` 使用 `torch.float8_e8m0fnu` 或存储 E8M0 编码的 `torch.uint8`。执行及 Meta/FakeTensor 推导结果均为 BF16。
- 详细类型、布局及接口限制见 [ACLNN V5 文档](docs/aclnnMoeFinalizeRoutingV5.md)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_finalize_routing_v2.cpp](examples/test_aclnn_moe_finalize_routing_v2.cpp) | 通过[aclnnMoeFinalizeRoutingV2](docs/aclnnMoeFinalizeRoutingV2.md)接口调用MoeFinalizeRoutingV2算子。 |
| aclnn V4 接口 | - | 通过 [aclnnMoeFinalizeRoutingV4](docs/aclnnMoeFinalizeRoutingV4.md) 接口调用。 |
| aclnn V5 接口 | - | 通过 [aclnnMoeFinalizeRoutingV5](docs/aclnnMoeFinalizeRoutingV5.md) 接口调用。 |
| PyTorch API | - | 通过[moe_finalize_routing](../../moe/moe_finalize_routing_v2/docs/torchapi_moe_finalize_routing.md)接口调用moe_finalize_routing算子。 |
