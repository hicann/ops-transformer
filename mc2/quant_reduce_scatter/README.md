# aclnnQuantReduceScatter

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950DT</term>                                                | √       |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        | ×       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | ×       |
| <term>Atlas 200I/500 A2 推理产品</term>                                         | ×       |
| <term>Atlas 推理系列产品</term>                                                  | ×       |
| <term>Atlas 训练系列产品</term>                                                 | ×       |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **算子功能**：实现quant + reduceScatter融合计算。
- **计算公式**：

    $$
    output=Reduce(AllToAllScales * AllToAllData)
    $$

    $$
    AllToAllData=AllToAll(x)
    $$

    $$
    AllToAllScales=AllToAll(scales)
    $$

    其中的Reduce计算是将来自不同rank的数据进行reduce计算。
    $$

- **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <th>连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>x</td>
        <td>输入</td>
        <td>公式中的输入x</td>
        <td><ul><li>不支持空Tensor。</li><li>支持的shape为：(BS, H)或者(B, S, H)。B为batch size，S为sequence length，H为hidden size。当前版本输入x的H支持1024~8192中任意128对齐泛化。</li></ul></td>
        <td>INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scales</td>
        <td>输入</td>
        <td>公式中的输入scales</td>
        <td><ul><li>不支持空Tensor。</li><li>当scales的数据类型为FLOAT8_E8M0时，x的数据类型必须为FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(BS, H)或者(B, S, H)，scales的shape必须对应x的shape为(BS, H/64, 2)或者(B, S, H/64, 2)。</li><li>当scales的数据类型为FLOAT时，x的数据类型必须为INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2，x的shape为(BS, H)或者(B, S, H)，scales的shape必须对应x的shape为(BS, H/128)或者(B, S, H/128)。</li></ul></td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>2-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域标识</td>
        <td>通信域标识</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>输入</td>
        <td>公式中的reduce操作类型。</td>
        <td>当前仅支持"sum"</td>
        <td>string</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>输出</td>
        <td>公式中的输出output。</td>
        <td><ul><li>不支持空Tensor。</li><li>当x的shape是(BS,H)的时候，output的shape必须为(BS/rankNum,H);当x的shape是(B,S,H)的时候，output的shape必须为(B*S/rankNum,H)。rankNum表示通信域大小。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
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
    </tbody>
    </table>

## 约束说明

- 当x的数据类型为FLOAT8_E4M3FN, FLOAT8_E5M2并且scales的数据类型为FLOAT8_E8M0时，输入数据的量化方式为mx量化。
- 当x的数据类型为INT8、HIFLOAT8、FLOAT8_E4M3FN, FLOAT8_E5M2并且scales的数据类型为FLOAT时，输入数据的量化方式为pertoken-pergroup量化(groupSize=128)。
- 只在Ascend950系列平台开启。
- 不支持空tensor输入。
- 通信引擎约束：
  - Ascend950DT: 仅支持UB-Memory通信。
- 通信域大小支持2、4、8。
- 通信域使用约束：同一通信域内仅允许连续执行`aclnnQuantAllReduce`和`aclnnQuantReduceScatter`算子，且该通信域中不允许有其他通信算子。
- `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求满足`HCCL_BUFFSIZE`>= 2 * (`xDataSize` + `scalesDataSize + 1`)。其中`xDataSize`为输入`x`的数据大小，计算公式为：`xDataSize = BS * H * 1 (Byte)`，`scalesDataSize`为`scales`的数据大小，当量化方式为pertoken-pergroup量化时，计算公式为：`scalesDataSize = BS * H / 128 * 4 (Byte)`，当量化方式为mx量化时，计算公式为：`scalesDataSize = BS * H / 32 * 1 (Byte)`。
- H范围仅支持[1024, 8192]，要求128对齐。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_quant_reduce_scatter.cpp](./examples/test_aclnn_quant_reduce_scatter.cpp) | 通过[aclnnQuantReduceScatter](./docs/aclnnQuantReduceScatter.md)接口方式调用quant_reduce_scatter算子。 |
| PyTorch API | - | 通过[quant_reduce_scatter](../../mc2/quant_reduce_scatter/docs/torchapi_quant_reduce_scatter.md)接口调用quant_reduce_scatter算子。 |
