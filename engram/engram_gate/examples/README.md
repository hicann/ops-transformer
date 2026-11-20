# EngramGate调用示例

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/950DT系列产品</term> | √ |
| <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term> | × |
| <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

`test_aclnn_engram_gate.cpp`展示通过两段式aclnn API调用`EngramGate`的完整流程，包括：

1. 初始化AscendCL、device和stream。
2. 构造`hidden_states`、`kv`和`weight_fused`，申请Device内存并创建ND Tensor。
3. 调用`aclnnEngramGateGetWorkspaceSize`获取workspace大小和executor。
4. 申请workspace，调用`aclnnEngramGate`执行计算。
5. 将`dot`拷贝至Host侧，检查未归一化加权点积的数值。
6. 释放Tensor、Device内存和AscendCL资源。

`test_geir_engram_gate.cpp`展示通过GE IR调用`EngramGate`的完整流程，包括：

1. 初始化GE并创建Session。
2. 创建输入Data节点和EngramGate节点，设置输入、输出描述及属性。
3. 添加并执行计算图。
4. 检查主输出和4个中间结果输出的shape、数据类型及数值。
5. 释放Session和GE资源。

## 参数说明

aclnn示例采用以下配置：

| 参数 | 取值或说明 |
| :--- | :--- |
| 产品 | Ascend 950PR/950DT系列产品 |
| 数据类型 | `hidden_states`、`kv`和`output`为FLOAT16；`weight_fused`和4个中间结果为FLOAT32 |
| 规模 | T=2，H=4，D=64 |
| 输入值 | `hidden_states`、Key、Value和`weight_fused`均为1 |
| 属性 | `eps=1e-6`、`clamp_value=1e-6`、`save_for_backward=true` |
| 输出检查 | 每个`dot`元素应为64 |

GE IR示例分别使用FLOAT16和BFLOAT16，覆盖`save_for_backward`的true/false取值以及以下3种shape：

| T | H | D | 覆盖场景 |
| :---: | :---: | :---: | :--- |
| 2 | 4 | 64 | 最小D规格 |
| 2 | 4 | 5120 | 网络模型规格 |
| 2 | 4 | 32768 | D轴分块计算 |

接口、计算公式和完整参数约束见[aclnnEngramGate](../docs/aclnnEngramGate.md)和[EngramGate](../README.md)。

## 约束说明

- 示例默认使用device 0，使用其他设备时需修改示例中的device编号。
- 运行示例前需安装EngramGate算子包，并加载安装路径下的环境变量。
- aclnn示例检查`dot`的数值；GE IR示例同时检查主输出和4个中间结果输出。

## 调用说明

在ops-transformer仓库根目录执行。使用`--soc=ascend950`。

```bash
bash build.sh --pkg --soc=ascend950 --ops=engram_gate
bash build_out/cann-ops-transformer-*.run --quiet --install-path=<install_path>
source <install_path>/vendors/custom_transformer/bin/set_env.bash
```

执行aclnn示例：

```bash
bash build.sh --run_example engram_gate eager cust \
  --soc=ascend950 --vendor_name=custom --example_name=test_aclnn_engram_gate
```

执行GE IR示例：

```bash
bash build.sh --run_example engram_gate graph cust \
  --soc=ascend950 --vendor_name=custom --example_name=test_geir_engram_gate
```

aclnn示例执行成功后输出：

```text
EngramGate PASS; dot=64, workspace=...
```

GE IR示例执行成功后输出：

```text
EngramGate GE IR: 12/12 PASS
```
