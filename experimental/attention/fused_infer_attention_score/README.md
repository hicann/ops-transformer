# FA算子

##  概述

本样例的FA算子实现使用自定义算子工程，其kernel包含两个模板：GQA模板以及MLA模板。本工程默认使用GQA模板，示例通过msprof工具（最新版CANN包自带）采集了模板的性能数据。

##  支持的AI处理器

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 目录结构介绍

```
|-- experimental
|   |-- attention                            // 通过aclnn调用的方式调用fa算子
|   |   |-- CMakeLists.txt                   // 算子编译文件
|   |   |-- common                           // 算子公共层文件
|   |   |   |-- CMakeLists.txt
|   |   |   |-- op_host                      // 算子tiling侧公共文件
|   |   |   `-- op_kernel                    // 算子kernel侧公共文件
|   |   `-- fused_infer_attention_score
|   |       |-- CMakeLists.txt
|   |       |-- README.md                    // 本demo使用说明
|   |       |-- op_host                      // FA算子tiling侧文件
|   |       |-- op_kernel                    // FA算子kernel侧入口文件
|   |       |-- run.sh                       // 一键式运行脚本
|   |       `-- tests                        // 测试文件

```

## 功能说明

- 算子功能：
本样例算子实现的是FA算子，不支持mask，其数学表达式为：

  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">FusedInferAttentionScore</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">layout</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">query</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  <tr><td align="center">key</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  <tr><td align="center">value</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">atten_out</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  </table>

不支持任何高阶特性（mask、pse、pa、fd、actual_seq等）不支持入图。本工程支持的case情况如下：

1. qk headdim = 128， rope = 0， v headdim = 128.
2. qk headdim = 64 rope = 0， v headdim = 64.
3. qk headdim = 128， rope = 64， v headdim = 128.
4. qk headdim = 512， rope = 64， kvn = 1, g = 1, 2, 4, 8, 16, 32, 64, 128.

## 环境变量配置

根据当前环境，安装对应的CANN开发套件包（toolkit包+ops包）。

1. TorchNPU安装包下载路径：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. CANN包可从[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann)获取，具体下载安装见昇腾社区[文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Ubuntu)。

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

## 编译运行

**注意：首先确认自己的机器是哪种npu型号，工程默认使用910b(A2)机器，如果是910c(A3)的机器,则需要修改编译命令，将编译命令中的-soc=ascend910b改成-soc=ascend910_93**

提供两种方式运行本Demo

- 一键式安装运行(run.sh脚本默认使用A2命令编译，若在A3上运行需要自行修改脚本内的编译命令)

```bash
# 切换到fused_infer_attention_score目录
cd ${git_clone_path}/experimental/attention/fused_infer_attention_score/
# 编译+执行aclnn接口+采集性能数据
bash run.sh
# 切换aclnn用例性能数据目录(在/experimental/attention/fused_infer_attention_score目录下会生成output目录，里面存放了性能数据)
cd ${git_clone_path}/experimental/attention/fused_infer_attention_score/output
```

- 分离式安装运行

1、编译与安装自定义算子包

```bash
# 切换到工程根目录(ops-transformer目录)
cd ${git_clone_path}
# 编译样例算子run包
bash build.sh --pkg  --experimental --soc=ascend910b --ops=fused_infer_attention_score
#安装自定义算子run包
./build_out/cann-ops-transformer-${vendor_name}-${arch}_linux.run --instal-path=${install_path} # install_path为CANN包安装路径
# source custom包
source ${install_path}/vendors/custom_transformer/bin/set_env.bash
```

2、pytest测试精度

```bash
# 切换到pytest目录
cd ${git_clone_path}/experimental/attention/fused_infer_attention_score/tests/pytest
# 执行pytest
python3 -m pytest -rA -s test.py -v -m ci
```

3、pytest测试性能

```bash
# 切换到pytest目录
cd ${git_clone_path}/experimental/attention/fused_infer_attention_score/tests/pytest
# 使用msprof测试性能
msprof_path=${install_path}/cann/tools/msopt/bin/msopprof # install_path为CANN包安装路径
$msprof_path --output=output --aic-metrics=Roofline,Occupancy,Default python3 -m pytest -rA -s ./tests/pytest/test.py -v -m ci
```

注意:
pytest使用详见[pytest框架使用说明](./tests/pytest/README.md)
run.sh中提供的性能收集命令只支持pytest框架单次使用单个用例测试，因此，在使用run.sh脚本时确保testcases.py文件中只有一个测试case被选中，算子执行时间会在屏幕上打印出来（Task Duration）。

本demo提供了preload流水优化对比：通过修改**attention/common/op_kernel/arch22/fia_kernel_nonquant.h** 和 **/common/op_kernel/arch22/fia_kernel_nonquant_mla.h** 中的**PRELOAD_NUM_ACTUAL**变量，然后再次运行run.sh即可得到修改后的性能数据。该变量值为0时表示无preload，该变量值为2时表示开启2轮preload流水优化。
下表为preload开启与关闭的性能对比

<table>
  <tr>
    <td align="center">fa算子类型</td>
    <td align="center">输入shape</td>
    <td align="center">layout</td>
    <td align="center">dtype</td>
    <td align="center">无preload算子执行时间</td>
    <td align="center">preload2轮算子执行时间</td>
  </tr>

  <tr>
    <td align="center">GQA</td>
    <td align="center">q:(24,64,128,128)<br>k:(24，32，512，128)<br>v:(24，32，512，128)</td>
    <td align="center">bnsd</td>
    <td align="center">float16</td>
    <td align="center">646.76us</td>
    <td align="center">377.20us</td>
  </tr>

  <tr>
    <td align="center">MLA</td>
    <td align="center">q:(24,128,64,128)<br>k:(24，64，512，128)<br>v:(24，64，512，128)<br>rope: 64</td>
    <td align="center">bnsd</td>
    <td align="center">float16</td>
    <td align="center">828.30us</td>
    <td align="center">583.24us</td>
  </tr>
</table>
