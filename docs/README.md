# 文档中心

## 目录结构

Docs目录结构说明如下：

```text
├── zh
  ├── context                            # 公共文档，如术语、基础概念等
  ├── debug                              # 算子调试指导文档
  │   ├── op_debug_prof.md
  │   ├── ...
  ├── develop                            # 算子开发指导文档
  │   ├── aicore_develop_guide.md
  │   ├── graph_develop_guide.md
  │   ├── ...
  ├── figures                            # 图片目录
  ├── install                            # 环境安装和编译指导文档
  │   ├── build.md                       # build参数说明
  │   ├── compile.md                     # 源码构建指南(离线编译、自定义算子包编译、transformer包编译等)
  │   ├── quick_install.md               # 环境部署、CANN包安装等
  │   ├── dir_structure.md               # 项目目录层级介绍
  │   └── ...
  ├── invocation                         # 算子调用指导文档（包括aclnn调用、图模式调用等）
  │   ├── quick_op_invocation.md
  │   ├── ...
  ├── menu_aclnn_api.md                  # 全量aclnn接口索引文件
  ├── menu_torch_api.md                  # 全量torch_extension接口索引文件
  ├── op_api_list.md                     # aclnn接口列表
  ├── op_list.md                         # 全量算子列表
  └── torch_api_list.md                  # torch_extension接口列表
├── CONTRIBUTING_DOCS.md                 # 文档贡献说明
├── QUICKSTART.md                        # 快速入门
└── README.md
```

## 进阶教程

### 指南类文档

|   文档                   |   说明                 |
| --------------------------- | -------------------- |
| [源码构建指南](zh/install/compile.md)                        | 介绍联网/未联网场景下不同的源码构建方式和验证方法。          |
| [算子调用指南](zh/invocation/quick_op_invocation.md)         | 介绍调用算子样例方法和不同算子调用方式（如PyTorch/aclnn/图等）。 |
| [标准算子开发指南](zh/develop/aicore_develop_guide.md)       | 介绍如何基于标准工程定义算子原型、实现Tiling和Kernel，此类算子称为“标准算子”。<br>标准算子支持aclnn和图模式调用。 |
| [简易算子开发指南](../examples/fast_kernel_launch_example/README.md) | 介绍如何基于简易工程实现fast_kernel_launch，即`<<<>>>`方式，此类算子称为“简易算子”。<br>简易算子仅支持PyTorch调用。 |
| [算子调试调优](zh/debug/op_debug_prof.md)                    | 介绍常见的算子功能调试和性能调优方法（如数据采集和仿真流水等）。 |

### API类文档

| 文档        | 说明                  |
| ----------------------- | ---------------------- |
| [算子列表](zh/op_list.md)                        | 介绍项目内置标准算子清单。 |
| [aclnn API列表](zh/op_api_list.md)               | 介绍项目提供的aclnn API清单，通过将所有算子封装为以`aclnn`为前缀的C语言API，旨在方便用户在Host侧直接调用。 |
| [PyTorch API列表](zh/torch_api_list.md) | 介绍项目提供的torch_extension API清单，通过JIT编译桥接PyTorch与aclnn API，并通过GE Converter支持TorchAir图模式。 |

### 工具类文档

| 文档        | 说明                  |
| ----------------------- | ---------------------- |
| [Simulator仿真工具](zh/debug/npu_sim.md) | 面向算子开发场景的SoC级仿真工具，用于分析运行在AI仿真器上AI任务在各阶段精度和性能数据。 |

### 样例实践

如需学习算子领域高性能实战案例，可参考[cann-samples Performance目录](https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/README.md)。

|  算子分类       |  样例算子       | 说明                  |
| ----------------------- | ---------------------- | ---------------------- |
|  attention  |   full_quant_fused_infer_attention_score_story    |  围绕FIA（Fused Infer Attention Score）算子提供per-block全量化实现示例，包含输入数据生成、算子执行与结果校验流程。|
|  mc2        | moe_dispatch_and_combine_story | 围绕moe dispatch/combine通信算子给出性能优化实践，包含构建运行命令、测试数据生成与精度校验流程。 |
|  moe        | moe_init_routing_story | 介绍MoeInitRoutingV3算子的完整性能优化实践。包括多核并行、内存带宽优化、核内流水线排布、SIMT编程、硬件特性适配等优化策略，从理论分析到代码实践的端到端调优指南。|

### 更多文档

欢迎前往[wiki中心](https://gitcode.com/cann/ops-transformer/wiki/Home.md)了解更多项目信息，包括项目定位、算子开发补充知识介绍、性能优化方法和实践样例、常见问题（FAQ）及问题定位方法等。

## 附录

| 文档                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [算子基本概念](zh/context/basic_concept.md) | 介绍算子领域相关的基础概念和术语，如量化/稀疏、数据类型、数据格式等。 |
| [build参数说明](zh/install/build.md)   | 介绍本项目build.sh参数功能和取值，包括源码编译、算子调用、调试等。 |
