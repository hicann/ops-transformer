# Batch一致性

## 简介

在应用开发过程中，部分算子为了追求较高的性能，针对同一个Token在不同批次大小或者同一批次不同位置场景中，可能存在计算结果偏差。
当前，针对部分算子在满足相同的运行环境条件下，可以通过配置计算过程采用Batch一致性算法，来使得无论如何组合输入，计算结果保持完全一致。
**batch一致性算法**：对于给定的一个Token，无论该Token在批次内所处的位置、批次大小、或是和哪些其他Token一同被批处理，输出结果必须逐比特完全一致。

## 注意事项

- 通常建议不开启Batch一致性计算，因为同一个算子开启Batch一致性计算后相关算子存在一定的性能劣化，因此模型的单次运行性能可能会下降。但在实验、调试和回归测试等需要保证多次运行结果相同来定位问题和实验算法的场景，Batch一致性计算可以提升效率。

- 当前配置为进程级开关配置。

- **版本约束**：TorchNPU版本大于等于26.2.0，CANN版本大于等于9.2.0。

## 使用方法

目前CANN算子的主流调用方式为aclnn API或PyTorch API（torch_extension）。部分算子API默认Batch一致性实现，部分算子API默认非Batch一致性实现。对于非Batch一致性实现的算子，部分可通过手动配置开启Batch一致性。

- **调用aclnn API**

  该场景下，通过[《Runtime运行时API》](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)中“运行时配置>aclrtSetSysParamOpt”接口（进程级）配置Batch一致性。具体通过设置`ACL_OPT_DETERMINISTIC=3`开启Batch一致性计算。

- **调用PyTorch API**

  该场景下，通过[《TorchNPU自定义API》](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/overview.md)中`torch_npu.npu.set_deterministic_level`接口开启Batch一致性计算。

对于不同框架的算子API，其默认的Batch一致性计算实现策略请以具体的aclnn API文档或PyTorch API文档描述为准。
