# QFA INT8 全量化测试

沿用 `qfa_fp8_test` 的参数集、公共执行器、功能/性能入口和缓存阶段。INT8 使用模式12，
覆盖 BNSD、BSND、TND、mask 0/3、metadata、SWA、LSE 和 D=1～256 的边界场景。

```text
qfa_int8_test/
├── conftest.py
├── common/
│   ├── quant_flash_attn_int8_golden.py  # 共享 tests/assets 的参考实现
│   └── test_runner.py                 # gen/cpu/npu/compare
├── quant_flash_attn_int8_paramset_common.py
├── quant_flash_attn_int8_paramset_func_rdv.py
├── quant_flash_attn_int8_paramset_perf_rdv.py
├── test_quant_flash_attn_int8_func_rdv.py
├── test_quant_flash_attn_int8_perf_rdv.py
└── test_quant_flash_attn_int8_contract.py  # 接口、metadata/Graph 和 TestSpec
```

安装当前算子与 Torch 扩展，并配置 CANN、PyTorch/torch-npu、pytest 和 ATK。默认导入
`cann_ops_transformer.ops`；自定义构建包可通过 `QFA_TEST_OP_PACKAGE=cann_ops_transformer_custom`
指定。设备由 `QFA_TEST_NPU_DEVICE` 指定。

以下命令在本目录执行：

```bash
pytest -q -m ci                                      # 代表性功能用例
pytest -q -m contract                                # 接口、metadata/Graph 和 TestSpec
pytest test_quant_flash_attn_int8_func_rdv.py --golden-mode=gen,cpu --cache-dir=/tmp/qfa_int8_cache
pytest test_quant_flash_attn_int8_func_rdv.py --golden-mode=npu,compare --cache-dir=/tmp/qfa_int8_cache
pytest test_quant_flash_attn_int8_perf_rdv.py --msprof --cache-dir=/tmp/qfa_int8_cache
pytest --parse-prof=/path/to/PROF_xxx --perf-baseline=/path/to/perf_report.log
```

`--golden-mode=all` 为默认值，按 gen → cpu → npu → compare 执行。
各 case 保存 `{name}_input.pt`、`{name}_cpu_output.pt`、`{name}_npu_output.pt`，读取时校验参数。
缓存阶段选项用于功能/性能入口，contract 用例独立执行。
原 regression 目录已删除，原有的尺度、布局、分块/补齐、长序列和空成员代表用例合并到功能参数集，
不再逐个内部优化分支展开形状全排列。LSE 开关一致性、dtype、shape 和非有限值检查由公共执行器统一完成。
性能入口覆盖全部模型头数组合；metadata 在计时前生成，5次预热后执行10次。
长序列检查完整输出是否有限，并对选定查询行做精度比较。
性能报告复用FP8的解析器，筛选 `QuantFlashAttn`，输出预热及正式执行的kernel明细。
该明细统计Attention kernel；完整调用耗时需要包含Torch接口的内部补齐等kernel。

精度标准保持 ATK 双标杆：独立 FP64 golden 和 FP16 数值流程，BF16 配置的
max/avg/RMSE 相对误差比上限为 10/2/2。LSE 对照独立 FP64 logsumexp，
`rtol=5e-4, atol=3e-3`，同时检查全屏蔽行的 `-inf`。

`tests/assets/spec.py` 也支持模式12，复用同一份 INT8 数据与参考实现；
contract 入口用6项组合验证输入生成、精度比较、metadata、有无LSE、NumPy BF16及graph绑定，
避免与功能参数集重复做完整精度矩阵。

Host tiling、输出推导和AICPU UT负责独立的校验及调度逻辑，保留在 `tests/ut` 中。
