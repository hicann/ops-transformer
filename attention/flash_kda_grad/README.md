# FlashKDA Grad — CANNBotDSL 0.8

本目录独立维护 FlashKDA 训练反向算子，使用 **CANNBotDSL 0.8**。
Host 参数检查、工作区准备、编译缓存与 Kernel 主入口继续放在同一个主文件中，
不改写为 PyPTO，也不接入仓库的 ACLNN/CMake 构建。

## 目录

```text
flash_kda_grad/
├── op_kernel/
│   ├── flash_kda_grad.py             # Host 与 Kernel 主入口、共享计算
│   ├── flash_kda_grad_metadata.py    # metadata 生成和分核
│   ├── flash_kda_grad_persistent.py  # persistent 调度
│   ├── flash_kda_grad_staged.py      # staged 调度
│   └── flash_kda_grad_recompute.py   # 重算
└── tests/
    ├── _kdag_paths.py                # 本目录源码加载，不依赖原仓 test 工具
    ├── pytest.ini                   # 仅声明 marker
    ├── ut/test_flash_kda_grad_ut.py  # UT、异常契约、性能/比较 CLI
    └── ttk/
        ├── flash_kda_grad_ttk/       # api、golden、spec
        ├── cases/
        │   ├── flash_kda_grad_stc.csv       # 130 条
        │   └── flash_kda_grad_exception.csv # 60 条
        ├── README.md
        └── run_one.sh
```

算子五个 Python 模块沿用已经验证的 0.8 实现；保存缓存和重算共用公开接口。
`op_kernel` 包含 Host 逻辑，这个目录名并不表示其中只有设备端代码。
测试的 golden、peer、CSV 内容和原生 `cross_check L2` 判据保持不变。

## 环境与导入

使用已激活的 CANN/Conda 环境，需要匹配的 `cannbotdsl==0.8.0`、PyTorch、
`torch_npu`、pytest；TTK 验收还需 ops-test-kit。脚本不切换环境或写入固定服务器路径。

从本仓根目录可直接导入：

```python
from attention.flash_kda_grad.op_kernel.flash_kda_grad import (
    flash_kda_grad_npu,
    flash_kda_grad_from_forward,
)
from attention.flash_kda_grad.op_kernel.flash_kda_grad_metadata import (
    flash_kda_grad_metadata,
)
```

`flash_kda_grad_npu` 接收 q/k/v、gate、beta、上游梯度 do/dht、Aqk/Akk、
反向 metadata 及相应前向缓存。输出顺序为
`dq, dk, dv, dh0, dg, dbeta, dA, dBias`。
`disable_recompute=True` 消费已保存缓存；`False` 在反向内重建可重算缓存。
支持 BNSD、BSND、TND；TND 的 token 总数由 `cu_seqlens` 分成多条序列。

## UT 与异常拦截

以下命令在本仓根目录执行。按 marker 显式选择，默认不会过滤任何用例：

```bash
python -m pytest -q attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -m 'not npu and not ascendc_toolchain and not whitebox'

python -m pytest -q -s attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -k 'exception_csv_host_contract or exception_csv_npu_contract or exception_csv_special_contract'
```

异常 CSV 的变异由上述契约测试执行；普通 TTK E2E 不解释异常 CSV 的 mutation，
不能将普通执行失败视作拦截成功。60 条异常目录对应 117 项契约检查。

独立运行时，三个“真实前向 metadata 兼容”UT 和一个“原仓 Native 打包”UT
会明确 skip。这些检查依赖未包含在本目录中的原仓组件；如需执行，设置
`KDA_DSL_ROOT=/path/to/compatible/cannbot-dsl_8`。其余 UT 和默认 STC 不依赖原仓。

## TTK 精度与性能

脚本从自身位置定位算子和资产，可以从任意目录调用。
如 TTK 尚未安装，可设置 `TTK_REPO=/path/to/ops-test-kit`。
输入/golden 和输出建议放在自己新建的目录，避免写入代码目录：

```bash
export KDA_ASSETS="$PWD/attention/flash_kda_grad/tests/ttk"
export KDA_DEVICES=0
export KDA_MANUAL_DATA=/path/to/input_and_golden
export KDA_OUT=/path/to/results.csv

bash "$KDA_ASSETS/run_one.sh" validate
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109
bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
KDA_PROF=true bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
```

不加 `-t` 默认选择全部 130 条 STC，按 CSV 的 `disable_recompute` 属性分别执行。
`dump` 模式生成新输入/golden，`replay` 读取已有数据；生成数据不代表 NPU 精度通过。
四条替换用例 022/065/066/067 必须使用新定义的数据，不能复用同编号旧 golden。
默认 `cache_source=ref` 由测试参考实现构造前向缓存；可选 `cache_source=fwd`
需要通过 `KDA_DSL_ROOT` 提供兼容的真实前向算子。

详细输入、比较日志、运行模式见 [TTK 说明](tests/ttk/README.md)。
