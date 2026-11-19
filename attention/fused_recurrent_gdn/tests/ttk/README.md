# Fused recurrent GDN 红线用例

[English](README_en.md)

ARC-011 用例使用 B=1、T=2048、Nqk=Nv=2、Dk=Dv=128、scale=1，以及 FP32 初态和终态。输入钩子提供 TND 张量，对 Q/K 归一化后转换为 BF16。插件使用独立 CPU FP64 递推比较 BF16 输出和 FP32 终态。

使用 CANNBot DSL 和 OpKit 0.8.3，并为两个构建产物指定相同 vendor。

```sh
bash build.sh --pkg --soc=ascend950 --ops=fused_recurrent_gdn --vendor_name=gdn_custom
bash build.sh --torch_extension --ops=fused_recurrent_gdn --vendor_name=gdn_custom
```

安装 custom 包和 wheel，加载 vendor 环境，将 wheel 安装目录、TTK 源码目录和算子的 tests 目录加入 PYTHONPATH。将 GDN_TESTS 设置为 tests 目录的绝对路径，在临时目录运行，验证产物保存到源码仓库外。

```sh
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python -B -m ttk e2e -i "$GDN_TESTS/ttk/cases/fused_recurrent_gdn_a5_redline.csv" \
  --plugin "$GDN_TESTS/gdn_ttk_ops.py" --backend npu --devices 0 --pc 1 --run 1 \
  --warmup=false --task-prof=false --seed 101 -o "$PWD/results.csv"
```
