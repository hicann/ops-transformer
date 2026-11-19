# Fused recurrent GDN redline

[简体中文](README.md)

ARC-011 uses B=1, T=2048, Nqk=Nv=2, Dk=Dv=128, scale=1, and FP32 initial and final states. The input hook supplies TND tensors and normalizes Q/K before rounding them to BF16. The plugin compares BF16 output and FP32 final state against the independent CPU FP64 recurrence.

Build both artifacts with CANNBot DSL and OpKit 0.8.3 using the same vendor.

```sh
bash build.sh --pkg --soc=ascend950 --ops=fused_recurrent_gdn --vendor_name=gdn_custom
bash build.sh --torch_extension --ops=fused_recurrent_gdn --vendor_name=gdn_custom
```

Install the custom package and the wheel. Source the vendor environment and
add the wheel, TTK checkout and the operator's tests directory to PYTHONPATH.
Set GDN_TESTS to the absolute path of the tests directory. Run in a scratch
directory so that TTK output stays outside the source checkout.

```sh
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python -B -m ttk e2e -i "$GDN_TESTS/ttk/cases/fused_recurrent_gdn_a5_redline.csv" \
  --plugin "$GDN_TESTS/gdn_ttk_ops.py" --backend npu --devices 0 --pc 1 --run 1 \
  --warmup=false --task-prof=false --seed 101 -o "$PWD/results.csv"
```
