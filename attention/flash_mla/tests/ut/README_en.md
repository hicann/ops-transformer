# FlashMla Host UT

English | [简体中文](./README.md)

Load the CANN SDK and run the command in the repository root directory. No NPU is required:

```bash
bash build.sh --ophost_test --ops=flash_mla --soc=ascend950 -j25
```

The CSVs in `op_host/arch35/` cover the positive and negative tiling cases; the CSVs in `op_host/` cover shape/dtype inference.
