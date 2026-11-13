#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# all_gather_matmul_v3 (ubmem) 测试运行包装 —— 交付版，环境无关，无需修改本文件。
#
# 前置(调用前自行完成):
#   1) conda activate <装了 cann_ops_transformer wheel 的环境>   (python3/torch/torch_npu 就绪)
#   2) source <CANN_HOME>/set_env.sh                             (已安装 custom_transformer 算子包)
#
# 用法:
#   bash run_test.sh --mode precision --cases all --world-size 4          # 精度(18 个默认形状)
#   bash run_test.sh --mode perf --cases all --world-size 4               # 性能+掩盖率
#   bash run_test.sh --mode precision --m 48 --k 3584 --n 2816            # 自定义形状
#   bash run_test.sh --list-cases                                          # 查看内置用例
# 可选环境变量: ASCEND_RT_VISIBLE_DEVICES(默认 0,1,2,3, 须覆盖 world-size 个设备)
# -----------------------------------------------------------------------------------------------------------
set -o pipefail

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] 未检测到 ASCEND_HOME_PATH, 请先 source CANN 的 set_env.sh"
    exit 1
fi

USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
SITE_PKG=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
PREPEND=""
if [ -n "${USER_SITE:-}" ] && [ -d "$USER_SITE" ]; then
    PREPEND="$USER_SITE"
fi
if [ -n "$SITE_PKG" ]; then
    if [ -n "$PREPEND" ]; then PREPEND="$PREPEND:$SITE_PKG"; else PREPEND="$SITE_PKG"; fi
fi
if [ -n "$PREPEND" ]; then
    if [ -n "${PYTHONPATH:-}" ]; then
        export PYTHONPATH="$PREPEND:$PYTHONPATH"
    else
        export PYTHONPATH="$PREPEND"
    fi
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}

# 精度模式不设该变量; 性能模式 golden 小算子走 CCU 编排(对融合算子无影响)
if [[ "$*" == *"--mode perf"* ]]; then
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
else
    unset HCCL_OP_EXPANSION_MODE
fi
unset HCCL_BUFFSIZE

cd "$(dirname "$0")"
echo "[ENV] python=$(which python3)"
echo "[ENV] cann_ops_transformer=$(python3 -c 'import cann_ops_transformer as c; print(c.__file__)' 2>&1 | tail -1)"
echo "[ENV] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
mkdir -p logs
python3 all_gather_matmul_v3_test.py "$@" 2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"
exit "${PIPESTATUS[0]}"
