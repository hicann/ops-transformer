#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build both QMLA operators and optionally run the NPU reference/comparison.
# Usage: bash attention/quant_flash_mla_with_kvcache/build_and_run.sh [build|run|all|profile] [golden arguments...]
set -eo pipefail
op_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$op_dir/../.." && pwd)
action=${1:-all}
if (($#)); then shift; fi
case "$action" in build|run|all|profile) ;; *) echo "Expected build, run, all, or profile" >&2; exit 2;; esac
conda_base=${CONDA_EXE:-/root/miniconda3/bin/conda}
conda_base=$(dirname "$(dirname "$conda_base")")
source "$conda_base/etc/profile.d/conda.sh"
conda activate qmla
cann_dir=${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.1.0}
source "$cann_dir/set_env.sh"
export MAX_JOBS=10 CMAKE_BUILD_PARALLEL_LEVEL=10 MAKEFLAGS=-j10
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
# Explicit opt-in because this container may not permit a cgroup hard limit.
if [[ ${QMLA_ALLOW_RSS_WATCHDOG:-0} != 1 ]]; then
    echo 'Set QMLA_ALLOW_RSS_WATCHDOG=1 to accept the 18 GiB RSS watchdog (not a 20 GiB hard limit).' >&2
    exit 2
fi
if [[ ${QMLA_RESOURCE_WRAPPED:-0} != 1 ]]; then
    export QMLA_RESOURCE_WRAPPED=1
    exec python "$op_dir/scripts/limit_resources.py" bash "$0" "$action" "$@"
fi
cd "$repo_dir"
ops=quant_flash_mla_with_kvcache,quant_flash_mla_with_kvcache_metadata
install_dir="$repo_dir/build_out/qmla_install"
log_dir="$repo_dir/build_out/qmla_logs"
mkdir -p "$log_dir" "$install_dir"
if [[ "$action" == build || "$action" == all ]]; then
    bash build.sh --pkg --soc=ascend950 --ops="$ops" --vendor_name=qmla -j10 2>&1 | tee "$log_dir/build.log"
    mapfile -t packages < <(find "$repo_dir/build_out" -maxdepth 1 -name '*qmla*.run' -type f 2>/dev/null)
    if ((${#packages[@]} != 1)); then
        echo "Expected one generated .run package, found ${#packages[@]}" >&2
        exit 1
    fi
    bash "${packages[0]}" --quiet --install-path="$install_dir" 2>&1 | tee "$log_dir/install.log"
    bash build.sh --torch_extension --incremental --soc=ascend950 --ops="$ops" --vendor_name=qmla -j10 2>&1 | tee "$log_dir/wheel.log"
    mapfile -t wheels < <(find "$repo_dir/torch_extension/dist" -maxdepth 1 -name '*qmla*.whl' -type f)
    if ((${#wheels[@]} != 1)); then
        echo "Expected one QMLA wheel, found ${#wheels[@]}" >&2
        exit 1
    fi
    python -m pip install --no-deps --force-reinstall "${wheels[0]}"
fi
export ASCEND_CUSTOM_OPP_PATH="$install_dir/vendors/qmla_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$install_dir/vendors/qmla_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$repo_dir/build_out/qmla_torch_extensions"
if [[ "$action" == profile ]]; then
    python "$op_dir/scripts/profile_golden.py"
elif [[ "$action" != build ]]; then
    python "$op_dir/tests/pytest/common/qmla_with_kvcache_golden.py" --golden-device=npu "$@" 2>&1 | tee "$log_dir/golden.log"
fi
