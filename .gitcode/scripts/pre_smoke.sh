#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set +e

echo "start run test case, please wait ..."
cd ${WORKSPACE}
arm_package="cann-ops-transformer-custom_linux-aarch64.run"
rm -f ${arm_package}
wget -nv -O ${arm_package} https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${arm_package} || true
if [ ! -f "${arm_package}" ] || [ ! -s "${arm_package}" ];then
    echo "No custom package found, This PR no need excute smoke."
    rm -f ${arm_package}
    exit 0
fi
log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=("is_finite")
echo $ops

# ==============================
# 运行测试主循环
# ==============================
log "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

for op in "${ops[@]}"; do
  echo "Processing: $op"
  mode="eager"
  [ "$op" = "crop_and_resize" ] && mode="graph"
  source /usr/local/Ascend/cann/set_env.sh
  arm_package="cann-ops-transformer-custom_linux-aarch64.run"
  wget -nv -O ${arm_package} https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${arm_package} 2>/dev/null
  chmod 755 ${arm_package}
  unset LD_LIBRARY_PATH
  unset ASCEND_HOME_PATH
  bash ${arm_package}  2>&1 | tee -a ./run_test.log
  source /usr/local/Ascend/cann/set_env.sh  &&  bash build.sh -f ./pr_filelist.txt 2>&1 | tee -a ./run_test.log
done

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
# if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${obs_path}/${slog_name}"; then
#   echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/slog.tar.gz"
# fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -w -e "Example failed" "./run_test.log"; then
    echo "$date_time : run test case failed"
    exit 1
else
  echo "$date_time : run test case success"
fi
