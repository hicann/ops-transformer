#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------s.

set -e
echo "#########################打印SN号#########################"
hdc list targets
echo "#########################下载kirin_cann_test代码仓#########################"
git clone https://gitcode.com/funna2000/kirin_cann_test.git
cd kirin_cann_test/harmony-infer-chs
cp ../entry-default-signed.hap ./
bash rdv_kirinx90_gitcode.sh $(hdc list targets) transformer.json KirinX90 ${repo_name} ${MERGE_ID} ${obs_path}
echo "#########################查看用例执行结果#########################"
cat result.txt
echo "#########################打印SN号#########################"
hdc list targets
echo "#########################清空工作目录#########################"
rm -rf ../../../ops-*
