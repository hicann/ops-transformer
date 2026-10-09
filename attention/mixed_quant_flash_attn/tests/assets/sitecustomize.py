#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import sys


# 把 pytests 目录加入 sys.path，让 assets/impl 可直接 import pytest core 模块
# (core.gen_data, core.cpu, core.compare 等)
_PYT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pytests")
_PYT_DIR = os.path.normpath(_PYT_DIR)
if _PYT_DIR not in sys.path:
    sys.path.insert(0, _PYT_DIR)
