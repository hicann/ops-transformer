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

import fileinput
import sys


def replace_paths_in_test_file(test_file_path, path):
    try:
        with fileinput.FileInput(test_file_path, inplace=True, backup=".bak") as file:
            for line in file:
                print(line.replace("__PATH__", path), end="")
        print(f"Replaced __PATH__ in {test_file_path}.")
    except Exception as err:
        print(f"Failed to replace path in {test_file_path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    replace_paths_in_test_file(sys.argv[1], sys.argv[2])
