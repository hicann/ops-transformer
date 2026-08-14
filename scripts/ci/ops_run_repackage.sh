#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

TOP_DIR=""
PKG_PATH=""
RUN_PKG_SAVE_PATH=""
OS_ARCH=$(uname -m)
PKG_NAME=""
SOC=""
OPP_PREFIX="opp"
PKG_TYPE=""

# -----------------------------
# 函数定义
# -----------------------------

parse_args() {
    local arg

    # 循环遍历所有命令行参数
    for arg in "$@"; do

        # 使用 case 语句匹配参数格式
        case "$arg" in
            # 匹配 --soc=... 格式
            --soc=*)
                # 截取等号后面的值，赋值给全局变量 SOC
                SOC="${arg#*=}"
                ;;
            # 匹配 --top-dir=... 格式
            --top_dir=*)
                TOP_DIR="${arg#*=}"
                ;;
            # 匹配 --pkg-path=... 格式
            --pkg_path=*)
                PKG_PATH="${arg#*=}"
                ;;
            --run_pkg_save_path=*)
                RUN_PKG_SAVE_PATH="${arg#*=}"
                ;;
            --pkg_name=*)
                PKG_NAME="${arg#*=}"
                ;;
            --pkg-type=*)
                PKG_TYPE="${arg#*=}"
                ;;
            # 处理未知参数
            *)
                echo "ERROR: Unknown param $arg" >&2
                ;;
        esac
    done
}


# 打印日志
log() {
    echo "$*"
}

# 错误退出
die() {
    log "ERROR: $*"
    exit 1
}

# 确保目录存在
ensure_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log "Creating directory: $dir"
        mkdir -p "$dir" || die "Failed to create directory $dir"
    fi
}

# 确保文件存在
ensure_file() {
    local file="$1"
    [[ -f "$file" ]] || die "Required file not found: $file"
}

# -----------------------------
# 主流程开始
# -----------------------------

log "Starting ops packaging process..."
parse_args "$@"

WORKDIR=${TOP_DIR}/${PKG_PATH}         # 当前工作目录
TEMP_RUN_DIR="./run_files"              # 临时存放拷贝的 .run 文件
HOST_RUN_NAME="host.run"
HOST_EXTRACT_DIR="host"
MAKESELF_TARGET_DIR="build/makeself"
RUNFILE_TARGET_DIR="build/_CPack_Packages/makeself_staging"
PACKAGE_SCRIPT="$TOP_DIR/open_source/cann-cmake/scripts/package/package.py"
MERGE_SCRIPT="$TOP_DIR/open_source/cann-cmake/scripts/package/merge_binary_info_config.py"
PKG_OUTPUT_DIR="build/_CPack_Packages/makeself_staging"
RUN_PACKAGE_SAVE_AB_PATH=${TOP_DIR}/${RUN_PKG_SAVE_PATH}
ARCHIVE_RUN_DIR="${TOP_DIR}/vendor/hisi/build/delivery/${SOC}/"
MERGE_OPS_SCRIPT="$TOP_DIR/open_source/cann-cmake/scripts/package/json_merger.py"


cd ${WORKDIR}/ || exit

# 1. 创建临时目录存放 .run 文件
ensure_dir "$TEMP_RUN_DIR"
cd "$TEMP_RUN_DIR"
log "Working in temporary directory: $(pwd)"

# 2. 直接从源目录查找 kernel run 文件
shopt -s globstar nullglob
kernel_run_files=("$RUN_PACKAGE_SAVE_AB_PATH"/**/cann-*custom_operator_group*.run)
shopt -u globstar nullglob

# 检查是否有 kernel run 文件
[[ ${#kernel_run_files[@]} -gt 0 ]] || \
    die "No .run files found in $RUN_PACKAGE_SAVE_AB_PATH"

# 拷贝makeself目录至build目录下，不存在build目录则创建
ensure_dir "../$MAKESELF_TARGET_DIR"
cp -rf "$TOP_DIR"/open_source/makeself/* "../$MAKESELF_TARGET_DIR" || die "Failed to copy $MAKESELF_TARGET_DIR"

# 3. 拷贝 host/cann.run 为 host.run, 通常生成的原始host包名中没有custom
cd $RUN_PACKAGE_SAVE_AB_PATH
host_file_name=$(find . -type f -name "*.run" | grep -v "custom")
host_file_name="${host_file_name#./}"
cd "${WORKDIR}"
cd "${TEMP_RUN_DIR}"
host_run_src="$RUN_PACKAGE_SAVE_AB_PATH/$host_file_name"
if [ -e $host_run_src ]; then

    cp -v $host_run_src "./$HOST_RUN_NAME" || die "Failed to copy host run file"
    log "Copied host run file: $host_run_src -> $HOST_RUN_NAME"
else
    log "Host run file not found: $host_run_src"
fi

# 4. 解压 host.run
if [[ ! -x "./$HOST_RUN_NAME" ]]; then
    chmod +x "./$HOST_RUN_NAME" || die "Cannot make $HOST_RUN_NAME executable"
fi

log "Extracting $HOST_RUN_NAME to ./$HOST_EXTRACT_DIR"
"./$HOST_RUN_NAME" --extract="$HOST_EXTRACT_DIR" --noexec || \
    die "Failed to extract $HOST_RUN_NAME"

# 移除 host.run
rm -f "./$HOST_RUN_NAME"
log "Removed $HOST_RUN_NAME after extraction"

ensure_dir "./$HOST_EXTRACT_DIR"

# 5. 并行解压+处理所有 kernel run 文件
PARALLEL_EXTRACT_NUM=${PARALLEL_EXTRACT_NUM:-16}
PROCESS_FIFO="/tmp/process_fifo_$$"
PROCESS_ERROR_FILE="/tmp/process_errors_$$"
> "$PROCESS_ERROR_FILE"

TARGET_KERNEL_DIR="$HOST_EXTRACT_DIR/${OPP_PREFIX}/built-in/op_impl/ai_core/tbe/kernel/$SOC/$PKG_NAME/"
TARGET_CONFIG_DIR="$HOST_EXTRACT_DIR/${OPP_PREFIX}/built-in/op_impl/ai_core/tbe/kernel/config/$SOC/ops_transformer"
ensure_dir "$TARGET_KERNEL_DIR"
ensure_dir "$TARGET_CONFIG_DIR"

if [[ ${#kernel_run_files[@]} -gt 0 ]]; then
    [ -e "$PROCESS_FIFO" ] || mkfifo "$PROCESS_FIFO"
    exec 8<>"$PROCESS_FIFO"
    rm -f "$PROCESS_FIFO"
    for ((i=1;i<=PARALLEL_EXTRACT_NUM;i++)); do echo >&8; done

    for runfile in "${kernel_run_files[@]}"; do
        read -u8
        {
            if [[ ! -x "$runfile" ]]; then
                chmod +x "$runfile"
            fi

            # 提取唯一标识：父目录名_文件序号
            parent_dir=$(basename "$(dirname "$runfile")")
            runfile_basename=$(basename "$runfile" .run)
            base_name="kernel_${parent_dir}_${runfile_basename}"
            extract_dir="./$base_name"

            log "Extracting and processing $runfile"
            "$runfile" --extract="$extract_dir" --noexec || {
                echo "$runfile: extract failed" >> "$PROCESS_ERROR_FILE"
                echo >&8
                exit 1
            }

            full_path=$(find "${extract_dir}/packages/vendors" -maxdepth 1 -type d -name "custom_*_transformer" 2>/dev/null | head -n 1)

            if [[ -z "$full_path" ]]; then
                echo "$runfile: kernel_dir not found" >> "$PROCESS_ERROR_FILE"
                echo >&8
                exit 1
            fi

            kernel_dir_name=$(basename "$full_path")
            kernel_src_dir="$extract_dir/packages/vendors/$kernel_dir_name/op_impl/ai_core/tbe/kernel/${SOC}"
            config_src_dir="$extract_dir/packages/vendors/$kernel_dir_name/op_impl/ai_core/tbe/kernel/config/${SOC}"

            [[ -d "$kernel_src_dir" && -d "$config_src_dir" ]] || { echo >&8; exit 0; }

            for json_file in "$config_src_dir"/*.json; do
                [[ -f "$json_file" ]] || continue
                json_name=$(basename "$json_file" .json)
                cp "$json_file" "$TARGET_CONFIG_DIR/${json_name}_${base_name}.json"
            done

            rsync -au "$kernel_src_dir/" "$TARGET_KERNEL_DIR/" 2>> "$PROCESS_ERROR_FILE"

            echo >&8
            exit 0
        }&
    done

    wait
    exec 8>&-

    if [[ -s "$PROCESS_ERROR_FILE" ]]; then
        log "Errors during parallel processing:"
        cat "$PROCESS_ERROR_FILE"
        rm -f "$PROCESS_ERROR_FILE"
        die "Parallel processing failed"
    fi
    rm -f "$PROCESS_ERROR_FILE"
fi
# 6. 批量合并JSON文件
merge_all_jsons() {
    local binary_script="$1" ops_script="$2"
    declare -A groups

    log "Merging JSON files..."

    for f in "$TARGET_CONFIG_DIR"/*_*.json; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f" .json)
        base="${name%_kernel_*}"
        [[ "$base" == "$name" ]] && continue
        groups[$base]+="$f "
    done

    for base in "${!groups[@]}"; do
        read -ra files <<< "${groups[$base]}"
        n=${#files[@]}
        [[ $n -eq 1 ]] && { mv "${files[0]}" "${TARGET_CONFIG_DIR}/${base}.json"; continue; }

        script="$ops_script"; priority="last"
        [[ "$base" =~ ^(binary_info_config|relocatable_kernel_info_config)$ ]] && { script="$binary_script"; priority="first"; }

        log "  $base: $n files"
        python3 "$script" --input-files "${files[@]}" --output-file "${TARGET_CONFIG_DIR}/${base}.json" --priority "$priority"
        rm -f "${files[@]}"
    done
}

merge_all_jsons "$MERGE_SCRIPT" "$MERGE_OPS_SCRIPT"

# 7. 清理解压目录
for runfile in "${kernel_run_files[@]}"; do
    [[ -f "$runfile" ]] || continue
    parent_dir=$(basename "$(dirname "$runfile")")
    runfile_basename=$(basename "$runfile" .run)
    base_name="kernel_${parent_dir}_${runfile_basename}"
    extract_dir="./$base_name"
    if [[ -d "$extract_dir" ]]; then
        rm -rf "$extract_dir"
    fi
done

filelist_src_path="$HOST_EXTRACT_DIR/share/info/ops_transformer/script/filelist.csv"
rm -f "$filelist_src_path"

# 8. 拷贝 host/ 到 makeself 目录
ensure_dir "../$RUNFILE_TARGET_DIR"
cp -rf "$HOST_EXTRACT_DIR"/* "../$RUNFILE_TARGET_DIR"/ || \
    die "Failed to copy host content to makeself directory"

log "Host content copied to $RUNFILE_TARGET_DIR"

# 9. 执行打包脚本
cd "../" || echo "Failed to go back to workdir"

# 执行 package.py
log "Executing package.py to generate final package..."
python3 "$PACKAGE_SCRIPT" \
    --pkg_name "$PKG_NAME" \
    --makeself_dir "${WORKDIR}/$MAKESELF_TARGET_DIR" \
    --pkg-output-dir "${WORKDIR}/$PKG_OUTPUT_DIR" \
    --independent_pkg \
    --chip_name "$SOC" \
    --os_arch linux-"$OS_ARCH"\
    --delivery_dir "${WORKDIR}/build" \
    --source_dir "${WORKDIR}"

log "Packaging completed successfully!"

# 9.5 生成 RPM/DEB 包
if [[ -n "$PKG_TYPE" && "$PKG_TYPE" != "run" ]]; then
    bash "${TOP_DIR}/vendor/hisi/build/scripts/ops_generate_rpm_deb.sh" \
        --top_dir="$TOP_DIR" \
        --pkg_path="$PKG_PATH" \
        --pkg_name="$PKG_NAME" \
        --soc="$SOC" \
        --pkg-type="$PKG_TYPE"
fi

# 10. 归档全量构建算子编译包至hdfs目录
ensure_dir "$ARCHIVE_RUN_DIR"
cp -r "$PKG_OUTPUT_DIR"/*.run $ARCHIVE_RUN_DIR
