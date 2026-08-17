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
set -euo pipefail

SOC="910b"
ARCH_INFO=$(uname -m)

HAS_PIGZ=0
command -v pigz >/dev/null 2>&1 && HAS_PIGZ=1

NJOBS=${NJOBS:-$(nproc 2>/dev/null || echo 4)}

log_info() {
    echo "=== $1 ==="
}

log_error() {
    echo "Error: $1" >&2
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [--soc <soc_name>] <packages_dir> <output_dir>

Decompress multiple cann-ops-transformer-static tar.gz packages and merge them.
Static library (.a) files will be merged into one.

Options:
  --soc <soc_name>  Specify the SoC name (default: 910b)

Arguments:
  packages_dir  Directory containing *.tar.gz packages
  output_dir    Output directory for merged result

Example:
  $(basename "$0") ./build_out ./merged_output
  $(basename "$0") --soc 910b ./build_out ./merged_output
EOF
    exit 0
}

remove_ascend_lower() {
    local input="$1"
    local lower_input=$(echo "$input" | tr '[:upper:]' '[:lower:]')
    local result=${lower_input#ascend}
    if [[ "$result" == "910_93" ]]; then
        result="A3"
    fi
    echo "$result"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --soc)
                [[ $# -lt 2 ]] && { log_error "--soc requires a value"; usage; }
                SOC=$(remove_ascend_lower "$2")
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            -*)
                log_error "unknown option '$1'"
                usage
                ;;
            *)
                break
                ;;
        esac
    done

    [[ $# -ne 2 ]] && usage

    PACKAGES_DIR="$1"
    OUTPUT_DIR="$2"

    [[ ! -d "$PACKAGES_DIR" ]] && { log_error "packages_dir '$PACKAGES_DIR' does not exist"; exit 0; }

    PACKAGES_DIR=$(realpath "$PACKAGES_DIR")
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
}

extract_packages() {
    TMPDIR="$PACKAGES_DIR/tmp"
    OBJ_WORKDIR="$TMPDIR/.objects"
    rm -rf "$TMPDIR"
    mkdir -p "$OBJ_WORKDIR"
    trap 'rm -rf "${OBJ_WORKDIR:-}"' EXIT

    log_info "Extracting packages"
    shopt -s nullglob
    PKGS=("$PACKAGES_DIR"/*.tar.gz)
    shopt -u nullglob
    PKG_COUNT=${#PKGS[@]}

    [[ $PKG_COUNT -eq 0 ]] && { log_error "no .tar.gz files found in '$PACKAGES_DIR'"; exit 0; }
    echo "  Total packages: $PKG_COUNT"

    local njobs=$NJOBS
    [[ $njobs -gt $PKG_COUNT ]] && njobs=$PKG_COUNT

    if [[ $HAS_PIGZ -eq 1 ]]; then
        printf '%s\0' "${PKGS[@]}" | xargs -0 -r -n 1 -P "$njobs" -I{} \
            tar -x --use-compress-program=pigz -f {} -C "$TMPDIR"
    else
        printf '%s\0' "${PKGS[@]}" | xargs -0 -r -n 1 -P "$njobs" -I{} \
            tar -xzf {} -C "$TMPDIR"
    fi
    echo "  Extracted $PKG_COUNT packages with $njobs parallel jobs (pigz: $HAS_PIGZ)"
}

merge_include() {
    local pkg_dir="$1"
    local base_dir="$2"

    [[ -d "$pkg_dir/include" ]] || return 0
    cp -rn "$pkg_dir/include/"* "$base_dir/include/" 2>/dev/null || true
}

# Merge one package's object files into base_dir.
merge_lib_objects() {
    local pkg_dir="$1"
    local base_dir="$2"
    local workdir="$OBJ_WORKDIR/$(basename "$pkg_dir")"

    [[ -d "$workdir" ]] || return 0

    local pkg_name
    pkg_name="$(basename "$pkg_dir")"; pkg_name="${pkg_name##*static-}"; pkg_name="${pkg_name%%_linux*}"

    local -a new_files=()
    local -a resource_files=()
    local -a overwrite_files=()

    local o_file name dst
    shopt -s nullglob
    for o_file in "$workdir"/*.o; do
        name="${o_file##*/}"
        dst="$base_dir/lib64/$name"
        if [[ ! -f "$dst" ]]; then
            new_files+=("$name")
        elif [[ "$name" == *"op_resource.cpp.o" ]]; then
            resource_files+=("$name")
        else
            overwrite_files+=("$name")
        fi
    done
    shopt -u nullglob

    local merged=0
    local skipped=0

    if [[ ${#new_files[@]} -gt 0 ]]; then
        ( cd "$workdir" && mv -f -t "$base_dir/lib64/" "${new_files[@]}" )
        merged=${#new_files[@]}
    fi

    for name in "${resource_files[@]}"; do
        dst="$base_dir/lib64/$name"
        if [[ $(stat -c%s "$workdir/$name") -gt $(stat -c%s "$dst") ]]; then
            mv -f "$workdir/$name" "$dst"
            merged=$((merged + 1))
        else
            skipped=$((skipped + 1))
        fi
    done

    for name in "${overwrite_files[@]}"; do
        dst="$base_dir/lib64/$name"
        if cmp -s "$workdir/$name" "$dst"; then
            skipped=$((skipped + 1))
        else
            mv -f "$workdir/$name" "$dst"
            merged=$((merged + 1))
        fi
    done

    rm -rf "$workdir"
    echo "    [$pkg_name] merged ${merged} objects, skipped ${skipped}"
}

merge_packages() {
    log_info "Merging packages"

    shopt -s nullglob
    PKG_DIRS=()
    for dir in "$TMPDIR"/cann-ops-transformer-static-*/; do
        [[ -d "$dir" ]] && PKG_DIRS+=("$dir")
    done
    shopt -u nullglob

    if [[ ${#PKG_DIRS[@]} -eq 0 ]]; then
        log_error "no cann-ops-transformer-static-* directories found after extraction"
        exit 0
    fi

    local VERSION
    VERSION=$(grep 'set_cann_package' "$(dirname "$0")/../../version.cmake" | sed 's/.*set_cann_package([^ ]* [^ ]* "\([^"]*\)".*/\1/')
    local base_name="cann-${SOC}-ops-transformer-static-${VERSION}_linux-${ARCH_INFO}"
    local base_dir="$TMPDIR/$base_name"

    rm -rf "$base_dir"
    mkdir -p "$base_dir/lib64" "$base_dir/include"

    # Phase 1: extract object files from every archive in parallel
    local njobs=$NJOBS
    [[ $njobs -gt ${#PKG_DIRS[@]} ]] && njobs=${#PKG_DIRS[@]}
    echo "  Extracting object files from ${#PKG_DIRS[@]} archives with $njobs jobs"
    export OBJ_WORKDIR
    printf '%s\0' "${PKG_DIRS[@]}" | xargs -0 -r -n 1 -P "$njobs" bash -c '
        lib="$1/lib64/libcann_transformer_static.a"
        [[ -f "$lib" ]] || exit 0
        workdir="$OBJ_WORKDIR/$(basename "$1")"
        mkdir -p "$workdir"
        ( cd "$workdir" && ar x "$lib" )
    ' _

    # Phase 2: serial merge into base_dir
    for pkg_dir in "${PKG_DIRS[@]}"; do
        echo "  Merging: $(basename "$pkg_dir")"
        merge_include "$pkg_dir" "$base_dir"
        merge_lib_objects "$pkg_dir" "$base_dir"
    done

    log_info "Removing per-package aggregate headers"
    find "$base_dir/include" -maxdepth 2 -name 'aclnn_ops_transformer_*.h' -delete 2>/dev/null || true

    merge_external_headers "$base_dir"

    generate_merged_header "$base_dir"

    package_result "$base_dir" "$base_name"
}

merge_external_headers() {
    local base_dir="$1"
    local external_dir
    external_dir="$(dirname "$0")/../../common/inc/external"

    [[ -d "$external_dir" ]] || return 0
    log_info "Copying external aclnn_kernels headers"
    cp -rn "$external_dir/"* "$base_dir/include/" 2>/dev/null || true
}

generate_merged_header() {
    local base_dir="$1"
    local include_dir="$base_dir/include"
    local output="$include_dir/aclnn_ops_transformer.h"

    log_info "Generating aclnn_ops_transformer.h"
    {
        echo "#ifndef ACLNN_OPS_TRANSFORMER_H_"
        echo "#define ACLNN_OPS_TRANSFORMER_H_"
        find "$include_dir" -maxdepth 1 \( -name 'aclnn_*.h' -o -name 'acl_*.h' \) \
            ! -name 'aclnn_ops_transformer_*.h' ! -name 'aclnn_ops_transformer.h' \
            -printf '#include "%f"\n' | sort
        echo '#endif // ACLNN_OPS_TRANSFORMER_H_'
    } > "$output"
    mkdir -p "$include_dir/aclnnop"
    cp "$output" "$include_dir/aclnnop/"
}

package_result() {
    local base_dir="$1"
    local base_name="$2"

    pushd "$base_dir/lib64" > /dev/null
    shopt -s nullglob
    local o_files=( *.o )
    shopt -u nullglob
    if [[ ${#o_files[@]} -gt 0 ]]; then
        ar rcs libcann_transformer_static.a "${o_files[@]}"
        rm -f "${o_files[@]}"
    else
        ar rcs libcann_transformer_static.a
    fi
    popd > /dev/null

    pushd "$TMPDIR" > /dev/null
    if [[ $HAS_PIGZ -eq 1 ]]; then
        tar -c --use-compress-program=pigz -f "${base_name}.tar.gz" "$base_name"
    else
        tar -czf "${base_name}.tar.gz" "$base_name"
    fi
    mv -f "${base_name}.tar.gz" "$OUTPUT_DIR"
    popd > /dev/null

    log_info "Done: merged output in $OUTPUT_DIR"
}

main() {
    parse_args "$@"
    extract_packages
    merge_packages
}

main "$@"
