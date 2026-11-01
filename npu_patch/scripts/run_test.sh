#!/usr/bin/env bash
# MoonEP test runner (npu_patch version, migrated from ascend-moonep).
#
# Usage (the script locates itself; run from the repo root):
#   bash npu_patch/scripts/run_test.sh                       # mte/ubmem engine (default)
#   bash npu_patch/scripts/run_test.sh --use_urma            # urma engine (A5/Ascend950 only)
#   bash npu_patch/scripts/run_test.sh --soc_type Ascend910_93  # A3
#   bash npu_patch/scripts/run_test.sh --prod                # op tests use the production 8-card shape
#   BUILD=1 bash npu_patch/scripts/run_test.sh               # rebuild _C before running tests
#   NPROC=4 RANK_N=4 bash npu_patch/scripts/run_test.sh      # set card count
#   MOONEP_GROUP_SIZE=2 bash npu_patch/scripts/run_test.sh    # subgroup size for grouped e2e
#
# Environment variables:
#   SOC_TYPE                    {Ascend910B|Ascend910_93|Ascend950}, default Ascend950.
#   NPROC / RANK_N              ranks per node for torchrun / ranks for script-style tests;
#                               default is the visible card count parsed from
#                               ASCEND_RT_VISIBLE_DEVICES, falling back to 8 when unset.
#   PROD=1                      Same as --prod: pass --prod-shape to the op tests
#                               (R=8, S=4096, K=16, H=7168, E=896). Requires RANK_N=8.
#   BUILD=1                     Rebuild the _C extension before tests (required after csrc/kernels changes).
#   MOONEP_GROUP_SIZE           Cards per subgroup for grouped e2e, default 2; when NPROC is
#                               below 2x this value the case is auto-skipped (not a failure).
#   ASCEND_RT_VISIBLE_DEVICES   Visible card list, supports "4,5" and "4-7"; **need not start
#                               at 0** (npu_patch._device maps it to visible indices 0..N-1).
#
# Three test groups:
#   1) buffer tests (test_npu/buffer_test): accuracy/e2e via the Buffer public API.
#      - pytest-style files use pytest -k to select the parametrized urma / mte
#        variant per --use_urma;
#      - script-style C / cpython interface tests (test_buffer_c_api /
#        test_buffer_cpython_api) have no engine switch (interface tests; keep
#        the in-code defaults) and run as-is;
#      - grouped e2e (test_buffer_grouped_e2e) has no parametrize id and **must
#        not use -k** (an empty -k selection makes pytest exit 5 -> false FAIL);
#        it runs in its own section without -k.
#   2) urma op tests (test_npu/urma): run only with --use_urma, A5 only.
#      The dedup domain is covered explicitly: dispatch / joint / combine run with
#      --dedup-mode rank and (when RANK_N % 8 == 0) server; the server dispatch
#      case uses the real planner (--plan-source kernel --output-mode both); an
#      extra small-shape --combine-engine udma case covers the rank-level UDMA
#      combine kernel. test_e2e_urma.py is parametrized over rank|server.
#   3) mte ubmem op tests (test_npu/mte): run by default (without --use_urma).
#   --prod (or PROD=1) adds --prod-shape to every op test (production 8-card
#   shape: S=4096, K=16, H=7168, E=896); it requires RANK_N=8.
#
# Notes:
#   - shmem bootstrap is auto-selected by kernel_test_utils.init_shmem: single
#     node (torchrun one node WORLD_SIZE==LOCAL_WORLD_SIZE, or scripts without
#     --nproc-per-node) or no master addr/port -> ShmemRuntime.init_with_buffer
#     (UID); multi-node with master specified -> ShmemRuntime.init_with_ip_port.
#     The pytest runs here (single-node torchrun) and the scripts (single-node
#     python mp.spawn) both take the single-node buffer path.
#   - Launch styles (matching each test_* header comment):
#       * pytest-style (test_e2e / test_buffer_* / test_prefetch* / test_grad_reduce*)
#         -> torchrun --nproc-per-node=N -m pytest -s <file> [-k urma|mte];
#       * script-style C/cpython interface (test_buffer_c_api / cpython_api)
#         -> torchrun --nproc_per_node=N --master_port=<P> <file> (no __main__ mp.spawn);
#       * mte/urma op scripts (test_planning / test_moonep_* / test_dispatch_combine_*)
#         -> python <file> [--mode kernel --R ...] (the script mp.spawns R ranks internally).
#   - After changing csrc/ or kernels/ you must rebuild the _C extension:
#     BUILD=1 ... or `bash npu_patch/scripts/build_base.sh -soc_type <SOC_TYPE>`
#     (thin wrapper: `build_a5.sh`). Python/test changes take effect immediately.
#   - Commands run one by one with PASS/FAIL aggregated; one failure does not
#     stop the rest (full picture in one pass). Note pytest exits 5 when -k
#     selects nothing, which also counts as FAIL; check the parametrize ids.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Args / environment
# ---------------------------------------------------------------------------
USE_URMA=0
PROD="${PROD:-0}"
SOC_TYPE="${SOC_TYPE:-Ascend950}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_urma) USE_URMA=1; shift ;;
        --prod) PROD=1; shift ;;
        --soc_type|-soc_type)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --soc_type requires a value, for example: --soc_type Ascend950"
                exit 1
            fi
            SOC_TYPE="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash npu_patch/scripts/run_test.sh [--use_urma] [--prod] [--soc_type {Ascend910B|Ascend910_93|Ascend950}]"
            exit 1 ;;
    esac
done

case "${SOC_TYPE}" in
    Ascend910B|Ascend910_93|Ascend950) ;;
    *)
        echo "[ERROR] Unsupported SOC_TYPE: ${SOC_TYPE}"
        echo "Supported values: Ascend910B, Ascend910_93, Ascend950"
        exit 1 ;;
esac

# urma/udma ops are A5(SIMT)-only: dispatch_udma uses __simt_vf__ +
# simt_api/device_sync_functions.h; A2/A3 do not define ASCEND_MOONEP_HAS_SIMT.
if [ "$USE_URMA" = 1 ] && [ "$SOC_TYPE" != "Ascend950" ]; then
    echo "[ERROR] --use_urma supports only Ascend950(A5): udma ops require SIMT,"
    echo "        which A2/A3 (${SOC_TYPE}) lacks; do not run urma on A3."
    exit 1
fi

# Derive the visible card count: parse ASCEND_RT_VISIBLE_DEVICES when NPROC/RANK_N are unset.
_count_visible_devices() {
    local v="${ASCEND_RT_VISIBLE_DEVICES:-}"
    if [ -z "$v" ]; then
        echo 8
        return
    fi
    local n=0 part a b
    local IFS=','
    for part in ${v}; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        if [[ "$part" == *-* ]]; then
            a="${part%%-*}"; b="${part##*-}"
            n=$(( n + b - a + 1 ))
        else
            n=$(( n + 1 ))
        fi
    done
    [ "$n" -le 0 ] && n=8
    echo "$n"
}

VISIBLE="$(_count_visible_devices)"
NPROC="${NPROC:-$VISIBLE}"
RANK_N="${RANK_N:-$NPROC}"
FAILED=0
PASSED=0

# Production 8-card shape for the op tests (S=4096, K=16, H=7168, E=896).
# The mte planning test hard-requires --ep-size 8, so --prod is 8-card only.
PROD_FLAG=()
if [ "$PROD" = 1 ]; then
    if [ "$RANK_N" -ne 8 ]; then
        echo "[ERROR] --prod requires RANK_N=8 (production 8-card shape); got RANK_N=$RANK_N"
        exit 1
    fi
    PROD_FLAG=(--prod-shape)
fi

# Engine selection: --use_urma -> urma, otherwise mte ubmem.
ENGINE_KW="mte"
[ "$USE_URMA" = 1 ] && ENGINE_KW="urma"

log() { echo -e "\n\033[1;34m=== $* ===\033[0m"; }
pass() { echo -e "\033[1;32m[PASS]\033[0m $*"; PASSED=$((PASSED+1)); }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; FAILED=$((FAILED+1)); }

run_cmd() {
    local label="$1"; shift
    log "$label"
    if "$@"; then
        pass "$label"
    else
        fail "$label"
    fi
}

log "Environment"
echo "  REPO_ROOT  = $REPO_ROOT"
echo "  SOC_TYPE   = $SOC_TYPE"
echo "  ENGINE     = $ENGINE_KW"
echo "  NPROC      = $NPROC (torchrun)"
echo "  RANK_N     = $RANK_N (script-style mp.spawn)"
echo "  PROD_SHAPE = $PROD (op tests ${PROD_FLAG[*]:-use their default shapes})"
echo "  ASCEND_RT_VISIBLE_DEVICES = ${ASCEND_RT_VISIBLE_DEVICES:-(unset, use all visible cards)}"

# pytest files that need NPROC cards
PYTEST_MTE_PREFETCH="test_npu/mte/test_prefetch.py"
PYTEST_URMA_PREFETCH="test_npu/urma/test_prefetch_udma.py"

# ---------------------------------------------------------------------------
# buffer tests: pytest-style files (select the urma/mte variant by engine)
# ---------------------------------------------------------------------------
BUFFER_PYTEST=(
    test_npu/buffer_test/test_buffer_grad_reduce.py
    test_npu/buffer_test/test_buffer_vmm.py
    test_npu/buffer_test/test_buffer_dispatch_with_reference_plan.py
    test_npu/buffer_test/test_buffer_plan.py
    test_npu/buffer_test/test_buffer_combine.py
    test_npu/buffer_test/test_buffer_prefetch.py
)

# buffer tests: script-style C / cpython interface tests (no engine switch; run as-is)
BUFFER_SCRIPT_C="test_npu/buffer_test/test_buffer_c_api.py"
BUFFER_SCRIPT_CPYTHON="test_npu/buffer_test/test_buffer_cpython_api.py"

if [ "${BUILD:-0}" = "1" ]; then
    log "Rebuilding _C extension (SOC_TYPE=$SOC_TYPE)"
    bash "$REPO_ROOT/npu_patch/scripts/build_base.sh" -soc_type "$SOC_TYPE" || { fail "build _C"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 1. buffer tests (pytest-style) - select the parametrized urma / mte variant
# ---------------------------------------------------------------------------
for f in "${BUFFER_PYTEST[@]}"; do
    run_cmd "pytest $f (engine=$ENGINE_KW, NPROC=$NPROC)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s "$f" -k "$ENGINE_KW"
done

# ---------------------------------------------------------------------------
# 2. buffer tests (script-style C / cpython interface) - no engine switch
# ---------------------------------------------------------------------------
run_cmd "test_buffer_c_api" \
    torchrun --nproc_per_node="$NPROC" --master_port=29503 "$BUFFER_SCRIPT_C"
run_cmd "test_buffer_cpython_api" \
    torchrun --nproc_per_node="$NPROC" --master_port=29507 "$BUFFER_SCRIPT_CPYTHON"

# Host-only check: per-allocation VMM page-size policy + mixed-granularity
# descriptor ordering. No collective/device allocation, so run it once.
run_cmd "test_buffer_vmm_page_size" \
    python -m pytest -s test_npu/buffer_test/test_buffer_vmm_page_size.py

# Host-only check: server-dedup env flags (ASCEND_MOONEP_DEDUP_MODE /
# ASCEND_MOONEP_SERVER_DEDUP) and the qp_count=2 clamp for that domain.
run_cmd "test_dedup_env_qp" \
    python -m pytest -s test_npu/buffer_test/test_dedup_env_qp.py

# ---------------------------------------------------------------------------
# 3. Op tests: --use_urma runs urma; otherwise mte ubmem
# ---------------------------------------------------------------------------
if [ "$USE_URMA" = 1 ]; then
    run_cmd "pytest test_npu/urma/test_grad_reduce_udma.py (NPROC=$NPROC)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s test_npu/urma/test_grad_reduce_udma.py
    run_cmd "pytest test_npu/urma/test_prefetch_udma.py ($NPROC cards)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s "$PYTEST_URMA_PREFETCH"

    # urma script-style op tests: all single-node mp.spawn (default --bootstrap
    # uid -> init_with_buffer). CLI differences: combine_udma has no --ep-size;
    # dispatch_udma has no --B; dispatch_combine_udma_joint has no --mode/--B.
    # R is RANK_N for all.
    #
    # Both dedup domains run every op test. "rank" is the Buffer default; the
    # "server" domain groups 8 consecutive ranks, needs UDMA dispatch+combine,
    # R % 8 == 0 and K >= 2, and uses a different duplicate-table ABI
    # (8*NvS full addresses, R+2 counts with per-source prefixes), so a green
    # rank run says nothing about it.
    #
    # The rank combine case defaults to the legacy MTE kernel (--combine-engine
    # auto); the extra small-shape case below covers the frozen rank-level UDMA
    # combine kernel (CombineUrma.compute_legacy) with rank tables. The server
    # dispatch case runs the real planner (--plan-source kernel) and validates
    # both the zero-copy and the copied outputs (--output-mode both).
    run_cmd "test_moonep_combine_udma_with_plan (kernel, dedup=rank)" \
        python test_npu/urma/test_moonep_combine_udma_with_plan.py --mode kernel --dedup-mode rank --R "$RANK_N" --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "test_moonep_combine_udma_with_plan (kernel, dedup=rank, combine=udma)" \
        python test_npu/urma/test_moonep_combine_udma_with_plan.py --mode kernel --combine-engine udma --dedup-mode rank --R "$RANK_N" --S 256 --K 8 --H 1024 --E 384 --plan-mode cyclic
    run_cmd "test_moonep_dispatch_udma_with_plan (kernel, dedup=rank)" \
        python test_npu/urma/test_moonep_dispatch_udma_with_plan.py --mode kernel --dedup-mode rank --R "$RANK_N" --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "test_dispatch_combine_udma_joint (dedup=rank)" \
        python test_npu/urma/test_dispatch_combine_udma_joint.py --dedup-mode rank --R "$RANK_N" --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    if [ $(( RANK_N % 8 )) -eq 0 ]; then
        run_cmd "test_moonep_combine_udma_with_plan (kernel, dedup=server)" \
            python test_npu/urma/test_moonep_combine_udma_with_plan.py --mode kernel --dedup-mode server --R "$RANK_N" --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
            ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
        run_cmd "test_moonep_dispatch_udma_with_plan (kernel+planner, dedup=server, K=16)" \
            python test_npu/urma/test_moonep_dispatch_udma_with_plan.py --mode kernel --plan-source kernel --dedup-mode server --R "$RANK_N" --S 8192 --K 16 --H 7168 --E 384 --plan-mode cyclic --output-mode both \
            ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
        run_cmd "test_dispatch_combine_udma_joint (dedup=server)" \
            python test_npu/urma/test_dispatch_combine_udma_joint.py --dedup-mode server --R "$RANK_N" --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
            ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    else
        log "skip server-dedup op tests: RANK_N=$RANK_N is not a multiple of 8"
    fi
    run_cmd "pytest test_npu/urma/test_e2e_urma.py (NPROC=$NPROC, dedup=rank|server)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s test_npu/urma/test_e2e_urma.py
else
    run_cmd "pytest test_npu/mte/test_grad_reduce.py (NPROC=$NPROC)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s test_npu/mte/test_grad_reduce.py
    run_cmd "pytest test_npu/mte/test_prefetch.py ($NPROC cards)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s "$PYTEST_MTE_PREFETCH"

    run_cmd "test_planning" \
        python test_npu/mte/test_planning.py --ep-size "$RANK_N" \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "test_moonep_combine" \
        python test_npu/mte/test_moonep_combine.py --ep-size "$RANK_N" \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "test_moonep_dispatch_with_plan (kernel)" \
        python test_npu/mte/test_moonep_dispatch_with_plan.py --mode kernel --R "$RANK_N" --S 64 --K 8 --H 128 --max-checks 8 \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "test_dispatch_combine_joint" \
        python test_npu/mte/test_dispatch_combine_joint.py --R "$RANK_N" --S 16 --K 4 --H 128 \
        ${PROD_FLAG[@]+"${PROD_FLAG[@]}"}
    run_cmd "pytest test_npu/mte/test_e2e.py (NPROC=$NPROC)" \
        torchrun --nproc-per-node="$NPROC" -m pytest -s test_npu/mte/test_e2e.py
fi

# ---------------------------------------------------------------------------
# 4. Grouped shmem-domain e2e (test_buffer_grouped_e2e)
#    Each subgroup builds its own Buffer + independent shmem symmetric heap and
#    checks self-consistency within a subset of the NPROC cards
#    (rank0-1 / rank2-3 ...). Pinned to MTE (use_udma=False).
#    Needs NPROC >= 2 * MOONEP_GROUP_SIZE for at least 2 subgroups; otherwise
#    skipped (not a failure).
# ---------------------------------------------------------------------------
GROUPED_TEST="test_npu/buffer_test/test_buffer_grouped_e2e.py"
GROUP_SIZE="${MOONEP_GROUP_SIZE:-2}"
if [ "$USE_URMA" = 0 ]; then
    if [ "$NPROC" -ge $(( GROUP_SIZE * 2 )) ]; then
        run_cmd "pytest $GROUPED_TEST ($(( NPROC / GROUP_SIZE )) groups x $GROUP_SIZE cards, NPROC=$NPROC)" \
            env MOONEP_GROUP_SIZE="$GROUP_SIZE" \
            torchrun --nproc-per-node="$NPROC" -m pytest -s "$GROUPED_TEST"
    else
        log "Skipping grouped e2e: NPROC=$NPROC cannot form 2 subgroups (need >= $(( GROUP_SIZE * 2 )), MOONEP_GROUP_SIZE=$GROUP_SIZE)"
    fi
fi

log "Summary"
echo "PASSED=$PASSED FAILED=$FAILED ENGINE=$ENGINE_KW SOC_TYPE=$SOC_TYPE NPROC=$NPROC PROD=$PROD"
[ "$FAILED" -eq 0 ]