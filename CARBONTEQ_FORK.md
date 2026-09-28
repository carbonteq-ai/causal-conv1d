# CarbonTeq AI causal-conv1d fork

This repository is a fork of
[`Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d). It
exists to retain CarbonTeq's binary rebuilds of an unmodified upstream release
for PyTorch versions that upstream does not publish wheels for. Posttrain
consumes the wheels so Transformers can run Qwen3.5 Gated DeltaNet layers on
the fast path (together with fla-core). The consumer record is
`docs/tooling/linear-attention-kernels/README.md` in the Posttrain repository
(`carbonteq-ai/posttrain`); the plan is `docs/plan/qwen35-fast-gdn-kernels.md`
there.

## Status

Published binary rebuild. Release `carbonteq-v1.7.0+cu130torch2.13` retains
one wheel; see Releases.

## Upstream base

`Dao-AILab/causal-conv1d` tag `v1.7.0`, commit
`cd81f0413cad2fc1e6f17e785ac39f59aae690cd` (2026-08-20). Upstream `main` was
at the same commit when the fork was created.

Expected remotes: `origin` is `https://github.com/carbonteq-ai/causal-conv1d.git`
and `upstream` is `https://github.com/Dao-AILab/causal-conv1d.git`.

## Maintained delta

None in source. The only commit on top of the base adds this ledger. The
wheel is built from the base commit's tree unchanged.

Why a rebuild is needed: causal-conv1d is a CUDA extension compiled against
the PyTorch C++ API, so a wheel works only with the PyTorch it was built
against. Upstream's v1.7.0 release carries wheels up to PyTorch 2.10; PyPI has
only the source distribution. Posttrain runs PyTorch 2.13.0+cu130 on CPython
3.13.

## Build recipe

Posttrain's `tools/kernel-wheels/causal-conv1d/build.sh WORK_DIR` clones this
base commit, verifies it, and builds inside Posttrain's
`posttrain-kind-online-rl-trl-py312` image
(`registry.lan/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:3a578d8b0f06453ea0975d05e03b20fed74537cae103d4eb5254d3806d3b1e47`):
CPython 3.13.12, torch 2.13.0+cu130 (C++11 ABI), the pip CUDA 13.0.88 toolkit,
GCC 12, `wheel==0.45.1`. Settings: `CAUSAL_CONV1D_FORCE_BUILD=TRUE`,
`CAUSAL_CONV1D_LOCAL_VERSION=cu130torch2.13`, `SOURCE_DATE_EPOCH` = base commit
time, `MAX_JOBS=4`, and
`NVCC_APPEND_FLAGS="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"`.
Upstream `setup.py` hard-codes sm75/80/87/90/100/103/110/120/121 for CUDA 13
(so `TORCH_CUDA_ARCH_LIST` is ignored); the appended flags add native RTX 30
(sm86) and Ada (sm89) code without patching source.

The build is hermetic but not bit-reproducible: nvcc embeds per-process
`tmpxft_<pid>` identifiers. The released bytes are the identity.

## Releases

| Tag | Wheel | SHA-256 |
| --- | --- | --- |
| `carbonteq-v1.7.0+cu130torch2.13` | `causal_conv1d-1.7.0+cu130torch2.13-cp313-cp313-linux_x86_64.whl` | `b69f39142ac88cac91cba5f954cb616c50bc49933cd84f349319420470a4947a` |

Publication follows Posttrain's retained-asset route: Posttrain's
`publish-causal-conv1d-internal.yml` downloads the release asset, checks the
hash and uploads the same bytes to `https://pypi.lan/carbonteq/dev/`. This fork
has no release automation and no index credentials.

## Validation

On an RTX 3070 Ti (sm86), against Transformers 5.14.1's torch reference:
forward/backward relative error 3.1e-3 (bf16), 3.7e-4 (fp16), 5.3e-8 (fp32);
`causal_conv1d_update` matches. Command: Posttrain
`scripts/qualification/qwen35_gdn_kernels.py kernels`. Hardware gate for a new
architecture: the same command on that GPU.

## Rebase procedure

For a new upstream release: fast-forward `main` to the upstream tag, keep this
file, update the base commit and the pinned revision in Posttrain's
`build.sh`, rebuild, and add a new `carbonteq-v<version>+cu<cuda>torch<torch>`
release. For a new PyTorch or CUDA version with the same upstream release,
rebuild from the same base with a new local version label.

## Deferred

No aarch64, ROCm, or other CPython builds are produced.
