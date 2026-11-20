#!/usr/bin/env bash
# Set up the GPU machine for sim/: Isaac Sim 5.1, LabUtopia at the pinned commit with the assets
# sim/ uses, and one Python 3.11 venv (venv-sim/) for both LabUtopia and our ladder code.
# Run from the repo root. Safe to re-run: every step skips what is already done.
set -euo pipefail

COMMIT=$(python3 -c "import re;print(re.search(r'LABUTOPIA_COMMIT = \"(\w+)\"', open('sim/common.py').read())[1])")
CUROBO=v0.7.8  # NVlabs/curobo; see the cuRobo step below

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qE "A100|A800|H100|H200|B200"; then
    echo "This GPU has no RT cores; Isaac Sim cannot render on it. Use an L4, L40S, A10G or RTX card." >&2
    exit 1
fi

# Isaac Sim 5.1 is validated on the 580 driver branch. The 590/595 branch crashes its RTX
# renderer at startup (segfault in librtx.scenedb.plugin.so). Swapping the driver needs a reboot,
# so this script stops and prints the commands rather than doing it.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRIVER%%.*}" -ge 590 ]; then
    FLAVOUR=$(uname -r | sed 's/.*-//')   # gcp, aws, azure, generic
    cat >&2 <<EOF
Driver $DRIVER: Isaac Sim 5.1 crashes on the 590+ branch. Swap to 580, then re-run this script:
  sudo apt-get install linux-modules-nvidia-580-server-open-$FLAVOUR nvidia-utils-580-server \\
      libnvidia-compute-580-server libnvidia-gl-580-server nvidia-compute-utils-580-server \\
      libnvidia-cfg1-580-server nvidia-kernel-common-580-server libnvidia-encode-580-server \\
      nvidia-fabricmanager-${DRIVER%%.*}- libnvidia-nscq-${DRIVER%%.*}-
  sudo apt-mark hold nvidia-utils-580-server libnvidia-gl-580-server libnvidia-compute-580-server
  sudo reboot
(Check the package list with apt-get install -s first; it should remove only the old driver.)
EOF
    exit 1
fi

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git-lfs ffmpeg fonts-dejavu-core libglu1-mesa libxt6 libxrandr2 \
    libxinerama1 libxcursor1 libxi6 libsm6 libice6 libegl1 libvulkan1 vulkan-tools >/dev/null
# Cloud images often ship the compute-only driver: CUDA works, but there is no Vulkan driver, and
# Isaac Sim's RTX renderer needs one ("GPU Foundation is not initialized"). Add the driver's
# graphics libraries at exactly the installed version, from the same branch (-server or not).
if ! ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | grep -qi nvidia; then
    PKG=$(dpkg-query -W -f='${Package} ${Version}\n' 'libnvidia-compute-*' | head -1)
    GL=${PKG%% *}; GL=${GL/compute/gl}; VER=${PKG#* }
    echo "Adding the Vulkan driver: $GL=$VER"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$GL=$VER" >/dev/null
fi
# sim/live.py's WebRTC stream encodes on the GPU (NVENC). Without the encode library it fails with
# "Couldn't initialize the capture device". Same rule: exactly the installed version.
if ! ldconfig -p | grep -q libnvidia-encode.so.1; then
    PKG=$(dpkg-query -W -f='${Package} ${Version}\n' 'libnvidia-compute-*' | head -1)
    ENC=${PKG%% *}; ENC=${ENC/compute/encode}; VER=${PKG#* }
    echo "Adding the video encoder: $ENC=$VER"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$ENC=$VER" >/dev/null
fi
vulkaninfo --summary 2>/dev/null | grep -q "deviceName.*NVIDIA" \
    || { echo "Vulkan does not see the NVIDIA GPU; Isaac Sim cannot render." >&2; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== LabUtopia @ $COMMIT"
if [ ! -d external/LabUtopia ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Rui-li023/LabUtopia.git external/LabUtopia
fi
git -C external/LabUtopia fetch -q origin
# -f: the README images show as modified (LFS pointers vs files) and would block a re-pin.
git -C external/LabUtopia -c advice.detachedHead=false checkout -q -f "$COMMIT"
# Only the assets sim/ uses: the chemistry-lab scenes and the robots, not the navigation lab.
git -C external/LabUtopia lfs install --local >/dev/null
git -C external/LabUtopia lfs pull --include "assets/chemistry_lab/**,assets/robots/**,assets/properties.json"

echo "== Python 3.11 venv (venv-sim/)"
[ -d venv-sim ] || uv venv venv-sim --python 3.11
PY=./venv-sim/bin/python
uv pip install --python $PY torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $PY "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python $PY -r external/LabUtopia/requirements.txt
uv pip install --python $PY loguru  # Dev imports it; its requirements.txt does not list it
uv pip install --python $PY -r ladder/requirements.txt pillow

echo "== cuRobo $CUROBO (the skills' motion planner, sim/motion.py)"
# v0.7.8: the last version with the MotionGen API, and one that runs on Isaac Sim 5.1's own warp
# (1.8.2). main (v0.8) needs warp 1.18, on which Isaac Sim's core fails to import. Its CUDA kernels
# are torch extensions, compiled here: they need nvcc, at the CUDA major version torch was built for.
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    # A fresh Brev image has no NVIDIA CUDA apt repository (16 Nov 2026: "Unable to locate package
    # cuda-toolkit-12-8"); add it through NVIDIA's keyring package first.
    if ! apt-cache show cuda-toolkit-12-8 >/dev/null 2>&1; then
        . /etc/os-release
        curl -fsSL -o /tmp/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update -qq
    fi
    sudo apt-get install -y cuda-toolkit-12-8
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
[ -d external/curobo ] || git clone -q https://github.com/NVlabs/curobo.git external/curobo
git -C external/curobo -c advice.detachedHead=false checkout -q "$CUROBO"
# Everything already installed is pinned, and cuRobo goes in with --no-deps: its requirements would
# move packages Isaac Sim depends on (packaging, rich, websockets) and add a pip warp beside Isaac's
# own. Dry-run against a working venv-sim (21 Oct 2026): this only adds packages.
uv pip freeze --python $PY | grep -v -e "^-e" -e "@ file" > /tmp/spai-venv-pins.txt
uv pip install --python $PY -c /tmp/spai-venv-pins.txt tomli wheel ninja setuptools_scm pybind11 networkx \
    numpy-quaternion trimesh "yourdfpy>=0.0.53" scikit-image importlib_resources
uv pip install --python $PY --no-build-isolation --no-deps -e external/curobo

echo "== Headless start (the first one compiles shaders: 10 minutes or more)"
export OMNI_KIT_ACCEPT_EULA=YES
# Isaac Sim starts even when it cannot use the GPU, and only logs it, so check the log.
$PY -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()" 2>&1 | tee isaac-start.log | tail -3
if grep -qE "GPU Foundation is not initialized|no suitable CUDA GPU" isaac-start.log; then
    echo "Isaac Sim started without the GPU; see isaac-start.log." >&2
    exit 1
fi
echo "Isaac Sim OK"

# The household tasks' kitchen (tasks/home): private_meshes' models as USD. Objects #001-#150 there
# have no settled licence yet (its NOTICE.md): fetched here, never committed.
MESHES=$($PY -c "import sys; sys.path.insert(0, 'sim'); from common import MESHES_COMMIT; print(MESHES_COMMIT)")
echo "== private_meshes @ $MESHES, as USD (sim/import_meshes.py)"
[ -d external/private_meshes ] || git clone -q https://github.com/sunava/private_meshes.git external/private_meshes
git -C external/private_meshes fetch -q origin
git -C external/private_meshes -c advice.detachedHead=false checkout -q "$MESHES"
$PY sim/import_meshes.py objects/articulated/kitchen-appliances/102-kitchen_run_appliances_long 2>/dev/null | tail -1
# The props that dress some household objects (assets/home, CC BY 4.0): glb to USD.
$PY sim/import_glb.py assets/home/*.glb 2>/dev/null | tail -2
$PY sim/render_photo.py tasks/home/home_mug_screen/photo/task.json external/home_assets/mug_photo.png 2>/dev/null | tail -1

echo
echo "Done. Before each session:  source venv-sim/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES"
echo "Next: (cd external/LabUtopia && python main.py --headless --config-name level2_pour_liquid --max-episodes 1)"
