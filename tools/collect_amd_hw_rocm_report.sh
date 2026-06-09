#!/usr/bin/env bash
set -u

OUT="amd_hw_rocm_report_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "================================================================================"
  echo "SYSTEM"
  echo "================================================================================"
  date
  uname -a
  cat /etc/os-release || true
  echo
  cat /proc/cmdline || true
  echo

  echo "================================================================================"
  echo "CPU"
  echo "================================================================================"
  lscpu || true
  echo

  echo "================================================================================"
  echo "PCI GPU / DISPLAY DEVICES"
  echo "================================================================================"
  lspci -nnk | grep -EA5 "VGA|Display|3D|AMD|ATI|Radeon" || true
  echo

  echo "================================================================================"
  echo "FULL AMD PCI VERBOSE"
  echo "================================================================================"
  for dev in $(lspci -Dn | awk '/1002:/{print $1}'); do
    echo "---- $dev ----"
    sudo lspci -vvv -s "$dev" || true
    echo
  done

  echo "================================================================================"
  echo "DMI / BIOS / PRODUCT"
  echo "================================================================================"
  sudo dmidecode -t system -t baseboard -t bios 2>/dev/null || true
  echo

  echo "================================================================================"
  echo "GPU DEVICE NODES / GROUPS"
  echo "================================================================================"
  ls -l /dev/kfd /dev/dri/renderD* /dev/dri/card* 2>/dev/null || true
  groups
  id
  echo

  echo "================================================================================"
  echo "ROCm / HSA DISCOVERY"
  echo "================================================================================"
  which rocminfo || true
  rocminfo 2>&1 | sed -n '1,220p' || true
  echo

  echo "================================================================================"
  echo "ROCm SMI / AMD SMI"
  echo "================================================================================"
  which rocm-smi || true
  rocm-smi 2>&1 || true
  echo
  which amd-smi || true
  amd-smi static 2>&1 || true
  echo
  amd-smi list 2>&1 || true
  echo

  echo "================================================================================"
  echo "AMDGPU MODULE"
  echo "================================================================================"
  modinfo amdgpu 2>/dev/null | grep -E "filename|version|srcversion|vermagic|parm" || true
  echo
  lsmod | grep -E "amdgpu|amd|kfd|drm" || true
  echo
  dkms status | grep -iE "amdgpu|rocm|amd" || true
  echo

  echo "================================================================================"
  echo "INSTALLED ROCm / AMDGPU PACKAGES"
  echo "================================================================================"
  dpkg -l | grep -Ei "rocm|hsa|hip|amdgpu|migraphx|hsakmt|rocminfo|rocm-smi|amd-smi|miopen|rocblas|hipblas|libdrm-amdgpu" || true
  echo

  echo "================================================================================"
  echo "APT POLICY IMPORTANT PACKAGES"
  echo "================================================================================"
  apt-cache policy \
    amdgpu-install amdgpu-dkms amdgpu-core amdgpu-dkms-firmware \
    rocm-core hsa-rocr hip-runtime-amd rocm-hip rocminfo migraphx migraphx-dev \
    libhsa-runtime64-1 libhsakmt1 libamdhip64-5 \
    2>/dev/null || true
  echo

  echo "================================================================================"
  echo "LINKER LIBRARIES"
  echo "================================================================================"
  ldconfig -p | grep -E "libhsa-runtime64|libhsakmt|libamdhip64|libmigraphx|librocblas|libMIOpen|libhiprtc" || true
  echo

  echo "================================================================================"
  echo "KERNELS AVAILABLE / INSTALLED"
  echo "================================================================================"
  dpkg -l | grep -E "linux-image|linux-headers|linux-modules" | grep -E "6\.|7\." || true
  echo
  apt-cache search linux-image | grep -E "6\.1|6\.8|6\.11|6\.14|6\.17|6\.18|6\.19|7\.0|oem|generic" | tail -120 || true
  echo

  echo "================================================================================"
  echo "FIRMWARE AMDGPU"
  echo "================================================================================"
  find /lib/firmware /usr/lib/firmware -path "*amdgpu*" -type f 2>/dev/null | sort | tail -200 || true
  echo

  echo "================================================================================"
  echo "RECENT AMDGPU / KFD / HSA DMESG"
  echo "================================================================================"
  sudo dmesg | grep -iE "amdgpu|kfd|hsa|gpu|gfx|ring|fault|timeout|reset|sdma|mes|vmid|pasid|permission|mapping|walker|firmware" | tail -240 || true
  echo

  echo "================================================================================"
  echo "PYTHON / TORCH ENV"
  echo "================================================================================"
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "VIRTUAL_ENV=$VIRTUAL_ENV"
  fi
  which python || true
  python --version || true
  python - <<'PY' 2>&1 || true
import sys, os
print("python:", sys.version)
print("executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch hip:", torch.version.hip)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import/test failed:", repr(e))
try:
    import migraphx
    print("migraphx:", getattr(migraphx, "__version__", None), migraphx.__file__)
    print("has parse_onnx:", hasattr(migraphx, "parse_onnx"))
except Exception as e:
    print("migraphx import failed:", repr(e))
PY

} | tee "$OUT"

echo
echo "Saved report: $OUT"
