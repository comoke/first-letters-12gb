# env.sh : shared settings for every chain in pc/ (source it). Override any of them from the environment.
#   VESUVIUS_DATA  data directory (winner downloads, outputs); default /mnt/d/Workspace/vesuvius-data
#   DOCKER         docker CLI; default /usr/bin/docker when present (on WSL with Docker Desktop, the native Linux CLI: the
#                  /usr/local/bin/docker symlink to docker.exe can't bind-mount /mnt/... paths), else the first docker on PATH
#   NVIDIA_SMI     default: nvidia-smi on PATH, else the WSL copy /usr/lib/wsl/lib/nvidia-smi
# Also sets PC (this pc/ directory, wherever the repository was cloned) and $D/scripts, where chains stage the helper scripts the
# winner's image runs. Exported, so private script copies (b0, retrain) and child chains see the same settings.
PC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=${VESUVIUS_DATA:-/mnt/d/Workspace/vesuvius-data}
DK=${DOCKER:-/usr/bin/docker}; [ -x "$DK" ] || DK=$(command -v docker || echo docker)   # no docker: a step that needs it fails with "docker: command not found"
NS=${NVIDIA_SMI:-$(command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi)}
mkdir -p "$D/scripts"
export PC VESUVIUS_DATA=$D DOCKER=$DK NVIDIA_SMI=$NS
