#!/usr/bin/env bash
# Copies code and inputs for scripts/cloud_vx_pod.sh to a pod.
#   [SSH_KEY=~/.ssh/key] scripts/cloud_vx_upload.sh HOST PORT [VX_TOKENIZER_DIR]
set -euo pipefail
host=${1:?host}; port=${2:?port}; vx=${3:-data/models/n-atlas-vx2}
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
e="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -p $port"
$e root@$host 'mkdir -p /workspace/work/data'
rsync -az -e "$e" --exclude .venv --exclude __pycache__ --exclude .pytest_cache wazocore root@$host:/workspace/
rsync -az -e "$e" scripts/cloud_vx_pod.sh scripts/cloud_vx_v6.sh \
    scripts/cloud_vx_v7.sh scripts/cloud_vx_v8.sh scripts/cloud_vx_v9.sh scripts/cloud_vx_v10.sh scripts/cloud_vx_v11.sh scripts/cloud_vx_v12.sh scripts/cloud_vx_v13.sh \
    scripts/v8_aya.py scripts/v9_audit.py scripts/v9_pick.py scripts/v9_probe.py scripts/v9_tasks.py scripts/v10_data.py scripts/v10_tasks.py scripts/v10_analysis.py \
    scripts/v11_tasks.py scripts/v11_pick.py scripts/v11_analysis.py scripts/v12_data.py scripts/v12_tasks.py scripts/v12_pick.py scripts/v12_analysis.py \
    scripts/v13_data.py scripts/v13_pick.py scripts/v13_analysis.py scripts/v7_analysis.py scripts/v8_analysis.py scripts/v9_analysis.py \
    scripts/africomet.py scripts/cloud_gguf.sh scripts/gguf_quality.py root@$host:/workspace/
echo "$(git rev-parse HEAD)$(git diff --quiet HEAD -- wazocore scripts || echo ' dirty')" | $e root@$host 'cat > /workspace/wazocore/REVISION'
# v7 to v13 on a fresh pod: send back what earlier pods finished, so it is skipped
[ -d results/vocab/v7 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v7/ root@$host:/workspace/work/results-v7/
[ -d results/vocab/v8 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v8/ root@$host:/workspace/work/results-v8/
[ -d results/vocab/v9 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v9/ root@$host:/workspace/work/results-v9/
[ -d results/vocab/v10 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v10/ root@$host:/workspace/work/results-v10/
[ -d results/vocab/v11 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v11/ root@$host:/workspace/work/results-v11/
[ -d results/vocab/v12 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v12/ root@$host:/workspace/work/results-v12/
[ -d results/vocab/v13 ] && rsync -az -e "$e" --exclude 'run-*.log' results/vocab/v13/ root@$host:/workspace/work/results-v13/
[ -d data/ntrex ] && rsync -az -e "$e" data/ntrex/ root@$host:/workspace/work/ntrex/
rsync -az -e "$e" data/wiki data/flores200_dataset root@$host:/workspace/work/data/
rsync -az -e "$e" "$vx"/ root@$host:/workspace/work/vx-tok/
# optional smaller vocabulary for the `small` arm of cloud_vx_v6.sh
[ -d data/models/n-atlas-vx-5k ] && rsync -az -e "$e" data/models/n-atlas-vx-5k/ root@$host:/workspace/work/vx-tok-small/
for f in data/runs/cloud/chat-distill.jsonl data/runs/cloud/chat-wiki.jsonl; do  # optional: regenerated on the pod if absent
    [ -s "$f" ] && rsync -az -e "$e" "$f" root@$host:/workspace/work/
done
# The pod image has no `hf`; the login below needs it before the pipeline runs.
$e root@$host '[ -x /workspace/venv/bin/hf ] || { python -m venv --system-site-packages /workspace/venv && /workspace/venv/bin/pip install -q "transformers==5.17.0" safetensors "huggingface_hub[hf_xet]" sacrebleu; }; /workspace/venv/bin/python -c "import torch; print(\"GPU:\", torch.cuda.get_device_name(0))"'
echo "uploaded; on the pod: /workspace/venv/bin/hf auth login (read token), then tmux new -s vx 'bash /workspace/cloud_vx_pod.sh 2>&1 | tee /workspace/run.log'"
