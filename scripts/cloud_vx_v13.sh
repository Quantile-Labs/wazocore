#!/usr/bin/env bash
# v13: v10's self-distilled data split into its maths and chat halves, each
# removed in turn from v12's base (same recipe, code and seed), as
# preregistered in docs/METHOD.md, section 4.6. Built like v12 for RunPod Community
# Cloud: every step is skipped once its result exists, results are small
# files the laptop pulls as they land (RUN=v13 scripts/cloud_pull.sh), and
# no weights leave the pod. Run after scripts/cloud_vx_upload.sh and
# `hf auth login`:
#   bash /workspace/cloud_vx_v13.sh
# UNITS, in priority order (default all):
#   check   the release rescored against v9; NTREX 0-499 for the release
#   data    the split (v13_data.py)
#   train   no-chat, no-math: v12 base's recipe at seed 2, then evaluated
#   pick    stages A and B (v13_pick.py)
#   comet   AfriCOMET on the held-out samples
#   decide  v13_analysis.py on the pod: decision.json
#   stage2  the candidate at seeds 3 and 4, only if decision.json says so
#   gguf    after stage 2: the candidate retrained at seed 2 and checked against
#           its evaluated run, the NFC tokenizer, Q4_K_M (and edge-q8_0) files,
#           an N-ATLaS Q4_K_M from the same llama.cpp, and NTREX quality and
#           defects for both, the release and the original (gguf_quality.py).
#           The files stay in $W/gguf; the laptop fetches them.
# Generative AfriXNLI comes from v11_tasks.py, domain-conditional from v12_tasks.py.
# On an out-of-memory error in training lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v13; M=$W; D=$R/data; mkdir -p "$R" "$D"
UNITS=${UNITS:-"check data train pick comet decide stage2"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
IB=16  # fixed lm-eval batch size, as in v9 to v12
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex; T=$W/tasks-v13; SDFT=$W/results-v10/data/chat-v10-sdft.jsonl
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=quantile-labs/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
RECIPE="--anchor canonical --canon-weight 1.0 --bottom-lr-scale 0.25"  # v11's lowlr
R_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l; done)
MMLU_TASKS=$(for l in yor hau ibo eng; do printf 'afrimmlu_direct_%s_prompt_1,' $l; done)
GEN_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1_gen,' $l; done)
DC_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1_dc,' $l; done)
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: the release was trained on it"
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v13 checks its inputs against v7's"
[ -d $W/results-v8/iroko-long-s1 ] || die "upload results/vocab/v8: stage 2 compares with release seed 1"
[ -d $W/results-v9/iroko-release ] || die "upload results/vocab/v9: the release must score as it did there"
[ -d $W/results-v12/iroko-base ] || die "upload results/vocab/v12: base is the reference"
{ [ -s $W/results-v10/data/answer.jsonl ] && [ -s $W/results-v10/data/aya.jsonl ]; } || die "upload results/vocab/v10/data: the split reads its prompts"
[ -s $SDFT ] || die "upload results/vocab/v10 with its data/ folder: v13 splits chat-v10-sdft.jsonl"
(cd $W/results-v10/data && sha256sum chat-v10-sdft.jsonl) | cmp -s - <(grep ' chat-v10-sdft.jsonl$' $W/results-v10/inputs.sha256) \
    || die "chat-v10-sdft.jsonl differs from v10's"
$HF download NCAIR1/N-ATLaS --revision $BASE_REV --local-dir $M/n-atlas --exclude "*.gguf" --exclude "original/*" >/dev/null \
    || die "no access to N-ATLaS: $HF auth login"
if [ ! -f $M/release/.checked ]; then
    $HF download $RELEASE --revision $RELEASE_REV --local-dir $M/release --exclude "*.gguf" >/dev/null
    echo "$RELEASE_SHA  $M/release/model.safetensors" | sha256sum -c --quiet || die "release weights differ from v6-long"
    touch $M/release/.checked
fi
[ -x $EV/lm_eval ] || { python -m venv --system-site-packages /workspace/venv-eval \
    && $EV/pip install -q "lm-eval[hf]==0.4.13" "transformers==5.17.0"; }
[ -s $T/score_v11.py ] || $EV/python -I /workspace/v11_tasks.py tasks --out $T --wazocore /workspace/wazocore
[ -s $T/dc_v12.py ] || $EV/python -I /workspace/v12_tasks.py --out $T

log setup
# v6's gold data, which must match v7's byte for byte: the expanded model is built from it
W=$W ARMS=data GGUF=none bash /workspace/cloud_vx_v6.sh
(cd $W && sha256sum chat-wiki.jsonl chat-v6-gold.jsonl vx-tok/new_tokens.json) > $R/gold.now
cmp -s $R/gold.now $W/results-v7/inputs.sha256 || die "gold data differs from v7's: see $R/gold.now"
rm $R/gold.now
{ date -u +%FT%TZ; cat /workspace/wazocore/REVISION 2>/dev/null || echo "no REVISION file"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"; } >> $R/env.txt

cd /workspace/wazocore
V="$PY -m wazocore.vx_torch"
iroko() {  # NAME MODEL_DIR TASKS [lm_eval flags]: IrokoBench, zero-shot, prompt 1, chat template
    local name=$1 dir=$2 tasks=$3; shift 3
    [ -f $R/iroko-$name/done ] && return 0
    rm -rf "${R:?}/iroko-$name"; log "irokobench $name"
    $EV/lm_eval --model hf --model_args pretrained=$dir,dtype=bfloat16 --tasks ${tasks%,} \
        --apply_chat_template --batch_size $IB --seed 0 --log_samples --output_path $R/iroko-$name "$@"
    touch $R/iroko-$name/done
}
sample() {  # FILE MODEL_DIR [sample flags]; NTREX needs --ntrex in the flags, FLORES devtest is the default
    local out=$1 dir=$2; shift 2
    [ -s $R/$out ] && return 0
    log "sample $out"
    $V sample --model $dir --flores $F --batch 64 "$@" --out $R/$out
}
same_items() {  # A_DIR A B_DIR B: every R item of B scored exactly as in A
    $PY -I - "$@" <<'EOF'
import json, sys
from pathlib import Path
def items(r, p):
    out = {}
    for f in Path(r).glob(f"iroko-{p}/*/samples_*.jsonl"):
        for x in map(json.loads, f.read_text().splitlines()):
            if x["filter"] in ("none", "flexible-extract"):
                out[(f.name.split("_prompt_1")[0], x["doc_id"])] = x.get("acc", x.get("exact_match"))
    return out
a, b = items(sys.argv[1], sys.argv[2]), items(sys.argv[3], sys.argv[4])
diff = [k for k in b if a.get(k) != b[k]]
print(f"{sys.argv[4]} against {sys.argv[2]}: {len(diff)} of {len(b)} items differ")
sys.exit(1 if diff or not b or set(a) != set(b) else 0)
EOF
}
extras() {  # NAME MODEL_DIR MAIN_RESULTS: generative and domain-conditional AfriXNLI, with the bug check
    iroko gen-$1 $2 "$GEN_TASKS" --include_path $T
    iroko dc-$1 $2 "$DC_TASKS" --include_path $T
    $PY -I /workspace/v13_analysis.py check-dc $R $3 $1 || die "domain-conditional task for $1 does not score the first item as lm-eval did"
}
init() {  # the expanded, untrained model with the release's initialisation
    [ -f $M/init-v6/.done ] && return 0
    rm -rf "${M:?}/init-v6"; log "expand init-v6"
    $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/init-v6 --out-weights chars --calibrate-input-norm
    touch $M/init-v6/.done
}
evaluate() {  # NAME: NTREX 0-499, held-out translation, BPC, IrokoBench with AfriMMLU, generative and domain-conditional
    local d way
    sample sel-$1.json $M/$1 --ntrex $N --start 0 --n 500
    for d in en-xx xx-en; do
        way=; [ $d = xx-en ] && way=--reverse
        sample heldout-$d-$1.json $M/$1 --ntrex $N --start 500 --n 1497 $way
    done
    [ -s $R/bpc-$1.json ] || { log "bpc $1"; $V bpc --model $M/$1 --flores $F --sentences 1012 --out $R/bpc-$1.json; }
    iroko $1 $M/$1 "$R_TASKS$MMLU_TASKS"
    extras $1 $M/$1 $R
}
fit() {  # NAME SEED CHAT [flags]: lowlr's recipe at SEED on CHAT
    local name=$1 seed=$2 chat=$3; shift 3
    init
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name $*"
        $V train --model $M/init-v6 --flores $F --out $M/$name --data $W/data --chat $chat \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH $RECIPE "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
}
train() {  # NAME SEED CHAT [flags]: fit, then evaluate and free the disk
    [ -f $R/done-$1 ] && { log "$1: done"; return 0; }
    fit "$@"
    local name=$1
    evaluate $name
    touch $R/done-$name
    rm -rf "${M:?}/$name"
}
arm() {  # MODEL SEED [NAME]: the arm's data
    local name=${3:-$1}
    case $1 in
    no-chat|no-math) train $name $2 $D/chat-v13-$1.jsonl ;;
    *) die "unknown arm $1" ;;
    esac
}
need() { [ -s $D/$1 ] || die "$1 missing: run the data unit first"; }

for u in $UNITS; do
    case $u in
    check)
        iroko release-v13 $M/release "$R_TASKS"
        same_items $W/results-v9 release $R release-v13 || die "the release scores differently from v9: see the log"
        sample sel-release.json $M/release --ntrex $N --start 0 --n 500 ;;
    data)
        [ -s $D/chat-v13-no-math.jsonl ] || { log "data: split"; $EV/python -I /workspace/v13_data.py --sdft $SDFT \
            --gold $W/chat-v6-gold.jsonl --v10 $W/results-v10/data --vx $W/vx-tok --base $M/n-atlas --out $D --report $R/split-report.json; }
        (cd $D && sha256sum chat-v13-no-chat.jsonl chat-v13-no-math.jsonl) > $R/inputs.now
        [ -s $R/inputs.sha256 ] || cp $R/inputs.now $R/inputs.sha256
        cmp -s $R/inputs.now $R/inputs.sha256 || die "v13 data differs from the first pod's: see $R/inputs.sha256"
        rm $R/inputs.now ;;
    train)
        need chat-v13-no-chat.jsonl; need chat-v13-no-math.jsonl
        for m in no-chat no-math; do arm $m 2; done ;;
    pick)
        if [ ! -s $R/pick.json ]; then
            log "stage A: $($PY -I /workspace/v13_pick.py rank $R)"
            $PY -I /workspace/v13_pick.py choose $R >/dev/null
        fi
        log "stage B: candidate $($PY -c "import json; p = json.load(open('$R/pick.json')); print(p['candidate'], '(under 0.5:', p['sensitivity']['candidate'], ')')")" ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json 2>/dev/null) ;;
    decide)
        log decide; $PY -I /workspace/v13_analysis.py $W > $R/analysis-pod.txt
        log "decision: $(tr -d '\n ' < $R/decision.json)" ;;
    stage2)
        [ -s $R/decision.json ] || die "no decision.json: run the decide unit first"
        if $PY -c "import json, sys; sys.exit(0 if json.load(open('$R/decision.json'))['stage2'] else 1)"; then
            c=$($PY -c "import json; print(json.load(open('$R/decision.json'))['candidate'])")
            for s in 3 4; do arm $c $s $c-s$s; done
            /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json)
            $PY -I /workspace/v13_analysis.py $W > $R/analysis-pod.txt
        else
            log "stage 2 does not run (preregistered decisions)"
        fi ;;
    gguf)
        c=$($PY -c "import json; d = json.load(open('$R/decision.json')); print(d['candidate'] if d['stage2'] else '')")
        [ -n "$c" ] || die "stage 2 did not pass: no file to build (preregistered decisions)"
        G=$R/gguf; Q=$W/gguf; S=$c-ship; mkdir -p $G $Q
        cd /workspace
        [ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
        command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake >/dev/null; }
        build=/workspace/llama.cpp/build-cuda/bin
        [ -x $build/llama-server ] || (export PATH=/usr/local/cuda/bin:$PATH; cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF \
            && cmake --build llama.cpp/build-cuda --target llama-server llama-tokenize -j 32) > $G/build-cuda.log 2>&1 &
        cd /workspace/wazocore
        fit $S 2 $D/chat-v13-$c.jsonl
        iroko $S $M/$S "$R_TASKS"
        same_items $R $c $R $S > $G/reproduction.txt 2>&1 || true
        log "reproduction: $(cat $G/reproduction.txt)"
        [ -s $M/$S-nfc/tokenizer.json ] || $EV/python -I /workspace/v11_tasks.py tokenizer --model $M/$S --out $M/$S-nfc
        cp $M/$S-nfc/tokenizer.json $G/tokenizer.json
        mkdir -p $G/convert-$c $G/convert-n-atlas
        [ -s $Q/WazoCore-v13.Q4_K_M.gguf ] || { log "gguf $c"; EDGE_TYPE=q8_0 bash /workspace/cloud_gguf.sh $M/$S-nfc $Q/WazoCore-v13.Q4_K_M.gguf $G/convert-$c; }
        [ -s $Q/N-ATLaS.Q4_K_M.gguf ] || { log "gguf n-atlas"; bash /workspace/cloud_gguf.sh $M/n-atlas $Q/N-ATLaS.Q4_K_M.gguf $G/convert-n-atlas; }
        [ -s $Q/release/WazoCore-8B.Q4_K_M.gguf ] || $HF download $RELEASE WazoCore-8B.Q4_K_M.gguf --revision $RELEASE_REV --local-dir $Q/release >/dev/null
        (cd $Q && sha256sum WazoCore-v13.Q4_K_M.gguf WazoCore-v13.Q4_K_M.edge-q8_0.gguf N-ATLaS.Q4_K_M.gguf release/WazoCore-8B.Q4_K_M.gguf) > $G/sha256.txt
        wait; [ -x $build/llama-server ] || die "llama.cpp CUDA build failed: see $G/build-cuda.log"
        # does the GGUF normalize? the same Yoruba line in NFC and NFD, tokenized by llama.cpp and by tokenizer.json
        $PY -I - $build/llama-tokenize $Q/WazoCore-v13.Q4_K_M.gguf $G/tokenizer.json > $G/nfc-gguf.json <<'PY' || log "nfc check failed: see $G/nfc-gguf.json"
import json, subprocess, sys, unicodedata
from tokenizers import Tokenizer
text = "Ọmọ náà ń lọ sí ilé-ìwé ní òwúrọ̀ yìí, ṣùgbọ́n òjò ń rọ̀."
tok = Tokenizer.from_file(sys.argv[3])
def llama(t):
    out = subprocess.run([sys.argv[1], "-m", sys.argv[2], "-p", t, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])
f = {k: unicodedata.normalize(k, text) for k in ("NFC", "NFD")}
r = {f"{who} {k}": ids for k, t in f.items() for who, ids in (("gguf", llama(t)), ("tokenizer.json", tok.encode(t, add_special_tokens=False).ids))}
print(json.dumps({"text_nfc": f["NFC"], "same_ids": {"gguf": r["gguf NFC"] == r["gguf NFD"], "tokenizer.json": r["tokenizer.json NFC"] == r["tokenizer.json NFD"],
                  "gguf NFC = tokenizer.json NFC": r["gguf NFC"] == r["tokenizer.json NFC"]}, "ids": r}, ensure_ascii=False, indent=1))
PY
        log "nfc in gguf: $(tr -d '\n' < $G/nfc-gguf.json | grep -o '"same_ids": {[^}]*}')"
        mkdir -p $G/q/data/models && ln -sfn $N $G/q/data/ntrex && ln -sfn $Q/N-ATLaS.Q4_K_M.gguf $G/q/data/models/N-ATLaS.Q4_K_M.gguf
        for x in release:$Q/release/WazoCore-8B.Q4_K_M.gguf v13:$Q/WazoCore-v13.Q4_K_M.gguf v13-edge:$Q/WazoCore-v13.Q4_K_M.edge-q8_0.gguf; do
            [ -f $G/quality-${x%%:*} ] && continue
            log "quality ${x%%:*}"
            (cd $G/q && PATH=$build:$PATH SKIP=wazocore $PY -I /workspace/gguf_quality.py 250 $G ${x%%:*} ${x#*:}) && touch $G/quality-${x%%:*}
        done
        rm -rf "$G/q" ;;
    *) die "unknown unit: $u" ;;
    esac
done
rm -rf "${M:?}/init-v6"
log ALL-DONE
