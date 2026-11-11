#!/usr/bin/env bash
# v11: fewer trained bottom-layer weights on v10's sdft data, chosen on the
# selection half and tested on the confirmation half; generative AfriXNLI;
# NFC input. As preregistered in docs/METHOD.md, section 4.6. Built like v10 for
# RunPod Community Cloud: every step is skipped once its result exists,
# results are small files the laptop pulls as they land
# (RUN=v11 scripts/cloud_pull.sh), and no weights leave the pod. Run after
# scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v11.sh
# UNITS, in priority order (default all):
#   check   the release rescored against v9; generative and NFC IrokoBench
#           for the original and the release; NTREX 0-499 for the release;
#           the NFC tokenizer check
#   train   sdft-s1, l1, lowlr, top: trained at seed 1, then evaluated
#   mid     sdft-mid (sdft-s1 with layers 0 and 1 halfway back), evaluated
#   pick    stages A and B (v11_pick.py)
#   comet   AfriCOMET on the held-out samples
#   decide  v11_analysis.py on the pod: decision.json
#   stage2  the candidate at seeds 2 and 3, only if decision.json says so
# On an out-of-memory error in training lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v11; M=$W; mkdir -p "$R"
UNITS=${UNITS:-"check train mid pick comet decide stage2"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
IB=16  # fixed lm-eval batch size, as in v9 and v10
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex; T=$W/tasks-v11; SDFT=$W/results-v10/data/chat-v10-sdft.jsonl
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=quantile-labs/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
CANON="--anchor canonical --canon-weight 1.0"
R_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l; done)
MMLU_TASKS=$(for l in yor hau ibo eng; do printf 'afrimmlu_direct_%s_prompt_1,' $l; done)
GEN_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1_gen,' $l; done)
NFC_TASKS=$(for l in yor ibo; do printf 'afrixnli_%s_prompt_1_nfc,afrimgsm_%s_prompt_1_nfc,afrimmlu_direct_%s_prompt_1_nfc,' $l $l $l; done)
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: the release was trained on it"
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v11 checks its inputs against v7's"
[ -d $W/results-v8/iroko-long-s1 ] || die "upload results/vocab/v8: stage 2 compares with release seed 1"
[ -d $W/results-v9/iroko-release ] || die "upload results/vocab/v9: the release must score as it did there"
[ -s $SDFT ] || die "upload results/vocab/v10 with its data/ folder: v11 trains on chat-v10-sdft.jsonl"
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
sample() {  # FILE MODEL_DIR [sample flags]
    local out=$1 dir=$2; shift 2
    [ -s $R/$out ] && return 0
    log "sample $out"
    $V sample --model $dir --flores $F --ntrex $N --batch 64 "$@" --out $R/$out
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
extras() {  # NAME MODEL_DIR: generative and NFC IrokoBench
    iroko gen-$1 $2 "$GEN_TASKS" --include_path $T
    iroko nfc-$1 $2 "$NFC_TASKS" --include_path $T
}
init() {  # the expanded, untrained model with the release's initialisation
    [ -f $M/init-v6/.done ] && return 0
    rm -rf "${M:?}/init-v6"; log "expand init-v6"
    $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/init-v6 --out-weights chars --calibrate-input-norm
    touch $M/init-v6/.done
}
evaluate() {  # NAME: NTREX 0-499, held-out translation, BPC, IrokoBench with AfriMMLU, generative and NFC
    local d way
    sample sel-$1.json $M/$1 --start 0 --n 500
    for d in en-xx xx-en; do
        way=; [ $d = xx-en ] && way=--reverse
        sample heldout-$d-$1.json $M/$1 --start 500 --n 1497 $way
    done
    [ -s $R/bpc-$1.json ] || { log "bpc $1"; $V bpc --model $M/$1 --flores $F --sentences 1012 --out $R/bpc-$1.json; }
    iroko $1 $M/$1 "$R_TASKS$MMLU_TASKS"
    extras $1 $M/$1
}
train() {  # NAME SEED KEEP [flags]: v10's sdft recipe at SEED, then evaluate; KEEP=1 keeps the weights
    local name=$1 seed=$2 keep=$3; shift 3
    [ -f $R/done-$name ] && { log "$name: done"; return 0; }
    init
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name $*"
        $V train --model $M/init-v6 --flores $F --out $M/$name --data $W/data --chat $SDFT \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH $CANON "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
    evaluate $name
    touch $R/done-$name
    [ "$keep" = 1 ] || rm -rf "${M:?}/$name"
}
mid() {  # NAME FROM: FROM with layers 0 and 1 halfway back to N-ATLaS, evaluated
    [ -f $R/done-$1 ] && { log "$1: done"; return 0; }
    [ -f $M/$2/.trained ] || die "$2 weights are gone (a new pod): delete $R/done-$2 to retrain it, then rerun"
    [ -f $M/$1/.done ] || { rm -rf "${M:?}/$1"; log "interp $1"
        $V interp --base $M/n-atlas --model $M/$2 --bottom 0.5 --top 1 --out $M/$1; touch $M/$1/.done; }
    evaluate $1
    touch $R/done-$1
    rm -rf "${M:?}/$1"
}
flags() {  # MODEL: the trainer flags that define it
    case $1 in
    sdft-s1|sdft-mid) echo "" ;; l1) echo "--layers 1 30 31" ;; lowlr) echo "--bottom-lr-scale 0.25" ;; top) echo "--layers 30 31" ;;
    *) die "no flags for $1" ;;
    esac
}

for u in $UNITS; do
    case $u in
    check)
        iroko release-v11 $M/release "$R_TASKS"
        same_items $W/results-v9 release $R release-v11 || die "the release scores differently from v9: see the log"
        for m in n-atlas release; do extras $m $M/$m; done
        sample sel-release.json $M/release --start 0 --n 500
        [ -f $M/release-nfctok/tokenizer.json ] || $EV/python -I /workspace/v11_tasks.py tokenizer --model $M/release --out $M/release-nfctok
        sample heldout-xx-en-release-nfc.json $M/release --start 500 --n 1497 --reverse
        sample heldout-xx-en-release-nfctok.json $M/release-nfctok --start 500 --n 1497 --reverse --keep-encoding ;;
    train)
        train sdft-s1 1 1
        train l1 1 0 $(flags l1)
        train lowlr 1 0 $(flags lowlr)
        train top 1 0 $(flags top) ;;
    mid)
        mid sdft-mid sdft-s1
        rm -rf "${M:?}/sdft-s1" ;;
    pick)
        if [ ! -s $R/pick.json ]; then
            top=$($PY -I /workspace/v11_pick.py rank $R); log "stage A: $top"
            $PY -I /workspace/v11_pick.py choose $R >/dev/null
        fi
        log "stage B: candidate $($PY -c "import json; print(json.load(open('$R/pick.json'))['candidate'])")" ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json 2>/dev/null) ;;
    decide)
        log decide; $PY -I /workspace/v11_analysis.py $W > $R/analysis-pod.txt
        log "decision: $(tr -d '\n ' < $R/decision.json)" ;;
    stage2)
        [ -s $R/decision.json ] || die "no decision.json: run the decide unit first"
        if $PY -c "import json, sys; sys.exit(0 if json.load(open('$R/decision.json'))['stage2'] else 1)"; then
            c=$($PY -c "import json; print(json.load(open('$R/decision.json'))['candidate'])")
            for s in 2 3; do
                if [ $c = sdft-mid ]; then
                    train sdft-s$s $s 1; mid sdft-mid-s$s sdft-s$s; rm -rf "${M:?}/sdft-s$s"
                else
                    train $c-s$s $s 0 $(flags $c)
                fi
            done
            /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json)
            $PY -I /workspace/v11_analysis.py $W > $R/analysis-pod.txt
        else
            log "stage 2 does not run (preregistered decisions)"
        fi ;;
    *) die "unknown unit: $u" ;;
    esac
done
rm -rf "${M:?}/init-v6" "${M:?}/release-nfctok"
log ALL-DONE
