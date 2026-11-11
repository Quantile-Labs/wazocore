#!/usr/bin/env bash
# v10: self-distilled reasoning and chat replies, grouped translation turns,
# a matched control and content-free calibration, as preregistered in
# docs/METHOD.md, section 4.6. Built like v9 for RunPod Community Cloud: every step is
# skipped once its result exists, results are small files the laptop pulls
# as they land (RUN=v10 scripts/cloud_pull.sh), and no weights leave the pod.
# Run after scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v10.sh
# UNITS, in priority order (default all):
#   check    the release rescored against v9 item for item; content-free
#            AfriXNLI for the original and the release; their AfriMMLU
#   data     GSM8K questions translated, maths and chat replies by N-ATLaS,
#            filtered and mixed (v10_data.py); grouped turns for para
#   sdft     train; held-out translation, BPC, IrokoBench, content-free AfriXNLI
#   para     the same for para
#   control  the same for control-sdft (no new tokens)
#   explore  sdft with its bottom layers halfway back to N-ATLaS, R only
#   comet    AfriCOMET on the held-out samples
#   decide   v10_analysis.py on the pod: decision.json
#   stage2   sdft at seeds 1 and 2, only if decision.json says so
# On an out-of-memory error in training lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v10; M=$W; D=$R/data; mkdir -p "$R" "$D"
UNITS=${UNITS:-"check data sdft para control explore comet decide stage2"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}; GEN_BATCH=${GEN_BATCH:-128}
IB=16  # fixed lm-eval batch size, as in v9, so the release scores exactly as there
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex; T=$W/tasks-v10
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=quantile-labs/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
CANON="--anchor canonical --canon-weight 1.0"
R_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l; done)
MMLU_TASKS=$(for l in yor hau ibo eng; do printf 'afrimmlu_direct_%s_prompt_1,' $l; done)
NULL_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1_null,' $l; done)
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: the release was trained on it"
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v10 checks its inputs against v7's"
[ -s $W/results-v8/inputs.sha256 ] || die "upload results/vocab/v8: the Aya prompts are checked against v8's mixture"
[ -d $W/results-v9/iroko-release ] || die "upload results/vocab/v9: the release must score as it did there"
$HF download NCAIR1/N-ATLaS --revision $BASE_REV --local-dir $M/n-atlas --exclude "*.gguf" --exclude "original/*" >/dev/null \
    || die "no access to N-ATLaS: $HF auth login"
if [ ! -f $M/release/.checked ]; then
    $HF download $RELEASE --revision $RELEASE_REV --local-dir $M/release --exclude "*.gguf" >/dev/null
    echo "$RELEASE_SHA  $M/release/model.safetensors" | sha256sum -c --quiet || die "release weights differ from v6-long"
    touch $M/release/.checked
fi
[ -x $EV/lm_eval ] || { python -m venv --system-site-packages /workspace/venv-eval \
    && $EV/pip install -q "lm-eval[hf]==0.4.13" "transformers==5.17.0"; }
[ -s $T/null_v10.py ] || $EV/python -I /workspace/v10_tasks.py --out $T

log setup
# v6's gold data, which must match v7's byte for byte: every arm starts from it
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
null() {  # NAME MODEL_DIR MAIN_RESULTS: content-free AfriXNLI and its bug check
    iroko null-$1 $2 "$NULL_TASKS" --include_path $T
    $PY -I /workspace/v10_analysis.py check-null $R $3 $1 || die "null task for $1 does not score the first item as lm-eval did: see the log"
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
generate() {  # PROMPTS OUT MAX_TOKENS: N-ATLaS's greedy replies, resumable
    local n=$(wc -l < $D/$1) have=0
    [ -s $D/$2 ] && have=$(wc -l < $D/$2)
    [ "$have" -ge "$n" ] && return 0
    log "generate $2 ($have of $n)"
    $V generate --model $M/n-atlas --prompts $D/$1 --out $D/$2 --batch $GEN_BATCH --max-tokens $3
}
init() {  # the expanded, untrained model with the release's initialisation
    [ -f $M/init-v6/.done ] && return 0
    rm -rf "${M:?}/init-v6"; log "expand init-v6"
    $V expand --base $M/n-atlas --vx $W/vx-tok --out $M/init-v6 --out-weights chars --calibrate-input-norm
    touch $M/init-v6/.done
}
evaluate() {  # NAME: held-out translation both ways, BPC, IrokoBench with AfriMMLU, content-free AfriXNLI
    local d way
    for d in en-xx xx-en; do
        [ -s $R/heldout-$d-$1.json ] && continue
        way=; [ $d = xx-en ] && way=--reverse
        log "sample $1 $d"
        $V sample --model $M/$1 --flores $F --ntrex $N --start 500 --n 1497 --batch 64 $way --out $R/heldout-$d-$1.json
    done
    [ -s $R/bpc-$1.json ] || { log "bpc $1"; $V bpc --model $M/$1 --flores $F --sentences 1012 --out $R/bpc-$1.json; }
    iroko $1 $M/$1 "$R_TASKS$MMLU_TASKS"
    null $1 $M/$1 $R
}
train() {  # NAME SEED INIT CHAT KEEP [flags]: the release recipe on CHAT, then evaluate; KEEP=1 keeps the weights
    local name=$1 seed=$2 from=$3 chat=$4 keep=$5; shift 5
    [ -f $R/done-$name ] && { log "$name: done"; return 0; }
    [ $from = n-atlas ] || init
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name"
        $V train --model $M/$from --flores $F --out $M/$name --data $W/data --chat $chat \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
    evaluate $name
    touch $R/done-$name
    [ "$keep" = 1 ] || rm -rf "${M:?}/$name"
}
need() { [ -s $D/$1 ] || die "$1 missing: run the data unit first"; }

for u in $UNITS; do
    case $u in
    check)
        iroko release-v10 $M/release "$R_TASKS"
        same_items $W/results-v9 release $R release-v10 || die "the release scores differently from v9: see the log"
        for m in n-atlas release; do
            null $m $M/$m $W/results-v9
            iroko $m-mmlu $M/$m "$MMLU_TASKS"
        done ;;
    data)
        if [ ! -s $D/chat-v10-sdft.jsonl ]; then
            if [ ! -s $W/chat-v8-inst.jsonl ]; then
                log "aya: v8's filter and mix, for its prompts"
                $EV/python -I /workspace/v8_aya.py --vx $W/vx-tok --base $M/n-atlas --flores $F --ntrex $N \
                    --gold $W/chat-v6-gold.jsonl --out $W/chat-v8-inst.jsonl --report $R/aya-check.json
            fi
            (cd $W && sha256sum chat-v8-inst.jsonl) | cmp -s - $W/results-v8/inputs.sha256 || die "chat-v8-inst.jsonl differs from v8's"
            [ -s $D/gsm8k.jsonl ] || { log "data: prompts"; $EV/python -I /workspace/v10_data.py prompts \
                --gold $W/chat-v6-gold.jsonl --aya-mix $W/chat-v8-inst.jsonl --dir $D; }
            generate translate.jsonl translate-out.jsonl 512
            [ -s $D/answer.jsonl ] || $EV/python -I /workspace/v10_data.py answers --dir $D
            generate answer.jsonl answer-out.jsonl 512
            generate aya.jsonl aya-out.jsonl 512
            log "data: mix"
            $EV/python -I /workspace/v10_data.py mix --dir $D --gold $W/chat-v6-gold.jsonl --vx $W/vx-tok --base $M/n-atlas \
                --flores $F --ntrex $N --out $D/chat-v10-sdft.jsonl --report $R/sdft-report.json
        fi
        [ -s $D/chat-v10-para.jsonl ] || { log "data: para"; $EV/python -I /workspace/v10_data.py para \
            --gold $W/chat-v6-gold.jsonl --vx $W/vx-tok --out $D/chat-v10-para.jsonl --report $R/para-report.json; }
        (cd $D && sha256sum chat-v10-sdft.jsonl chat-v10-para.jsonl) > $R/inputs.now
        [ -s $R/inputs.sha256 ] || cp $R/inputs.now $R/inputs.sha256
        cmp -s $R/inputs.now $R/inputs.sha256 || die "v10 data differs from the first pod's: see $R/inputs.sha256"
        rm $R/inputs.now ;;
    sdft)    need chat-v10-sdft.jsonl; train sdft 0 init-v6 $D/chat-v10-sdft.jsonl 1 $CANON ;;
    para)    need chat-v10-para.jsonl; train para 0 init-v6 $D/chat-v10-para.jsonl 0 $CANON ;;
    control) need chat-v10-sdft.jsonl; train control-sdft 0 n-atlas $D/chat-v10-sdft.jsonl 0 ;;
    explore)
        if [ -f $R/iroko-sdft-b05-t1/done ]; then :
        elif [ -f $M/sdft/.trained ]; then
            [ -f $M/sdft-b05-t1/.done ] || { rm -rf "${M:?}/sdft-b05-t1"; log "interp sdft-b05-t1"
                $V interp --base $M/n-atlas --model $M/sdft --bottom 0.5 --top 1 --out $M/sdft-b05-t1; touch $M/sdft-b05-t1/.done; }
            iroko sdft-b05-t1 $M/sdft-b05-t1 "$R_TASKS"
            rm -rf "${M:?}/sdft-b05-t1"
        else
            log "explore: sdft weights are gone (a new pod); exploratory, skipped"
        fi
        rm -rf "${M:?}/sdft" ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json 2>/dev/null) ;;
    decide)
        log decide; $PY -I /workspace/v10_analysis.py $W > $R/analysis-pod.txt
        log "decision: $(tr -d '\n ' < $R/decision.json)" ;;
    stage2)
        [ -s $R/decision.json ] || die "no decision.json: run the decide unit first"
        if $PY -c "import json, sys; sys.exit(0 if json.load(open('$R/decision.json'))['stage2'] else 1)"; then
            for s in 1 2; do train sdft-s$s $s init-v6 $D/chat-v10-sdft.jsonl 0 $CANON; done
            /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json)
            $PY -I /workspace/v10_analysis.py $W > $R/analysis-pod.txt
        else
            log "stage 2 does not run (preregistered decisions)"
        fi ;;
    *) die "unknown unit: $u" ;;
    esac
done
rm -rf "${M:?}/init-v6"
log ALL-DONE
