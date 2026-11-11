#!/usr/bin/env bash
# v9: the trained layers moved back toward N-ATLaS, a format probe and
# unmarked input, as preregistered in docs/METHOD.md, section 4.6. Built like v8 for
# RunPod Community Cloud: every step is skipped once its result exists,
# results are small files the laptop pulls as they land
# (RUN=v9 scripts/cloud_pull.sh), and no weights leave the pod. Run after
# scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v9.sh
# UNITS, in priority order (default all):
#   audit     weight drift of the release, tokenizer audit (CPU)
#   control   control-s0 retrained with v7's command; drift; R for it, the original and the release
#   grid      16 interpolated models, R tasks each, built and deleted one at a time
#   pick      stages A and B: rank on the selection half, NTREX 0-499 for the top three
#   confirm   stage C: the full v8 evaluation of the candidate
#   unmarked  question 3 on the original, the release and control-s0
#   probe     question 2: format probe on the original, the release and control-s0
#   comet     AfriCOMET on the new samples
# On an out-of-memory error in `control` lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v9; M=$W; mkdir -p "$R"
UNITS=${UNITS:-"audit control grid pick confirm unmarked probe comet"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
IB=16  # fixed lm-eval batch size, so a model scored twice gives the same numbers
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex; T=$W/tasks-v9
BASE_REV=e294476928aca9030e924ca27bb8e085e8581273
RELEASE=quantile-labs/WazoCore-8B-v6-long; RELEASE_REV=be50a5df45de4abaecd1bd51231987fe24bacf1f
RELEASE_SHA=b4cc2a6891529ff1827ee66699eb82aafbb23588d6f0988e74577644168db381
R_TASKS=$(for l in yor hau ibo eng; do printf 'afrixnli_%s_prompt_1,afrimgsm_%s_prompt_1,' $l $l; done)
MMLU_TASKS=$(for l in yor hau ibo eng; do printf 'afrimmlu_direct_%s_prompt_1,' $l; done)
UNMARKED_TASKS=$(for l in yor hau ibo; do printf 'afrixnli_%s_prompt_1_unmarked,' $l; done)
log() { echo "== $(date +%H:%M:%S) $*"; }
die() { echo "!! $*" >&2; exit 1; }

log preflight  # fail in the first minute, not the fifth hour
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 79000 ] || die "need an 80 GB GPU, found $mem MiB"
[ "$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9)" -ge 150 ] || die "need 150 GB free in $W"
[ -s $W/chat-wiki.jsonl ] || die "upload data/runs/cloud/chat-wiki.jsonl: control-s0 is trained on it"
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v9 checks its inputs against v7's"
[ -s $W/results-v7/heldout-en-xx-release.json ] || die "upload results/vocab/v7: the audit reads the release's replies"
$HF download NCAIR1/N-ATLaS --revision $BASE_REV --local-dir $M/n-atlas --exclude "*.gguf" --exclude "original/*" >/dev/null \
    || die "no access to N-ATLaS: $HF auth login"
if [ ! -f $M/release/.checked ]; then
    $HF download $RELEASE --revision $RELEASE_REV --local-dir $M/release --exclude "*.gguf" >/dev/null
    echo "$RELEASE_SHA  $M/release/model.safetensors" | sha256sum -c --quiet || die "release weights differ from v6-long"
    touch $M/release/.checked
fi
[ -x $EV/lm_eval ] || { python -m venv --system-site-packages /workspace/venv-eval \
    && $EV/pip install -q "lm-eval[hf]==0.4.13" "transformers==5.17.0"; }
[ -s $T/strip_v9.py ] || $EV/python -I /workspace/v9_tasks.py --out $T --wazocore /workspace/wazocore

log data
# v6's gold data, which must match v7's byte for byte: control-s0 is retrained on it
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
num() { case $1 in 0) echo 0 ;; 05) echo 0.5 ;; 1) echo 1 ;; *) die "bad grid value $1" ;; esac; }
interp() {  # NAME TRAINED B T: N-ATLaS's edge layers blended into TRAINED's
    [ -f $M/$1/.done ] && return 0
    rm -rf "${M:?}/$1"; log "interp $1"
    $V interp --base $M/n-atlas --model $M/$2 --bottom $(num $3) --top $(num $4) --out $M/$1
    touch $M/$1/.done
}
point() {  # NAME: build a release grid point from its name, rel-bB-tT
    local b=${1#rel-b}; b=${b%%-t*}
    interp $1 release $b ${1##*-t}
}
same_items() {  # A B: every R item of B scored exactly as in A
    $PY -I - "$R" "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
def items(p):
    out = {}
    for f in r.glob(f"iroko-{p}/*/samples_*.jsonl"):
        for x in map(json.loads, f.read_text().splitlines()):
            if x["filter"] in ("none", "flexible-extract"):
                out[(f.name.split("_prompt_1")[0], x["doc_id"])] = x.get("acc", x.get("exact_match"))
    return out
a, b = items(sys.argv[2]), items(sys.argv[3])
diff = [k for k in b if a.get(k) != b[k]]
print(f"{sys.argv[3]} against {sys.argv[2]}: {len(diff)} of {len(b)} items differ")
sys.exit(1 if diff or not b or set(a) != set(b) else 0)
EOF
}

for u in $UNITS; do
    case $u in
    audit)
        [ -s $R/audit.json ] || $PY -I /workspace/v9_audit.py --vx $W/vx-tok --base $M/n-atlas --flores $F \
            --replies $W/results-v7/heldout-en-xx-release.json --out $R/audit.json
        [ -s $R/drift-release.json ] || $V drift --base $M/n-atlas --model $M/release --out $R/drift-release.json ;;
    control)
        if [ ! -f $M/control-s0/.trained ]; then
            rm -rf "${M:?}/control-s0"; log "train control-s0"
            $V train --model $M/n-atlas --flores $F --out $M/control-s0 --data $W/data --chat $W/chat-v6-gold.jsonl \
                --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
                --seed 0 --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH
            cp $M/control-s0/log.jsonl $R/train-control-s0-log.jsonl
            touch $M/control-s0/.trained
        fi
        [ -s $R/drift-control-s0.json ] || $V drift --base $M/n-atlas --model $M/control-s0 --out $R/drift-control-s0.json
        for m in n-atlas release control-s0; do iroko $m $M/$m "$R_TASKS"; done ;;
    grid)
        for who in rel ctl; do
            src=release; [ $who = ctl ] && src=control-s0
            for b in 0 05 1; do for t in 0 05 1; do
                [ $b$t = 11 ] && continue
                p=$who-b$b-t$t
                [ -f $R/iroko-$p/done ] && continue
                interp $p $src $b $t
                if [ $p = ctl-b0-t0 ]; then  # N-ATLaS exactly: the bug check the plan fixes
                    $V drift --base $M/n-atlas --model $M/$p --out $R/drift-$p.json
                    $PY -c "import json, sys; sys.exit(json.load(open('$R/drift-$p.json'))['max'] != 0)" \
                        || die "ctl-b0-t0 weights differ from N-ATLaS: interpolation bug"
                fi
                iroko $p $M/$p "$R_TASKS"
                [ $p = ctl-b0-t0 ] && { same_items n-atlas $p || die "ctl-b0-t0 scores differ from N-ATLaS's: see the log"; }
                rm -rf "${M:?}/$p"
            done; done
        done ;;
    pick)
        if [ ! -s $R/pick.json ]; then
            top=$($PY -I /workspace/v9_pick.py rank $R); log "stage A: $top"
            sample sel-release.json $M/release --start 0 --n 500
            for p in $top; do
                [ $p = release ] && continue
                point $p; sample sel-$p.json $M/$p --start 0 --n 500; rm -rf "${M:?}/$p"
            done
            $PY -I /workspace/v9_pick.py choose $R >/dev/null
        fi
        log "stage B: candidate $($PY -c "import json; print(json.load(open('$R/pick.json'))['candidate'])")" ;;
    confirm)
        c=$($PY -c "import json; print(json.load(open('$R/pick.json'))['candidate'])")
        case $c in
        None)    log "no candidate: stage C does not run (preregistered)" ;;
        release) log "the release itself ranks first: nothing to confirm" ;;
        *)
            [ -f $R/done-cand ] || {
                point $c
                for d in en-xx xx-en; do
                    way=; [ $d = xx-en ] && way=--reverse
                    sample heldout-$d-$c.json $M/$c --start 500 --n 1497 $way
                done
                [ -s $R/bpc-$c.json ] || $V bpc --model $M/$c --flores $F --sentences 1012 --out $R/bpc-$c.json
                iroko $c $M/$c "$R_TASKS"
                for m in $c release n-atlas; do iroko $m-mmlu $M/$m "$MMLU_TASKS"; done
                rm -rf "${M:?}/$c"; touch $R/done-cand
            } ;;
        esac ;;
    unmarked)
        for m in n-atlas release control-s0; do
            for s in 0 1 2; do sample unmarked-l$s-xx-en-$m.json $M/$m --start 500 --n 1497 --reverse --strip $s; done
            iroko unmarked-$m $M/$m "$UNMARKED_TASKS" --include_path $T
        done ;;
    probe)
        [ -s $W/probe.jsonl ] || $EV/python -I /workspace/v9_probe.py --out $W/probe.jsonl --report $R/probe-report.json
        for m in n-atlas release control-s0; do
            [ -f $R/iroko-probe-$m/done ] && continue
            [ -f $M/probe-$m/.done ] || { rm -rf "${M:?}/probe-$m"; log "probe $m"
                $V probe --model $M/$m --data $W/probe.jsonl --out $M/probe-$m; touch $M/probe-$m/.done; }
            cp $M/probe-$m/log.jsonl $R/probe-$m-log.jsonl
            iroko probe-$m $M/probe-$m "$R_TASKS"
            rm -rf "${M:?}/probe-$m"
        done ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json $R/unmarked-*.json 2>/dev/null) ;;
    *) die "unknown unit: $u" ;;
    esac
done
log ALL-DONE
