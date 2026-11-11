#!/usr/bin/env bash
# v12: a KL anchor on N-ATLaS's own short answers, on v11's lowlr recipe;
# contradiction recall; NFC on fresh lines. As preregistered in
# docs/METHOD.md, section 4.6. Built like v11 for RunPod Community Cloud: every step is
# skipped once its result exists, results are small files the laptop pulls
# as they land (RUN=v12 scripts/cloud_pull.sh), and no weights leave the pod.
# Run after scripts/cloud_vx_upload.sh and `hf auth login`:
#   bash /workspace/cloud_vx_v12.sh
# UNITS, in priority order (default all):
#   check   the release rescored against v9; generative and domain-conditional
#           AfriXNLI for the original and the release; NFC samples on NTREX
#           0-499 and FLORES devtest; NTREX 0-499 for the release
#   data    short-answer prompts, N-ATLaS's replies, filter and mix (v12_data.py)
#   train   base, short, anchor: lowlr's recipe at seed 2, then evaluated
#   pick    stages A and B (v12_pick.py, margin 1.0, 0.5 reported)
#   comet   AfriCOMET on the held-out and NFC samples
#   decide  v12_analysis.py on the pod: decision.json
#   stage2  the candidate at seeds 3 and 4, only if decision.json says so
# On an out-of-memory error in training lower KL_BATCH to 2, then BATCH to 4 with ACCUM 4.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${W:-/workspace/work}; R=$W/results-v12; M=$W; D=$R/data; mkdir -p "$R" "$D"
UNITS=${UNITS:-"check data train pick comet decide stage2"}
KL_BATCH=${KL_BATCH:-4}; BATCH=${BATCH:-8}; ACCUM=${ACCUM:-2}
IB=16  # fixed lm-eval batch size, as in v9 to v11
PY=/workspace/venv/bin/python; HF=/workspace/venv/bin/hf; EV=/workspace/venv-eval/bin
F=$W/data/flores200_dataset; N=$W/ntrex; T=$W/tasks-v12; SDFT=$W/results-v10/data/chat-v10-sdft.jsonl
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
[ -s $W/results-v7/inputs.sha256 ] || die "upload results/vocab/v7: v12 checks its inputs against v7's"
[ -d $W/results-v8/iroko-long-s1 ] || die "upload results/vocab/v8: stage 2 compares with release seed 1"
[ -d $W/results-v9/iroko-release ] || die "upload results/vocab/v9: the release must score as it did there"
[ -d $W/results-v11/iroko-lowlr ] || die "upload results/vocab/v11: base is compared with lowlr"
[ -s $SDFT ] || die "upload results/vocab/v10 with its data/ folder: v12 trains on chat-v10-sdft.jsonl"
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
    $PY -I /workspace/v12_analysis.py check-dc $R $3 $1 || die "domain-conditional task for $1 does not score the first item as lm-eval did"
}
nfc() {  # MODEL: X -> English with sources as given, tokenizer as released and with an NFC normalizer
    [ -f $M/$1-nfctok/tokenizer.json ] || $EV/python -I /workspace/v11_tasks.py tokenizer --model $M/$1 --out $M/$1-nfctok
    sample nfc-ntrex-raw-$1.json $M/$1 --ntrex $N --start 0 --n 500 --reverse --keep-encoding
    sample nfc-ntrex-nfc-$1.json $M/$1-nfctok --ntrex $N --start 0 --n 500 --reverse --keep-encoding
    sample nfc-flores-raw-$1.json $M/$1 --start 0 --n 1012 --reverse --keep-encoding
    sample nfc-flores-nfc-$1.json $M/$1-nfctok --start 0 --n 1012 --reverse --keep-encoding
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
train() {  # NAME SEED CHAT [flags]: lowlr's recipe at SEED on CHAT, then evaluate and free the disk
    local name=$1 seed=$2 chat=$3; shift 3
    [ -f $R/done-$name ] && { log "$name: done"; return 0; }
    init
    if [ ! -f $M/$name/.trained ]; then
        rm -rf "${M:?}/$name"; log "train $name $*"
        $V train --model $M/init-v6 --flores $F --out $M/$name --data $W/data --chat $chat \
            --chat-share 0.85 --careful-wiki --batch $BATCH --accum $ACCUM --eval-every 100 --tokens 12000000 \
            --seed $seed --teacher $M/n-atlas --kl-weight 1.0 --kl-batch $KL_BATCH $RECIPE "$@"
        cp $M/$name/log.jsonl $R/train-$name-log.jsonl
        touch $M/$name/.trained
    fi
    evaluate $name
    touch $R/done-$name
    rm -rf "${M:?}/$name"
}
arm() {  # MODEL SEED [NAME]: the arm's data and flags
    local name=${3:-$1}
    case $1 in
    base)   train $name $2 $SDFT ;;
    short)  train $name $2 $D/chat-v12-short.jsonl ;;
    anchor) train $name $2 $SDFT --anchor-extra $D/short.jsonl ;;
    *) die "unknown arm $1" ;;
    esac
}
need() { [ -s $D/$1 ] || die "$1 missing: run the data unit first"; }

for u in $UNITS; do
    case $u in
    check)
        iroko release-v12 $M/release "$R_TASKS"
        same_items $W/results-v9 release $R release-v12 || die "the release scores differently from v9: see the log"
        for m in n-atlas release; do extras $m $M/$m $W/results-v9; nfc $m; done
        sample sel-release.json $M/release --ntrex $N --start 0 --n 500
        [ -s $R/tokenizer-check.json ] || $PY -I - $M/release $M/release-nfctok $R/nfc-ntrex-raw-release.json $R/nfc-flores-raw-release.json $R/tokenizer-check.json <<'EOF'
import json, sys, unicodedata
from transformers import AutoTokenizer
plain, normal = AutoTokenizer.from_pretrained(sys.argv[1]), AutoTokenizer.from_pretrained(sys.argv[2])
out = {}
for f in sys.argv[3:5]:
    for r in json.load(open(f))["rows"]:
        k = f"{f.split('/')[-1]} {r['lang']}"
        o = out.setdefault(k, {"sentences": 0, "not_nfc": 0, "mismatch": 0})
        o["sentences"] += 1
        o["not_nfc"] += unicodedata.normalize("NFC", r["source"]) != r["source"]
        o["mismatch"] += normal.encode(r["source"]) != plain.encode(unicodedata.normalize("NFC", r["source"]))
json.dump(out, open(sys.argv[5], "w"), indent=1)
print(f"tokenizer check: {out}")
sys.exit(1 if any(o["mismatch"] for o in out.values()) else 0)
EOF
        ;;
    data)
        if [ ! -s $D/chat-v12-short.jsonl ]; then
            [ -s $D/short-prompts.jsonl ] || $EV/python -I /workspace/v12_data.py prompts --wiki $W/data/wiki --dir $D
            n=$(wc -l < $D/short-prompts.jsonl); have=0; [ -s $D/short-out.jsonl ] && have=$(wc -l < $D/short-out.jsonl)
            [ "$have" -ge "$n" ] || { log "generate short-out.jsonl ($have of $n)"
                $V generate --model $M/n-atlas --prompts $D/short-prompts.jsonl --out $D/short-out.jsonl --batch 128 --max-tokens 17; }
            log "data: mix"
            $EV/python -I /workspace/v12_data.py mix --dir $D --gold $W/chat-v6-gold.jsonl --sdft $SDFT --vx $W/vx-tok \
                --base $M/n-atlas --flores $F --ntrex $N --report $R/short-report.json
        fi
        (cd $D && sha256sum short.jsonl chat-v12-short.jsonl) > $R/inputs.now
        [ -s $R/inputs.sha256 ] || cp $R/inputs.now $R/inputs.sha256
        cmp -s $R/inputs.now $R/inputs.sha256 || die "v12 data differs from the first pod's: see $R/inputs.sha256"
        rm $R/inputs.now ;;
    train)
        need short.jsonl; need chat-v12-short.jsonl
        for m in base short anchor; do arm $m 2; done ;;
    pick)
        if [ ! -s $R/pick.json ]; then
            log "stage A: $($PY -I /workspace/v12_pick.py rank $R)"
            $PY -I /workspace/v12_pick.py choose $R >/dev/null
        fi
        log "stage B: candidate $($PY -c "import json; p = json.load(open('$R/pick.json')); print(p['candidate'], '(under 0.5:', p['sensitivity']['candidate'], ')')")" ;;
    comet)
        [ -x /workspace/venv-comet/bin/python ] || { python -m venv --system-site-packages /workspace/venv-comet \
            && /workspace/venv-comet/bin/pip install -q "unbabel-comet==2.2.7"; }
        log comet; /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json $R/nfc-*.json 2>/dev/null) ;;
    decide)
        log decide; $PY -I /workspace/v12_analysis.py $W > $R/analysis-pod.txt
        log "decision: $(tr -d '\n ' < $R/decision.json)" ;;
    stage2)
        [ -s $R/decision.json ] || die "no decision.json: run the decide unit first"
        if $PY -c "import json, sys; sys.exit(0 if json.load(open('$R/decision.json'))['stage2'] else 1)"; then
            c=$($PY -c "import json; print(json.load(open('$R/decision.json'))['candidate'])")
            for s in 3 4; do arm $c $s $c-s$s; done
            /workspace/venv-comet/bin/python -I /workspace/africomet.py $R $(ls $R/heldout-*.json)
            $PY -I /workspace/v12_analysis.py $W > $R/analysis-pod.txt
        else
            log "stage 2 does not run (preregistered decisions)"
        fi ;;
    *) die "unknown unit: $u" ;;
    esac
done
rm -rf "${M:?}/init-v6" "${M:?}/release-nfctok" "${M:?}/n-atlas-nfctok"
log ALL-DONE
