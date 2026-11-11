"""v12 short-answer data, as fixed in docs/METHOD.md, section 4.6. The GPU work between
the steps is `vx_torch generate` with N-ATLaS (scripts/cloud_vx_v12.sh).

  prompts  1,000 careful Wikipedia lines per language (en, yo, ha, ig; seed 0),
           each with one of four short-answer instructions in turn
  mix      keep replies that end within 16 tokens; drop 13-gram overlaps with
           the test sets; write short.jsonl (every kept conversation, for the
           anchor) and chat-v12-short.jsonl (v6 gold rows filling 60/85, v10's
           self-distilled rows 20/85, short answers 5/85 of the conversation
           tokens, in the release's tokenizer)

On the pod, with venv-eval:
  python -I scripts/v12_data.py prompts --wiki WIKI_DIR --dir D
  python -I scripts/v12_data.py mix --dir D --gold GOLD --sdft chat-v10-sdft.jsonl --vx VX --base N_ATLAS \
      --flores F --ntrex N --report short-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v8_aya import IROKO, N, benchmark_ngrams, chat_ids, consumed_share  # noqa: E402
from v10_data import dump, jsonl, lines  # noqa: E402
from wazocore.vx_core import careful_text, fill_share, word_ngrams  # noqa: E402

LANGS = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}
PER_LANG, MAX_REPLY = 1000, 16
TEMPLATES = {
    "number": "Does the following text contain a number? Answer with yes or no only.",
    "language": "Is the following text written in {L}? Answer with yes or no only.",
    "true": "Is the following statement true? Answer with true or false only.",
    "name": "What language is the following text written in? Answer with one word.",
}
ORDER = list(TEMPLATES)
SHORT, SDFT, CHAT = 0.05, 0.20, 0.85


def prompt_for(k: int, lang: str, text: str) -> tuple[str, str, str | None]:
    """Line k of a language: its template (in turn), the prompt and the true
    answer where one exists. The language template names the line's own
    language on its even occurrences and the next language on its odd ones."""
    kind = ORDER[k % len(ORDER)]
    names = list(LANGS)
    if kind == "language":
        own = (k // len(ORDER)) % 2 == 0
        named = lang if own else names[(names.index(lang) + 1) % len(names)]
        return kind, TEMPLATES[kind].format(L=LANGS[named]) + "\n\n" + text, "yes" if own else "no"
    truth = {"number": "yes" if re.search(r"\d", text) else "no", "name": LANGS[lang].lower(), "true": None}[kind]
    return kind, TEMPLATES[kind] + "\n\n" + text, truth


def first_word(reply: str) -> str:
    m = re.match(r"\W*(\w+)", reply.lower())
    return m.group(1) if m else ""


def step_prompts(args) -> int:
    out = []
    for lang in LANGS:
        text = careful_text((args.wiki / f"{lang}.txt").read_text(encoding="utf-8"), lang)
        pool = [l.strip() for l in text.split("\n") if 40 <= len(l.strip()) <= 300]
        random.Random(0).shuffle(pool)
        for k, line in enumerate(pool[:PER_LANG]):
            kind, prompt, truth = prompt_for(k, lang, line)
            out.append({"id": f"s{k}-{lang}", "prompt": prompt, "kind": kind, "lang": lang, "truth": truth})
    args.dir.mkdir(parents=True, exist_ok=True)
    tmp = args.dir / "short.part"
    tmp.write_text("".join(dump(r) + "\n" for r in out), encoding="utf-8")
    tmp.replace(args.dir / "short-prompts.jsonl")
    print(f"{len(out)} short-answer prompts", file=sys.stderr)
    return 0


def step_mix(args) -> int:
    from transformers import AutoTokenizer

    d = args.dir
    prompts = {p["id"]: p for p in jsonl(d / "short-prompts.jsonl")}
    replies = {r["id"]: r for r in jsonl(d / "short-out.jsonl")}
    grams = benchmark_ngrams(args.flores, args.ntrex)
    why: dict = {}
    kept = []
    for pid, p in prompts.items():
        w = why.setdefault(f"{p['kind']} {p['lang']}", Counter())
        r = replies.get(pid)
        if r is None or not r["ended"] or not r["reply"] or r["tokens"] > MAX_REPLY:
            w["not ended within 16 tokens" if r else "no reply"] += 1
            continue
        if next((s for s, g in grams.items() if (word_ngrams(p["prompt"], N) | word_ngrams(r["reply"], N)) & g), None):
            w["overlap"] += 1
            continue
        w["kept"] += 1
        answer = first_word(r["reply"])
        w[f"answer {answer if answer in ('yes', 'no', 'true', 'false') else 'other'}"] += 1
        if p["truth"] is not None:
            w["agrees with truth"] += answer == p["truth"]
        kept.append([{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": r["reply"]}])
    report: dict = {"irokobench_revisions": IROKO, "ngram": N, "per_language": PER_LANG, "by_template_language": {k: dict(v) for k, v in why.items()},
                    "kept": len(kept)}
    tmp = d / "short.part"
    tmp.write_text("".join(dump({"messages": m}) + "\n" for m in kept), encoding="utf-8")
    tmp.replace(d / "short.jsonl")

    vx = AutoTokenizer.from_pretrained(args.vx)
    gold_set = set(lines(args.gold))
    mixed = lines(args.sdft)
    gold = [l for l in mixed if l in gold_set]
    sdft = [l for l in mixed if l not in gold_set]
    cost = lambda l: len(chat_ids(vx, json.loads(l)["messages"]))
    s_tokens = sum(map(cost, sdft))
    gold_costs = [cost(l) for l in gold]
    short_costs = [len(chat_ids(vx, m)) for m in kept]
    # gold : sdft : short = 60 : 20 : 5 of the conversation tokens; the 5 points come from the translation rows
    g_pick, g_rep = fill_share(gold_costs, round(s_tokens * (CHAT - SDFT - SHORT) / SDFT), seed=0)
    s_pick, s_rep = fill_share(short_costs, round(s_tokens * SHORT / SDFT), seed=0)
    assert g_rep == 0, "the gold rows cannot fill their share without repeats"
    report["tokens"] = {"sdft": s_tokens, "gold_used": sum(gold_costs[i] for i in g_pick), "gold_available": sum(gold_costs),
                        "short_used": sum(short_costs[i] for i in s_pick), "short_available": sum(short_costs),
                        "short_rows_used": len(s_pick), "short_repeats": s_rep}
    rows = [(gold[i], "gold") for i in g_pick] + [(l, "sdft") for l in sdft] + [(dump({"messages": kept[i]}), "short") for i in s_pick]
    random.Random(0).shuffle(rows)
    tmp = d / "chat.part"
    tmp.write_text("\n".join(l for l, _ in rows) + "\n", encoding="utf-8")
    tmp.replace(d / "chat-v12-short.jsonl")
    convs = [json.loads(l)["messages"] for l, _ in rows]
    report["consumed"] = {name: {kind: consumed_share(t, convs, [k == kind for _, k in rows]) for kind in ("sdft", "short")}
                          for name, t in (("release_tokenizer", vx), ("original_tokenizer", AutoTokenizer.from_pretrained(args.base)))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--wiki", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    m = sub.add_parser("mix")
    for k in ("dir", "gold", "sdft", "vx", "base", "flores", "ntrex", "report"):
        m.add_argument(f"--{k}", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"prompts": step_prompts, "mix": step_mix}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
