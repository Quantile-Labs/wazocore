"""v10 data, as fixed in docs/METHOD.md, section 4.6. Four steps; the GPU work between
them is `vx_torch generate` with N-ATLaS (see scripts/cloud_vx_v10.sh).

  prompts  GSM8K train questions (seed 0) and translation requests into
           yo/ha/ig; the prompts of v8's Aya rows
  answers  the questions as user turns: English as written, the others as
           N-ATLaS translated them; no "Question:"/"Answer:" frame
  mix      keep correct, ended, non-looping maths replies and ended,
           non-looping chat replies; drop 13-gram overlaps with the test
           sets; fill 20/85 of the conversation tokens, half maths (equal
           per language) and half chat; shuffle into v6's gold data
  para     v6's gold translation pairs grouped into turns of 3 to 8

On the pod, with venv-eval:
  python -I scripts/v10_data.py prompts --gold GOLD --aya-mix chat-v8-inst.jsonl --dir D
  python -I scripts/v10_data.py answers --dir D
  python -I scripts/v10_data.py mix --dir D --gold GOLD --vx VX --base N_ATLAS --flores F --ntrex N \
      --out chat-v10-sdft.jsonl --report sdft-report.json
  python -I scripts/v10_data.py para --gold GOLD --vx VX --out chat-v10-para.jsonl --report para-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v8_aya import IROKO, N, benchmark_ngrams, chat_ids, consumed_share  # noqa: E402
from wazocore.vx_core import (fill_share, group_pairs, gsm8k_final, joined, looping, loops, math_correct,  # noqa: E402
                              reply_language, split_request, word_ngrams)

GSM8K = ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866")
QUESTIONS = 4000  # first questions of the shuffled order, per language; a generation cap, not a choice
LANGS = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}
SDFT, CHAT = 0.20, 0.85  # self-distilled share of training; conversation share


def lines(path: Path) -> list[str]:
    """Lines split at newlines only: str.splitlines also breaks at U+2028,
    which GSM8K questions contain."""
    return [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in lines(path)]


BREAKS = "\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"  # where str.splitlines breaks, besides \n


def dump(obj) -> str:
    """One JSON line that every reader splits the same way: the characters
    str.splitlines breaks at are written as escapes (still valid JSON), so
    the trainer, which uses splitlines, reads the file unchanged."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.translate({ord(c): f"\\u{ord(c):04x}" for c in BREAKS})


def write_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text("".join(dump(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def translation_request(text: str, lang: str) -> str:
    """The training data's English -> X request (vx_torch.translation_prompts)."""
    return f"Translate the following English text into {LANGS[lang]}. Reply with the translation only.\n\n{text}"


def clean(r: dict | None) -> bool:
    return bool(r and r["ended"] and r["reply"] and not loops(r["reply"]) and not looping(r["reply"]))


def step_prompts(args) -> int:
    from datasets import load_dataset

    rows = list(load_dataset(GSM8K[0], GSM8K[1], revision=GSM8K[2], split="train"))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    qs = [{"n": k, "row": i, "question": rows[i]["question"].strip(), "gold": gsm8k_final(rows[i]["answer"])}
          for k, i in enumerate(order[:QUESTIONS])]
    args.dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dir / "gsm8k.jsonl", qs)
    write_jsonl(args.dir / "translate.jsonl", ({"id": f"t{q['n']}-{l}", "prompt": translation_request(q["question"], l)}
                                               for l in ("yo", "ha", "ig") for q in qs))
    gold = set(lines(args.gold))
    aya, seen = [], set()
    for line in lines(args.aya_mix):
        if line in gold:
            continue
        p = json.loads(line)["messages"][0]["content"]
        if p not in seen:
            seen.add(p)
            aya.append({"id": f"aya{len(aya)}", "prompt": p})
    write_jsonl(args.dir / "aya.jsonl", aya)
    print(f"{len(qs)} questions, {3 * len(qs)} translation requests, {len(aya)} Aya prompts", file=sys.stderr)
    return 0


def step_answers(args) -> int:
    qs = jsonl(args.dir / "gsm8k.jsonl")
    tr = {r["id"]: r for r in jsonl(args.dir / "translate-out.jsonl")}
    out, missing = [], Counter()
    for l in LANGS:
        for q in qs:
            if l == "en":
                text = q["question"]
            else:
                r = tr.get(f"t{q['n']}-{l}")
                if not clean(r):
                    missing[l] += 1
                    continue
                text = r["reply"]
            out.append({"id": f"a{q['n']}-{l}", "prompt": text})
    write_jsonl(args.dir / "answer.jsonl", out)
    print(f"{len(out)} questions to answer; translations unusable: {dict(missing)}", file=sys.stderr)
    return 0


def step_mix(args) -> int:
    from transformers import AutoTokenizer

    d = args.dir
    report: dict = {"gsm8k": {"dataset": GSM8K[0], "config": GSM8K[1], "revision": GSM8K[2], "questions": QUESTIONS},
                    "irokobench_revisions": IROKO, "ngram": N}
    qs = {q["n"]: q for q in jsonl(d / "gsm8k.jsonl")}
    prompts = {p["id"]: p["prompt"] for p in jsonl(d / "answer.jsonl")} | {p["id"]: p["prompt"] for p in jsonl(d / "aya.jsonl")}
    replies = {r["id"]: r for r in jsonl(d / "answer-out.jsonl")} | {r["id"]: r for r in jsonl(d / "aya-out.jsonl")}
    grams = benchmark_ngrams(args.flores, args.ntrex)

    def keep(pid: str, kind: str, why: Counter) -> list[dict] | None:
        r = replies.get(pid)
        if r is None:
            why["no reply"] += 1
        elif not r["ended"]:
            why["not ended"] += 1
        elif not r["reply"]:
            why["empty"] += 1
        elif loops(r["reply"]) or looping(r["reply"]):
            why["loops"] += 1
        elif kind == "math" and not math_correct(r["reply"], qs[int(pid[1:].split("-")[0])]["gold"]):
            why["wrong"] += 1
        else:
            mine = word_ngrams(prompts[pid], N) | word_ngrams(r["reply"], N)
            hit = next((src for src, g in grams.items() if mine & g), None)
            if hit:
                why[f"overlap {hit}"] += 1
                return None
            why["kept"] += 1
            return [{"role": "user", "content": prompts[pid]}, {"role": "assistant", "content": r["reply"]}]
        return None

    math: dict[str, list[list[dict]]] = {}
    report["math"] = {}
    for l in LANGS:
        why: Counter = Counter()
        ids = sorted((p for p in prompts if p.startswith("a") and p.endswith(f"-{l}")), key=lambda p: int(p[1:].split("-")[0]))
        math[l] = [m for p in ids if (m := keep(p, "math", why))]
        report["math"][l] = {"asked": len(ids), **why,
                             "reply_language": dict(Counter(reply_language(m[1]["content"]) for m in math[l]))}
    why = Counter()
    chat = [m for p in sorted((p for p in prompts if p.startswith("aya")), key=lambda p: int(p[3:])) if (m := keep(p, "chat", why))]
    report["chat"] = {"asked": sum(p.startswith("aya") for p in prompts), **why,
                      "reply_language": dict(Counter(reply_language(m[1]["content"]) for m in chat))}

    vx = AutoTokenizer.from_pretrained(args.vx)
    gold = lines(args.gold)
    gold_tokens = sum(len(chat_ids(vx, json.loads(l)["messages"])) for l in gold)
    target = round(gold_tokens * SDFT / (CHAT - SDFT))
    per_lang = target // 2 // len(LANGS)
    used: list[tuple[str, list[dict]]] = []
    tokens: dict = {"gold": gold_tokens, "target": target, "math_target_per_language": per_lang}
    short = 0
    for l in LANGS:  # in GSM8K's shuffled order, no repeats
        got = 0
        for m in math[l]:
            if got >= per_lang:
                break
            used.append((f"math-{l}", m))
            got += len(chat_ids(vx, m))
        tokens[f"math_{l}"] = got
        short += max(0, per_lang - got)
    chat_target = target - sum(tokens[f"math_{l}"] for l in LANGS)
    costs = [len(chat_ids(vx, m)) for m in chat]
    picked, repeats = fill_share(costs, chat_target, seed=0)
    used += [("chat", chat[i]) for i in picked]
    tokens |= {"math_shortfall_moved_to_chat": short, "chat_target": chat_target, "chat_used": sum(costs[i] for i in picked),
               "chat_rows_used": len(picked), "chat_repeats": repeats}
    report["tokens"] = tokens
    report["used"] = dict(Counter(k for k, _ in used))

    mixed = [(l, False) for l in gold] + [(dump({"messages": m}), True) for _, m in used]
    random.Random(0).shuffle(mixed)
    tmp = args.out.with_suffix(".part")
    tmp.write_text("\n".join(l for l, _ in mixed) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    convs, flags = [json.loads(l)["messages"] for l, _ in mixed], [f for _, f in mixed]
    report["consumed"] = {"release_tokenizer": consumed_share(vx, convs, flags),
                          "original_tokenizer": consumed_share(AutoTokenizer.from_pretrained(args.base), convs, flags)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), file=sys.stderr)
    return 0


def step_para(args) -> int:
    from transformers import AutoTokenizer

    vx = AutoTokenizer.from_pretrained(args.vx)
    gold = [json.loads(l)["messages"] for l in lines(args.gold)]
    assert all(len(m) == 2 for m in gold), "gold has a conversation that is not one request and one reply"
    pairs = [(*split_request(m[0]["content"]), m[1]["content"]) for m in gold]
    groups = group_pairs(pairs, cost=lambda g: len(chat_ids(vx, joined(g))), k_min=3, k_max=8, max_tokens=480, seed=0)
    random.Random(0).shuffle(groups)
    convs = [joined(g) for g in groups]
    write_jsonl(args.out, ({"messages": c} for c in convs))
    sizes = Counter(len(g) for g in groups)
    report = {"pairs": len(pairs), "turns": len(groups), "pairs_per_turn": dict(sorted(sizes.items())),
              "mean_pairs_per_turn": round(len(pairs) / len(groups), 2),
              "tokens_gold": sum(len(chat_ids(vx, m)) for m in gold), "tokens_para": sum(len(chat_ids(vx, c)) for c in convs),
              "by_instruction": dict(Counter(g[0][0] for g in groups))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--aya-mix", type=Path, required=True, help="v8's chat-v8-inst.jsonl, hash checked")
    p.add_argument("--dir", type=Path, required=True)
    a = sub.add_parser("answers")
    a.add_argument("--dir", type=Path, required=True)
    m = sub.add_parser("mix")
    for k in ("dir", "gold", "vx", "base", "flores", "ntrex", "out", "report"):
        m.add_argument(f"--{k}", type=Path, required=True)
    g = sub.add_parser("para")
    for k in ("gold", "vx", "out", "report"):
        g.add_argument(f"--{k}", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"prompts": step_prompts, "answers": step_answers, "mix": step_mix, "para": step_para}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
