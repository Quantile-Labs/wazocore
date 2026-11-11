"""Pure helpers for vocabulary expansion: new-row initialisation, the
byte-level alphabet, data windows, mixing and the learning-rate schedule.
No I/O; the GPU shell is `vx_torch`, the tokenizer builder `vocab_expand`."""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np


def decompose(new_tokens: Sequence[dict], vocab: dict[str, int]) -> dict[int, list[int]]:
    """Original-vocabulary ids each new token is made of, by recursively
    splitting its (left, right) merge until pieces are original tokens."""
    by_token = {t["token"]: t for t in new_tokens}

    def pieces(s: str) -> list[int]:
        if s in vocab and s not in by_token:
            return [vocab[s]]
        t = by_token[s]
        return pieces(t["left"]) + pieces(t["right"])

    return {t["id"]: pieces(t["token"]) for t in new_tokens}


def init_rows(table: np.ndarray, parts: dict[int, list[int]], weights: dict[int, list[float]] | None = None) -> np.ndarray:
    """New rows in id order: the (weighted) mean of each token's piece rows."""
    ids = sorted(parts)
    rows = []
    for i in ids:
        p = np.asarray(parts[i])
        w = np.ones(len(p)) if weights is None else np.asarray(weights[i], dtype=np.float64)
        rows.append((table[p].astype(np.float64) * (w / w.sum())[:, None]).sum(0))
    return np.stack(rows).astype(table.dtype)


def char_lengths(parts: dict[int, list[int]], id_to_bytes, unit: str = "bytes") -> dict[int, list[float]]:
    """Weights for the output head: each piece's length in decoded bytes, or
    with `unit="chars"` in characters (arXiv 2608.03494 weights by characters;
    ị is three bytes but one character, so bytes favour accented pieces)."""
    size = (lambda b: sum((x & 0xC0) != 0x80 for x in b)) if unit == "chars" else len
    return {i: [max(1, size(id_to_bytes(p))) for p in ps] for i, ps in parts.items()}


def norm_calibrated(rows: np.ndarray, target: float) -> np.ndarray:
    """Rows rescaled to L2 norm `target`. A mean of several piece rows is
    shorter than any of them (WazoCore v4: 0.38 against 0.69 for original
    rows), so the model reads a new token as a faint one."""
    norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True)
    return (rows * (target / np.maximum(norms, 1e-12))).astype(rows.dtype)


def mbr_pick(candidates: Sequence[str], utility) -> int:
    """Index of the candidate that agrees most with the others (minimum
    Bayes risk): the highest mean `utility(candidate, other)`. Training on
    these instead of greedy replies lets a student pass its teacher's
    greedy decoding (arXiv 2309.10966). Ties go to the earliest candidate,
    so put the greedy reply first."""
    if len(candidates) < 3:
        return 0
    mean = lambda i: sum(utility(candidates[i], c) for j, c in enumerate(candidates) if j != i) / (len(candidates) - 1)
    return max(range(len(candidates)), key=lambda i: (mean(i), -i))


def smol_pairs(record: dict) -> list[tuple[str, str]]:
    """(English, translation) sentence pairs of one SMOL record (google/smol,
    CC BY 4.0, professional translations made for training): SmolSent holds
    one pair in src/trg, SmolDoc aligned sentence lists in srcs/trgs. Both
    sides come back in NFC, the form the new tokens were learned on."""
    import unicodedata

    pairs = zip(record["srcs"], record["trgs"]) if "srcs" in record else [(record["src"], record["trg"])]
    nfc = lambda t: unicodedata.normalize("NFC", t.strip())
    return [(nfc(a), nfc(b)) for a, b in pairs if a and b and a.strip() and b.strip()]


def merge_table(new_tokens: Sequence[dict], vocab: dict[str, int], size: int) -> tuple[list[int], list[int]]:
    """The new merges as rows of a sparse table: `rights[offsets[a]:offsets[a + 1]]`
    are the ids that may not follow token `a`, because the tokenizer would
    have merged the pair. Text from the tokenizer has no such pair (13 in
    1.2M on Wikipedia), so a model that writes one is spelling a word with
    its pieces and losing the saving."""
    by_left: dict[int, list[int]] = {}
    for t in new_tokens:
        by_left.setdefault(vocab[t["left"]], []).append(vocab[t["right"]])
    offsets, rights = [0], []
    for a in range(size):
        rights += sorted(by_left.get(a, []))
        offsets.append(len(rights))
    return offsets, rights


def bits_per_char(nll_nats: float, chars: int) -> float:
    return nll_nats / math.log(2) / chars


def byte_decoder():
    """GPT-2 byte-level alphabet: printable stand-in character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def chunks(ids: Sequence[int], length: int, bos: int) -> list[list[int]]:
    """Consecutive windows of `length` tokens, each starting with BOS."""
    body = length - 1
    return [[bos, *ids[i:i + body]] for i in range(0, len(ids) - body + 1, body)]


def mixture(per_lang: dict[str, list[list[int]]], shares: dict[str, float], total: int, seed: int) -> list[list[int]]:
    """Deterministic shuffled mix with each language's share of `total`
    chunks (cycling through a language's chunks if it runs short)."""
    rng = random.Random(seed)
    # Largest-remainder allocation: the counts add up to exactly `total`.
    exact = {lang: total * share / sum(shares.values()) for lang, share in shares.items()}
    counts = {lang: int(x) for lang, x in exact.items()}
    for lang in sorted(exact, key=lambda l: exact[l] - counts[l], reverse=True)[: total - sum(counts.values())]:
        counts[lang] += 1
    out: list[list[int]] = []
    for lang, n in counts.items():
        pool = per_lang[lang]
        out += [pool[i % len(pool)] for i in range(n)] if pool else []
    rng.shuffle(out)
    return out


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


ENGLISH_MARKERS = (" the ", " of the ", " and the ", " is a ", " in the ")

# Plain spellings of common Hausa words that are written with hooked letters
# (ɗan, ƙasa, ...) and undotted spellings of common Igbo words (ndị, bụ, ...):
# Wikipedia uses these careless forms often enough for a model to learn them.
CARELESS = {
    "ha": {"dan", "kasa", "kasar", "daya", "wadanda", "kungiyar", "karkashin", "yaki", "bangare", "kwallon", "kafa", "hada", "dari"},
    "ig": {"ndi", "bu", "ozo", "ulo", "ochichi", "muru"},
}
IGBO_DOTS = set("ịọụṅỊỌỤṄ")
YORUBA_MARKS = {"̀", "́", "̣"}  # grave, acute, dot below (after NFD)


def careful_line(line: str, lang: str) -> bool:
    """True if a Wikipedia line looks carefully written in `lang`: not
    English, and using the language's special letters the way its
    references do. Lines failing this taught v3 careless spellings."""
    import unicodedata

    text = unicodedata.normalize("NFC", line.strip())
    if len(text) < 20:
        return False
    if lang == "en":
        return True
    padded = f" {text.lower()} "
    if any(m in padded for m in ENGLISH_MARKERS):
        return False
    words = set("".join(c if c.isalnum() else " " for c in text.lower()).split())
    if words & CARELESS.get(lang, set()):
        return False
    if lang == "ig":
        return sum(c in IGBO_DOTS for c in text) / len(text) >= 0.03
    if lang == "yo":
        return sum(c in YORUBA_MARKS for c in unicodedata.normalize("NFD", text)) / len(text) >= 0.05
    return True


def careful_text(text: str, lang: str) -> str:
    return "\n".join(l for l in text.splitlines() if careful_line(l, lang))


def chunk_align(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int, int, int]] | None:
    """Aligns two tokenisations of the same byte string, given each token's
    length in bytes. Returns chunks (a_start, a_end, b_start, b_end): the
    shortest spans that end where both have a token boundary. Each WazoCore
    token and its original pieces fall in one chunk. None if the two do not
    cover the same number of bytes."""
    if sum(a) != sum(b):
        return None
    chunks, i, j, ia, jb, pa, pb = [], 0, 0, 0, 0, 0, 0
    while i < len(a) or j < len(b):
        if pa <= pb and i < len(a):
            pa += a[i]; i += 1
        else:
            pb += b[j]; j += 1
        if pa == pb:
            chunks.append((ia, i, jb, j))
            ia, jb = i, j
    return chunks


def chunk_sums(values: Sequence[float], spans: Sequence[tuple[int, int]]) -> list[float]:
    """Log-probability of each chunk: the sum of its tokens' log-probabilities."""
    return [sum(values[s:e]) for s, e in spans]


def nfc_rows(rows: Sequence[dict]) -> list[dict]:
    """Rows with `output` and `reference` in Unicode NFC. NTREX-128 Yoruba
    references are stored decomposed (letter + combining accent) while the
    models write composed characters; scored raw, chrF++ counts the two as
    different letters and halves every model's score."""
    import unicodedata

    return [{**r, "output": unicodedata.normalize("NFC", r["output"]), "reference": unicodedata.normalize("NFC", r["reference"])} for r in rows]


def sign_test(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value that paired disagreements (`only_a` items
    where just A shows the event, `only_b` where just B does) are balanced:
    McNemar's exact test."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n)


def looping(text: str, unit: tuple[int, int] = (2, 6), repeats: int = 6) -> bool:
    """A short unit repeated back to back (e.g. 'ereereere…'): the failure
    WazoCore showed more often than the original."""
    import re

    lo, hi = unit
    return re.search(rf"(.{{{lo},{hi}}})\1{{{repeats - 1},}}", text) is not None


def word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Word `n`-grams of `text` after NFC and lower-casing, for the v8
    contamination check. A text shorter than `n` words has none."""
    import unicodedata

    words = unicodedata.normalize("NFC", text).lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def aya_messages(row: dict) -> list[dict] | None:
    """One Aya Dataset row (CohereLabs/aya_dataset, human-written) as a
    user/assistant conversation in NFC, or None if either side is empty."""
    import unicodedata

    q, a = (unicodedata.normalize("NFC", (row.get(k) or "").strip()) for k in ("inputs", "targets"))
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}] if q and a else None


def fill_share(costs: Sequence[int], target: int, seed: int) -> tuple[list[int], int]:
    """Indices of items, in an order fixed by `seed`, whose costs first add
    up to at least `target`, repeating the order if one pass is short.
    Returns the indices and the number of passes started beyond the first."""
    if target <= 0 or not costs or not any(costs):
        return [], 0
    order = list(range(len(costs)))
    random.Random(seed).shuffle(order)
    picked, total = [], 0
    while total < target:
        i = order[len(picked) % len(order)]
        picked.append(i)
        total += costs[i]
    return picked, (len(picked) - 1) // len(order)


# ---- v9 (docs/METHOD.md, section 4.6) ------------------------------------------------------

TONES = {"̀", "́", "̄"}  # grave, acute, macron
DOTS = {"̣", "̇"}  # dot below (ẹ ọ ṣ ị ụ), dot above (ṅ)
HOOKS = str.maketrans("ɓɗƙƴƁƊƘƳ", "bdkyBDKY")


def strip_marks(text: str, lang: str, level: int) -> str:
    """`text` as many people type it: level 1 drops Yoruba and Igbo tone
    marks, level 2 also their dots and Hausa's hooked letters. Hausa does
    not write tone, so its level 1 is the text itself. Returns NFC."""
    import unicodedata

    drop = (TONES if level >= 1 and lang in ("yo", "ig") else set()) | (DOTS if level >= 2 and lang in ("yo", "ig") else set())
    out = "".join(c for c in unicodedata.normalize("NFD", text) if c not in drop)
    out = unicodedata.normalize("NFC", out)
    return out.translate(HOOKS) if level >= 2 and lang == "ha" else out


def layer_weight(name: str, bottom: float, top: float, n_layers: int, edge: int = 2) -> float | None:
    """The weight on the trained model for parameter `name` when moving the
    trained layers back toward the original: `bottom` for the first `edge`
    layers, `top` for the last `edge`, None for every other tensor (kept
    as trained, or never trained)."""
    parts = name.split(".")
    if "layers" not in parts:
        return None
    i = int(parts[parts.index("layers") + 1])
    return bottom if i < edge else top if i >= n_layers - edge else None


NLI_PROMPT = ("Please identify whether the premise entails or contradicts the hypothesis in the following premise "
              "and hypothesis. The answer should be exact entailment, contradiction, or neutral.\n\n"
              "Premise: {premise}\nHypothesis: {hypothesis}\n\nIs it entailment, contradiction, or neutral?")
NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def probe_messages(kind: str, row: dict) -> list[dict] | None:
    """One English training item in IrokoBench's prompt-1 format, as a
    user/assistant conversation: an MNLI pair (`nli`) answered with its
    label, or a GSM8K problem (`math`) answered with its worked solution
    and final number. None for an unlabelled pair."""
    import re

    if kind == "nli":
        label = NLI_LABELS.get(row.get("label"))
        q = NLI_PROMPT.format(premise=row["premise"].strip(), hypothesis=row["hypothesis"].strip())
        return [{"role": "user", "content": q}, {"role": "assistant", "content": label}] if label else None
    work, _, final = row["answer"].partition("####")
    work = re.sub(r"<<[^>]*>>", "", work).strip()
    return [{"role": "user", "content": f"Question: {row['question'].strip()}\nAnswer:"},
            {"role": "assistant", "content": f"{work}\nThe answer is {final.strip()}."}]


def half(doc_id: int) -> str:
    """v9's fixed split of IrokoBench items: even ids choose, odd confirm."""
    return "selection" if doc_id % 2 == 0 else "confirmation"


def top_points(r: dict[str, float], closeness: dict[str, float], k: int = 3) -> list[str]:
    """Stage A: the `k` points with the highest R, ties broken towards the
    larger `closeness` (b + t, nearer the release), then by name."""
    return sorted(r, key=lambda p: (-r[p], -closeness[p], p))[:k]


def qualifies(point: dict[str, dict], release: dict[str, dict], margin: float = 0.5, ratio: float = 0.95) -> bool:
    """Stage B: no language more than `margin` chrF++ below the release, and
    characters per token at least `ratio` of the release's in each.
    Both arguments map language to {"chrf++", "chars_per_token"}."""
    return all(point[l]["chrf++"] >= release[l]["chrf++"] - margin
               and point[l]["chars_per_token"] >= ratio * release[l]["chars_per_token"] for l in release)


def token_groups(raw: bytes) -> dict[str, bool]:
    """What a new token is made of, for the v9 tokenizer audit."""
    import unicodedata

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    return {"digit": any(48 <= b <= 57 for b in raw), "ascii": all(b < 128 for b in raw), "invalid_utf8": text is None,
            "combining": text is not None and any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text))}


# ---- v10 (docs/METHOD.md, section 4.6) -----------------------------------------------------

NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def last_number(text: str) -> float | None:
    """The last number written in digits in `text`, commas dropped; None if
    there is none. The rule lm-eval's flexible-extract follows on AfriMGSM."""
    import re

    found = re.findall(NUMBER, text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_final(answer: str) -> float:
    """GSM8K's final answer, the number after '####'."""
    return float(answer.rpartition("####")[2].strip().replace(",", ""))


def math_correct(reply: str, gold: float) -> bool:
    n = last_number(reply)
    return n is not None and abs(n - gold) < 1e-6


def one_sentence(reply: str) -> bool:
    """At most one of '.', '!' and '?' in the reply: v9's sign that the
    model answers a problem the way it answers a translation request."""
    return sum(reply.count(c) for c in ".!?") <= 1


def loops(reply: str, n: int = 8, times: int = 4, min_words: int = 40) -> bool:
    """An `n`-word sequence at least `times` times in a reply of at least
    `min_words` words."""
    from collections import Counter

    words = reply.split()
    if len(words) < min_words:
        return False
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1)).most_common(1)[0][1] >= times


STOPWORDS = {
    "en": {"the", "and", "is", "of", "to", "in", "that", "he", "she", "it", "has"},
    "yo": {"ni", "ti", "àti", "sí", "ó", "ní", "tí", "kí", "jẹ́", "rẹ̀", "náà"},
    "ha": {"da", "na", "ta", "ya", "ba", "ne", "ce", "kuma", "shi", "wannan", "zai"},
    "ig": {"na", "nke", "ọ", "ka", "bụ", "ya", "ndị", "dị", "ga", "a", "mere"},
}


def reply_language(text: str) -> str:
    """The language whose common words make up most of `text`, for
    reporting the share of replies in each language. 'other' if none."""
    import unicodedata

    words = unicodedata.normalize("NFC", text).lower().split()
    hits = {l: sum(w.strip(".,!?;:") in s for w in words) for l, s in STOPWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else "other"


def split_request(prompt: str) -> tuple[str, str]:
    """A translation request as (instruction, source): the instruction is
    everything before the first blank line."""
    head, sep, src = prompt.partition("\n\n")
    if not sep or not head.startswith("Translate the following"):
        raise ValueError(f"not a translation request: {prompt[:60]!r}")
    return head, src


def group_pairs(rows: Sequence[tuple[str, str, str]], cost, k_min: int = 3, k_max: int = 8,
                max_tokens: int = 480, seed: int = 0) -> list[list[tuple[str, str, str]]]:
    """v10's `para` turns: translation pairs (instruction, source, reply),
    kept apart by instruction, in an order fixed by `seed`, joined into
    groups of `k_min` to `k_max` (drawn per group, same seed). A group
    closes early when one more pair would make `cost(group)` exceed
    `max_tokens`; a single pair is never dropped. Every pair is used once."""
    rng = random.Random(seed)
    by_head: dict[str, list[tuple[str, str, str]]] = {}
    for r in rows:
        by_head.setdefault(r[0], []).append(r)
    groups = []
    for head in sorted(by_head):
        pool = by_head[head][:]
        rng.shuffle(pool)
        i = 0
        while i < len(pool):
            want, g = rng.randint(k_min, k_max), [pool[i]]
            i += 1
            while len(g) < want and i < len(pool) and cost(g + [pool[i]]) <= max_tokens:
                g.append(pool[i])
                i += 1
            groups.append(g)
    return groups


def joined(group: Sequence[tuple[str, str, str]]) -> list[dict]:
    """One `para` group as a user/assistant conversation."""
    head = group[0][0]
    assert all(g[0] == head for g in group), "a group mixes instructions"
    return [{"role": "user", "content": f"{head}\n\n" + " ".join(g[1] for g in group)},
            {"role": "assistant", "content": " ".join(g[2] for g in group)}]


def calibrated_choice(scores: Sequence[float], prior: Sequence[float]) -> int:
    """The option with the highest log-probability once the content-free
    log-probability of each option is subtracted (Calibrate Before Use)."""
    return max(range(len(scores)), key=lambda i: (scores[i] - prior[i], -i))


# ---- v11 (docs/METHOD.md, section 4.6) -----------------------------------------------------

LABEL_WORDS = ("entailment", "neutral", "contradiction")


def written_label(reply: str) -> str | None:
    """The first of the three AfriXNLI labels that occurs as a word in a
    generated reply, case ignored; None if none does."""
    import re

    m = re.search(r"\b(" + "|".join(LABEL_WORDS) + r")\b", reply.lower())
    return m.group(1) if m else None


def nfc_strings(value):
    """`value` with every string in it, however nested, in NFC. Any mapping
    counts, since `datasets` hands `map` lazy rows that are not dicts."""
    import unicodedata
    from collections.abc import Mapping

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (list, tuple)):
        return [nfc_strings(v) for v in value]
    if isinstance(value, Mapping):
        return {k: nfc_strings(v) for k, v in value.items()}
    return value


def train_layers(n_layers: int, edge: int = 2, only: Sequence[int] | None = None) -> list[int]:
    """The transformer layers `vx_torch train` updates: the bottom and top
    `edge` layers (the release recipe), or exactly `only`."""
    if only is not None:
        assert all(0 <= i < n_layers for i in only) and len(set(only)) == len(only), f"bad layers {only}"
        return sorted(only)
    return [*range(edge), *range(n_layers - edge, n_layers)]


def lr_scales(layers: Sequence[int], n_layers: int, bottom: float = 1.0) -> dict[int, float]:
    """Learning-rate factor per trained layer: `bottom` for layers in the
    lower half, 1 for the rest."""
    return {i: (bottom if i < n_layers // 2 else 1.0) for i in layers}
