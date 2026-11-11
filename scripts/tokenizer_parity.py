"""Tokenisation parity: llama.cpp reading the shipped GGUF against the
reference tokenizer.json, on 500 FLORES-200 devtest sentences per language.

    python scripts/tokenizer_parity.py GGUF_SHA256_PREFIX [GGUF TOKENIZER_JSON] > OUT.json
"""
import json, sys, unicodedata
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from llama_cpp import Llama

GGUF = Path(sys.argv[2] if len(sys.argv) > 2 else "data/models/gguf-v6/WazoCore-v6-long.Q4_K_M.gguf")
REF = sys.argv[3] if len(sys.argv) > 3 else None  # a local tokenizer.json in place of the release's
LANGS = {"yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn", "en": "eng_Latn"}
PER_LANG = 500

ref = Tokenizer.from_file(REF or hf_hub_download("quantile-labs/WazoCore-8B-v6-long", "tokenizer.json"))
gguf = Llama(str(GGUF), vocab_only=True, verbose=False)
rows, mismatches = {}, []
for lang, code in LANGS.items():
    lines = Path(f"data/flores200_dataset/devtest/{code}.devtest").read_text(encoding="utf8").splitlines()[:PER_LANG]
    same = tokens = 0
    for i, text in enumerate(lines):
        text = unicodedata.normalize("NFC", text)
        a = ref.encode(text, add_special_tokens=False).ids
        b = gguf.tokenize(text.encode("utf8"), add_bos=False, special=False)
        tokens += len(a)
        if a == b: same += 1
        else: mismatches.append({"lang": lang, "index": i, "reference": len(a), "llama_cpp": len(b)})
    rows[lang] = {"texts": len(lines), "identical": same, "reference_tokens": tokens}
out = {"gguf": GGUF.name, "gguf_sha256_16": sys.argv[1], "reference": REF or "tokenizer.json from quantile-labs/WazoCore-8B-v6-long",
       "vocab_size_reference": ref.get_vocab_size(), "vocab_size_gguf": gguf.n_vocab(),
       "source": "FLORES-200 devtest, first 500 sentences per language, NFC", "by_language": rows,
       "texts": sum(r["texts"] for r in rows.values()), "identical": sum(r["identical"] for r in rows.values()), "mismatches": mismatches[:20]}
print(json.dumps(out, indent=1, ensure_ascii=False))
