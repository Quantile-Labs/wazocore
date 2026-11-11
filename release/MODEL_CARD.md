---
license: other
license_name: n-atlas-open-source-research-and-innovation-license
base_model: NCAIR1/N-ATLaS
base_model_relation: finetune
language: [yo, ha, ig, en]
pipeline_tag: text-generation
library_name: gguf
datasets: [google/smol, openai/gsm8k]
metrics: [chrf, accuracy]
tags:
  - yoruba
  - hausa
  - igbo
  - nigerian-languages
  - african-languages
  - low-resource-languages
  - vocabulary-expansion
  - tokenizer
  - translation
  - llama-3
  - llama-cpp
  - gguf
---

# WazoCore

WazoCore is an 8B language model for Yoruba, Hausa and Igbo, adapted from
N-ATLaS with 15,000 additional tokens for the three languages. It encodes
them in roughly a third fewer tokens than its base model, making generation
1.6 to 1.7 times faster on commodity hardware, and is a drop-in replacement
in llama.cpp, LM Studio and OpenAI-compatible stacks.

This release, v13, supersedes [WazoCore v6](https://huggingface.co/quantile-labs/WazoCore-8B-v6-long).
It retains v6's efficiency and translation quality and recovers most of the
reasoning lost in vocabulary adaptation.

| Change from v6 | v6 | v13 |
|---|---|---|
| Mathematical reasoning (AfriMGSM), vs base model | −11.6 | +1.1 (−2.3, +4.4) |
| Natural-language inference (AfriXNLI), vs base model | −7.5 | −5.3 |
| Degenerate repetition, 750 test translations | 5 | 2 |
| Truncated outputs, 750 test translations | 13 | 8 |
| Decomposed Unicode input | efficiency lost | normalised (NFC) |

## Model details

| | |
|---|---|
| Developer | [Quantile Labs](https://quantilelabs.com/) |
| Base model | NCAIR1/N-ATLaS (Llama-3 architecture, 8B parameters) |
| Licensed name | WazoCore Powered by Awarri. |
| Vocabulary | 143,256 tokens: the base model's 128,256, unchanged in id and embedding, plus 15,000 for Yoruba, Hausa and Igbo |
| Release files | `WazoCore-8B.Q4_K_M.gguf` (5.0 GB). Full-precision weights to follow |
| Languages | Yoruba, Hausa, Igbo, English |
| Licence | N-ATLaS Open-Source Research and Innovation License ([details](#licence)) |
| Code and evaluation | [github.com/Quantile-Labs/wazocore](https://github.com/Quantile-Labs/wazocore) |

**Motivation.** The base model inherits the Llama-3 tokenizer, which was
trained predominantly on English. A Yoruba word requires about 2.9 tokens,
against 1.3 for an English word, which raises latency, serving cost and
context consumption for every Nigerian-language request. WazoCore adds
word and subword tokens for the three languages while leaving English
tokenisation effectively unchanged.

## Usage

WazoCore is a standard Llama-3 GGUF, validated in llama.cpp and LM Studio.

```sh
hf download quantile-labs/WazoCore-8B WazoCore-8B.Q4_K_M.gguf --local-dir .
llama-server -m WazoCore-8B.Q4_K_M.gguf -c 4096 --port 8080
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Translate into Hausa: The clinic opens at eight in the morning."}]}'
```

Set the context size explicitly with `-c`; otherwise llama-server allocates
memory for the 131,072-token maximum declared in the file. The model runs
offline on machines with 8 GB of memory or more.

## Intended use

Generation and translation of Yoruba, Hausa and Igbo text on local or
on-premise hardware, as a replacement for the base model where tokenisation
cost constrains latency, cost or context length. Use is subject to the
licence terms below.

## Evaluation

All comparisons are against the base model. Intervals are 95% paired
bootstrap confidence intervals.

| WazoCore vs base model | Yoruba | Hausa | Igbo |
|---|---|---|---|
| Tokens per word | 2.87 → 1.91 | 2.12 → 1.34 | 2.53 → 1.56 |
| Generation speed-up (words/s), laptop CPU | 1.63× | 1.61× | 1.72× |
| ΔchrF++, English → X, Q4_K_M | +3.1 (+2.0, +4.1) | −0.4 (−1.2, +0.4) | +0.3 (−1.3, +1.7) |
| ΔchrF++, English → X, held-out | +2.2 (+1.6, +2.8) | −0.4 (−0.9, 0.0) | +0.1 (−0.5, +0.7) |
| ΔchrF++, X → English, held-out | +0.2 (−0.4, +0.7) | −0.1 (−0.5, +0.4) | +0.9 (+0.5, +1.4) |
| AfriMGSM accuracy (WazoCore / base) | 37.2 / 33.6 | 40.4 / 38.8 | 26.0 / 28.0 |
| AfriXNLI accuracy (WazoCore / base) | 52.7 / 57.7 | 53.0 / 57.5 | 52.5 / 58.8 |

- **Speed.** llama.cpp build 9430, Q4_K_M, Apple M4 CPU, 20 prompts per
  language, paired and interleaved, measured with the v6 release, which
  shares v13's tokenizer and token efficiency. Per-token decoding throughput
  is equivalent for both models; the speed-up derives entirely from shorter
  token sequences.
- **Translation, Q4_K_M.** Both quantised models in llama.cpp, 250 NTREX-128
  sentences per language.
- **Translation, held-out.** 1,497 NTREX-128 sentences per language reserved
  from all training and model-selection decisions, full precision.
- **Reasoning.** IrokoBench, zero-shot, chat template, full precision.
  Pooled over the three languages: AfriMGSM +1.1 (−2.3, +4.4), AfriXNLI
  −5.3 (−7.4, −3.1).

Full protocol and reproduction commands:
[`docs/METHOD.md`](https://github.com/Quantile-Labs/wazocore/blob/main/docs/METHOD.md).

## Training

1. **Vocabulary extension.** 15,000 byte-level BPE merges learned on Yoruba,
   Hausa and Igbo text and appended to the base tokenizer. Merges that would
   split a UTF-8 character are excluded. Input is normalised to Unicode NFC.
2. **Initialisation.** New embeddings are initialised from the mean of their
   constituent subword embeddings.
3. **Fine-tuning.** Approximately 80 minutes on a single A100. Only the new
   embedding rows and four of 32 transformer layers are updated. Original
   vocabulary rows are frozen, and a KL-divergence term anchors predictions
   to the base model.

| Data | Volume |
|---|---|
| Base-model translations of Wikipedia sentences, English to and from each language | ~24,000 requests |
| google/smol professional translations (CC BY 4.0) | 23,662 sentence pairs |
| Self-distilled GSM8K solutions in Yoruba, Hausa, Igbo and English, retained only when the final answer is correct | 5,831 solutions |
| Yoruba, Hausa and Igbo Wikipedia, filtered for orthographic quality | 15% of training tokens |

**Decontamination.** Training items sharing any 13-gram with IrokoBench,
FLORES-200 or NTREX-128 are removed.

## Limitations

| Area | Observation | Recommendation |
|---|---|---|
| Natural-language inference | AfriXNLI about 5 points below the base model | Prefer the base model for inference-heavy workloads |
| Output encoding | 1 of 750 test outputs contained an incomplete UTF-8 character | Retry on decode error |
| Dialogue | Not benchmarked; evaluation covers translation and reasoning | Validate on representative prompts before deployment |

## Licence

WazoCore is a derivative of N-ATLaS and is released under the **N-ATLaS
Open-Source Research and Innovation License (Terms of Use v1.0)**. N-ATLaS is
an initiative of the Federal Ministry of Communications, Innovation and
Digital Economy, powered by Awarri Technologies. Principal terms:

- this attribution must be retained;
- renamed derivatives must end in "Powered by Awarri." and carry the same licence;
- use is limited to 1,000 monthly active end-users without a commercial licence;
- surveillance, discriminatory profiling, disinformation, military use and
  unauthorised personal-data scraping are prohibited.

Consult the full terms on the N-ATLaS model page before use.
