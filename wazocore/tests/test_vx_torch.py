"""The PyTorch expansion recipe on the tiny random Llama-3 fixture (CPU)."""

from pathlib import Path

import numpy as np
import pytest

TINY = Path(__file__).resolve().parent / "fixtures/tiny"


@pytest.fixture()
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)


def test_expand_appends_initialised_rows_and_keeps_originals(tiny):
    from wazocore.vx_torch import expand_model

    emb0 = tiny.get_input_embeddings().weight.detach().clone()
    head0 = tiny.get_output_embeddings().weight.detach().clone()
    parts = {256: [3, 5], 257: [7, 8, 9]}
    m = expand_model(tiny, parts, {256: [1.0, 3.0], 257: [1.0, 1.0, 1.0]})
    emb, head = m.get_input_embeddings().weight, m.get_output_embeddings().weight
    assert emb.shape[0] == head.shape[0] == 258 and m.config.vocab_size == 258
    assert np.allclose(emb[:256].detach(), emb0)
    assert np.allclose(emb[256].detach(), (emb0[3] + emb0[5]) / 2, atol=1e-6)
    assert np.allclose(head[256].detach(), 0.25 * head0[3] + 0.75 * head0[5], atol=1e-6)


def test_training_moves_only_new_rows_and_chosen_layers(tiny):
    import torch

    from wazocore.vx_torch import expand_model, lm_loss, set_trainable

    m = expand_model(tiny, {256: [3, 5], 257: [7, 8]}, {256: [1.0, 1.0], 257: [1.0, 1.0]})
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    params = set_trainable(m, 256, [0])
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    batch = torch.tensor([[0, 256, 257, 10, 256, 11, 257, 12]] * 2)
    losses = []
    for _ in range(5):
        loss = lm_loss(m, batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    after = dict(m.named_parameters())
    emb = "model.embed_tokens.weight"
    assert torch.equal(after[emb][:256], before[emb][:256]), "original rows must not move"
    assert not torch.equal(after[emb][256:], before[emb][256:]), "new rows must train"
    assert torch.equal(after["lm_head.weight"][:256], before["lm_head.weight"][:256])
    assert not torch.equal(after["model.layers.0.mlp.up_proj.weight"], before["model.layers.0.mlp.up_proj.weight"])
    assert torch.equal(after["model.layers.1.mlp.up_proj.weight"], before["model.layers.1.mlp.up_proj.weight"])


def test_generation_stats_counts_new_tokens_and_stop():
    from wazocore.vx_torch import generation_stats

    s = generation_stats([5, 130000, 130001, 9, 2], canonical=2, first_new=128256, eos={2})
    assert s == {"stopped": True, "tokens": 4, "new_tokens": 2, "canonical_tokens": 2, "vs_canonical": 2.0}
    assert not generation_stats([5, 6], 2, 128256, {2})["stopped"]


def test_windows_pack_without_overlap_or_padding():
    from wazocore.vx_torch import windows

    assert windows(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5]]
    assert windows([1, 2], 3) == []


def test_prompts_name_the_language_and_carry_the_sentence():
    from wazocore.vx_torch import reverse_prompts, translation_prompts

    assert translation_prompts(["Hi."], "Hausa")[0].startswith("Translate the following English text into Hausa.")
    assert reverse_prompts(["Sannu."], "Hausa")[0].endswith("\n\nSannu.")


def test_score_rows_counts_per_language():
    from wazocore.vx_torch import score_rows

    rows = [{"lang": "yo", "output": "ab�", "reference": "ab", "stopped": True, "tokens": 2, "new_tokens": 1},
            {"lang": "ha", "output": "xyz", "reference": "xyz", "stopped": False, "tokens": 3, "new_tokens": 0}]
    r = score_rows(rows, lambda hyp, ref: float(len(hyp)))
    assert r["yo"] == {"n": 1, "chrf++": 1.0, "stopped": 1, "tokens": 2, "chars": 3, "new_tokens": 1, "invalid_utf8": 1}
    assert r["ha"]["stopped"] == 0 and r["ig"]["n"] == 0


def test_wiki_sentences_are_deterministic_and_bounded():
    from wazocore.vx_torch import wiki_sentences

    text = "Short. " + ". ".join(f"This is sentence number {i} about a topic" for i in range(50)) + "\nAnother line that is long enough to keep"
    a = wiki_sentences(text, 10, seed=1)
    assert a == wiki_sentences(text, 10, seed=1) and a != wiki_sentences(text, 10, seed=2)
    assert len(a) == 10 and all(40 <= len(x) <= 301 and x.endswith(".") for x in a)
    assert "Short." not in a


def test_paired_bootstrap_sees_a_consistent_improvement():
    from wazocore.vx_torch import paired_bootstrap

    refs = ["x"] * 20
    score = lambda hyp, ref: sum(h == r for h, r in zip(hyp, ref)) / len(ref) * 100
    r = paired_bootstrap(["y"] * 10 + ["x"] * 10, ["x"] * 20, refs, score, rounds=200)
    assert r["delta"] == 50.0 and r["ci95"][0] > 0 and r["p_worse"] == 0.0


def test_pooled_bootstrap_averages_runs_and_bounds_the_loss():
    from wazocore.vx_torch import pooled_bootstrap

    score = lambda s: float(s.sum(0)[0] / s.sum(0)[1] * 100)  # hits / sentences
    base = np.array([[1.0, 1.0]] * 40)
    same, worse = base.copy(), np.array([[1.0, 1.0]] * 20 + [[0.0, 1.0]] * 20)
    r = pooled_bootstrap(base, [same, worse], [(0, 30), (30, 40)], score, margin=1.0, rounds=300)
    assert r["delta"] == -25.0 and r["ci95"][1] < 0 and not r["non_inferior"] and r["n"] == 40 and r["runs"] == 2
    assert pooled_bootstrap(base, [same], [(0, 40)], score, margin=1.0, rounds=300)["non_inferior"]


def test_expand_can_calibrate_new_input_rows_to_their_pieces_norm(tiny):
    from wazocore.vx_torch import expand_model

    emb0 = tiny.get_input_embeddings().weight.detach().clone()
    head0 = tiny.get_output_embeddings().weight.detach().clone()
    m = expand_model(tiny, {256: [3, 5], 257: [7, 8, 9]}, {256: [1.0, 1.0], 257: [1.0, 1.0, 1.0]}, calibrate=True)
    target = float(np.median(np.linalg.norm(emb0[[3, 5, 7, 8, 9]].numpy(), axis=1)))
    new_rows = m.get_input_embeddings().weight[256:].detach()
    assert np.allclose(np.linalg.norm(new_rows.numpy(), axis=1), target, rtol=1e-5)
    mean = (emb0[3] + emb0[5]) / 2
    assert np.allclose(new_rows[0] / new_rows[0].norm(), mean / mean.norm(), atol=1e-5), "direction is unchanged"
    assert np.allclose(m.get_output_embeddings().weight[256].detach(), (head0[3] + head0[5]) / 2, atol=1e-6), "output rows are not calibrated"


# Toy vocabulary for the canonical-path losses: a=0, b=1, c=2 are original; ab=3 is new (a + b).
def _toy():
    torch = pytest.importorskip("torch")
    from wazocore.vx_core import merge_table

    offsets, rights = merge_table([{"token": "ab", "left": "a", "right": "b"}], {"a": 0, "b": 1, "c": 2, "ab": 3}, 4)
    return torch, torch.tensor(offsets), torch.tensor(rights)


def test_canon_penalty_charges_only_the_piece_that_completes_a_merge():
    torch, offsets, rights = _toy()
    from wazocore.vx_torch import canon_penalty

    logits = torch.log(torch.tensor([[0.1, 0.5, 0.2, 0.2], [0.1, 0.5, 0.2, 0.2]]))
    after_a = canon_penalty(logits, torch.tensor([0, 2]), torch.tensor([2, 1]), offsets, rights)  # b after a is charged; b after c is free
    assert float(after_a) == pytest.approx(-np.log(0.5) / 2, rel=1e-5)
    skipped = canon_penalty(logits[:1], torch.tensor([0]), torch.tensor([1]), offsets, rights)  # the text itself has a, b: not charged
    assert float(skipped) == 0.0


def test_projected_kl_counts_a_new_token_for_its_first_piece_and_debiases_the_teacher():
    torch, offsets, rights = _toy()
    from wazocore.vx_torch import projected_kl

    first_piece = torch.tensor([0])  # ab starts with a
    teacher = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    # After c: the student may split the teacher's 0.6 on `a` between a and ab at will.
    student = torch.log(torch.tensor([[0.2, 0.3, 0.1, 0.4]]))
    assert float(projected_kl(student, teacher, first_piece, torch.tensor([2]), offsets, rights)) == pytest.approx(0.0, abs=1e-6)
    # After a: b would complete the merge, so the teacher's 0.3 on b is removed and the rest renormalised.
    student = torch.log(torch.tensor([[0.6 / 0.7 - 0.2, 1e-9, 0.1 / 0.7, 0.2]]))
    assert float(projected_kl(student, teacher, first_piece, torch.tensor([0]), offsets, rights)) == pytest.approx(0.0, abs=1e-5)
    wrong = torch.log(torch.tensor([[0.3, 0.3, 0.1, 0.3]]))
    assert float(projected_kl(wrong, teacher, first_piece, torch.tensor([0]), offsets, rights)) > 0.01


def test_teacher_reads_the_same_text_in_pieces(tiny):
    import torch

    from wazocore.vx_torch import expand_model, piece_stream, teacher_at

    pieces = [[i] for i in range(256)] + [[3, 5], [7, 8, 9]]
    flat, ends = piece_stream([[0, 256, 10, 257], [0, 11, 12, 13]], pieces)
    assert flat == [[0, 3, 5, 10, 7, 8, 9], [0, 11, 12, 13]] and ends == [[0, 2, 3, 6], [0, 1, 2, 3]]
    got = teacher_at(tiny, [[0, 256, 10, 257], [0, 11, 12, 13]], pieces, 0, "cpu")
    with torch.no_grad():
        direct = tiny(input_ids=torch.tensor([[0, 3, 5, 10, 7, 8, 9]])).logits[0]
        short = tiny(input_ids=torch.tensor([[0, 11, 12, 13]])).logits[0]
    assert got.shape == (2, 3, 256)
    assert torch.allclose(got[0], direct[[0, 2, 3]], atol=1e-5) and torch.allclose(got[1], short[:3], atol=1e-5), "padding must not change earlier positions"


def test_canonical_processor_blocks_the_piece_spelling():
    torch, offsets, rights = _toy()
    from wazocore.vx_torch import canonical_processor

    scores = canonical_processor(offsets, rights)(torch.tensor([[2, 0], [0, 2]]), torch.zeros(2, 4))
    assert scores[0].tolist() == [0.0, float("-inf"), 0.0, 0.0] and scores[1].tolist() == [0.0] * 4


def test_wiki_sentences_can_drop_english():
    from wazocore.vx_torch import wiki_sentences

    text = "This is one of the English sentences in the dump\nỌ natara mmachibido iwu nke abụọ maka afọ asatọ"
    assert wiki_sentences(text, 5, 0, drop_english=True) == ["Ọ natara mmachibido iwu nke abụọ maka afọ asatọ."]
    assert len(wiki_sentences(text, 5, 0)) == 2


def test_anchor_kl_is_zero_for_matching_old_logits_and_ignores_new_ones():
    torch = pytest.importorskip("torch")
    from wazocore.vx_torch import anchor_kl

    teacher = torch.randn(2, 3, 5)
    student = torch.cat([teacher, torch.full((2, 3, 2), 9.0)], dim=-1)  # 2 extra (new) tokens
    assert anchor_kl(student, teacher, 5).item() < 1e-6
    other = torch.cat([torch.randn(2, 3, 5), torch.zeros(2, 3, 2)], dim=-1)
    assert anchor_kl(other, teacher, 5).item() > 0


def test_test_set_reads_ntrex_and_flores(tmp_path):
    from wazocore.vx_torch import test_set

    (tmp_path / "newstest2019-src.eng.txt").write_text("a\nb\nc\n", encoding="utf-8")
    (tmp_path / "newstest2019-ref.yor.txt").write_text("A\nB\nC\n", encoding="utf-8")
    assert test_set("ntrex", tmp_path, "yo", 2) == (["a", "b"], ["A", "B"])
    (tmp_path / "devtest").mkdir()
    (tmp_path / "devtest" / "eng_Latn.devtest").write_text("x\n", encoding="utf-8")
    (tmp_path / "devtest" / "hau_Latn.devtest").write_text("X\n", encoding="utf-8")
    assert test_set("flores", tmp_path, "ha", 5) == (["x"], ["X"])


def test_expand_without_weights_takes_the_plain_mean(tiny):
    from wazocore.vx_torch import expand_model

    head0 = tiny.get_output_embeddings().weight.detach().clone()
    head = expand_model(tiny, {256: [3, 5]}, None).get_output_embeddings().weight
    assert np.allclose(head[256].detach(), (head0[3] + head0[5]) / 2, atol=1e-6)


def test_test_set_takes_a_slice(tmp_path):
    from wazocore.vx_torch import test_set

    (tmp_path / "newstest2019-src.eng.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    (tmp_path / "newstest2019-ref.ibo.txt").write_text("A\nB\nC\nD\n", encoding="utf-8")
    assert test_set("ntrex", tmp_path, "ig", 10, start=2) == (["c", "d"], ["C", "D"])


def test_interp_moves_edge_layers_and_drift_measures_it(tiny, tmp_path):
    import json

    import torch
    from safetensors.torch import load_file

    from wazocore.vx_torch import main

    base, trained, out = tmp_path / "base", tmp_path / "trained", tmp_path / "out"
    tiny.to(torch.bfloat16).save_pretrained(base)
    with torch.no_grad():
        for p in tiny.parameters():
            p.add_(1.0)
    tiny.save_pretrained(trained)
    (trained / "new_tokens.json").write_text("{}")
    assert main(["interp", "--base", str(base), "--model", str(trained), "--out", str(out),
                 "--edge-layers", "1", "--bottom", "0", "--top", "0.5"]) == 0
    o, t, m = (load_file(d / "model.safetensors") for d in (base, trained, out))
    k0, k1 = "model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight"
    assert torch.equal(m[k0], o[k0])
    assert torch.allclose(m[k1].float(), (o[k1].float() + t[k1].float()) / 2, atol=1e-2)
    assert torch.equal(m["model.embed_tokens.weight"], t["model.embed_tokens.weight"])
    assert (out / "new_tokens.json").read_text() == "{}"
    assert main(["drift", "--base", str(base), "--model", str(base), "--out", str(tmp_path / "d.json"), "--edge-layers", "1"]) == 0
    assert json.loads((tmp_path / "d.json").read_text())["max"] == 0.0


def _char_model(dst):
    """The tiny fixture with a character-level tokenizer and a chat template."""
    import shutil

    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    shutil.copytree(TINY, dst)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!>"
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{c: 4 + i for i, c in enumerate(chars)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    t.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}>{% endif %}"
    tok.save_pretrained(dst)
    return dst


def test_generate_resumes_after_a_cut_line(tmp_path):
    pytest.importorskip("torch")
    import json

    from wazocore.vx_torch import main

    model = _char_model(tmp_path / "m")
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"q{i}", "prompt": p}) + "\n" for i, p in enumerate(["two plus two", "hi", "what is 7 times 6"])))
    out = tmp_path / "out.jsonl"
    run = lambda: main(["--device", "cpu", "generate", "--model", str(model), "--prompts", str(prompts), "--out", str(out), "--batch", "2", "--max-tokens", "6"])
    assert run() == 0
    first = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert sorted(first) == ["q0", "q1", "q2"]
    assert all(set(r) == {"id", "reply", "ended", "tokens"} and r["tokens"] <= 6 for r in first.values())
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:2]) + "\n" + lines[2][:5])  # a host lost mid-write
    assert run() == 0
    again = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["id"] for r in again) == ["q0", "q1", "q2"]
    assert {r["id"]: r for r in again} == first
    small = tmp_path / "small.jsonl"  # a token budget that forces one prompt per batch gives the same greedy replies
    assert main(["--device", "cpu", "generate", "--model", str(model), "--prompts", str(prompts), "--out", str(small),
                 "--batch", "2", "--max-tokens", "6", "--token-budget", "1"]) == 0
    assert {r["id"]: r["reply"] for r in map(json.loads, small.read_text().splitlines())} == {k: v["reply"] for k, v in first.items()}


def _trained_layers(tmp_path, *flags):
    """Which layers' weights a short training run changed, on the tiny
    model (2 layers) with the character tokenizer and no new tokens."""
    import torch
    from safetensors.torch import load_file

    from wazocore.vx_torch import main

    model = _char_model(tmp_path / "m")
    (tmp_path / "d" / "wiki").mkdir(parents=True)
    for lang in ("en", "yo", "ha", "ig"):
        (tmp_path / "d" / "wiki" / f"{lang}.txt").write_text("the cat sat on the mat and ate. " * 40)
    out = tmp_path / "out"
    assert main(["--device", "cpu", "train", "--model", str(model), "--data", str(tmp_path / "d"), "--out", str(out),
                 "--tokens", "96", "--seq", "16", "--batch", "2", "--lr", "1e-2", "--eval-every", "0", "--edge-layers", "1", *flags]) == 0
    before, after = load_file(model / "model.safetensors"), load_file(out / "model.safetensors")
    # frozen layers are loaded in bf16, so compare against the original rounded the same way
    same = lambda k: torch.equal(before[k].to(after[k].dtype), after[k])
    moved = lambda i: not all(same(k) for k in before if f"layers.{i}." in k)
    return [i for i in (0, 1) if moved(i)]


def test_train_layers_flag_freezes_the_rest(tmp_path):
    pytest.importorskip("torch")
    assert _trained_layers(tmp_path / "a") == [0, 1]
    assert _trained_layers(tmp_path / "b", "--layers", "1") == [1]


def test_bottom_lr_scale_zero_keeps_the_bottom_layer(tmp_path):
    pytest.importorskip("torch")
    assert _trained_layers(tmp_path, "--bottom-lr-scale", "0") == [1]


def test_extra_anchor_is_zero_for_the_teacher_itself_and_trains_the_student(tiny):
    import copy

    import torch

    from wazocore.vx_core import merge_table
    from wazocore.vx_torch import expand_model, extra_anchor_kl

    rows = [[0, 3, 5, 10, 11], [0, 12, 13, 14, 15]]
    # The original model against itself, no new tokens: nothing to anchor.
    offsets, rights = (torch.tensor(x, dtype=torch.long) for x in merge_table([], {}, 256))
    same = extra_anchor_kl(tiny, tiny, rows, [[i] for i in range(256)], 0, torch.tensor([], dtype=torch.long), offsets, rights)
    assert float(same) == pytest.approx(0.0, abs=1e-5)
    # An expanded student reading a new token: the anchor is positive and reaches the student's weights only.
    student = expand_model(copy.deepcopy(tiny), {256: [3, 5]}, {256: [1.0, 1.0]})
    vocab = {**{f"t{i}": i for i in range(256)}, "t3t5": 256}
    offsets, rights = map(torch.tensor, merge_table([{"token": "t3t5", "left": "t3", "right": "t5"}], vocab, 257))
    tiny.requires_grad_(False)
    kl = extra_anchor_kl(student, tiny, [[0, 256, 10, 11], [0, 12, 13, 14]], [[i] for i in range(256)] + [[3, 5]], 0,
                         torch.tensor([3]), offsets, rights)
    assert torch.isfinite(kl) and float(kl) > 0
    kl.backward()
    assert student.get_output_embeddings().weight.grad is not None
    assert all(p.grad is None for p in tiny.parameters())
