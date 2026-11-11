import unicodedata

from wazocore.vx_core import half, layer_weight, probe_messages, qualifies, strip_marks, token_groups, top_points


def test_strip_marks_levels():
    yo = unicodedata.normalize("NFC", "Ọmọ náà ń ṣeré")
    assert strip_marks(yo, "yo", 1) == unicodedata.normalize("NFC", "Ọmọ naa n ṣere")
    assert strip_marks(yo, "yo", 2) == "Omo naa n sere"
    assert strip_marks("Ụlọ akwụkwọ ṅụ", "ig", 2) == "Ulo akwukwo nu"
    assert strip_marks("ƙasa ɗaya ɓarna", "ha", 1) == "ƙasa ɗaya ɓarna"
    assert strip_marks("ƙasa ɗaya ɓarna", "ha", 2) == "kasa daya barna"
    assert strip_marks(yo, "yo", 0) == yo


def test_layer_weight_touches_only_edge_layers():
    w = lambda n: layer_weight(n, 0.25, 0.75, 32)
    assert w("model.layers.0.self_attn.q_proj.weight") == 0.25
    assert w("model.layers.1.mlp.up_proj.weight") == 0.25
    assert w("model.layers.30.input_layernorm.weight") == 0.75
    assert w("model.layers.31.mlp.down_proj.weight") == 0.75
    assert w("model.layers.2.mlp.down_proj.weight") is None
    assert w("model.embed_tokens.weight") is None and w("lm_head.weight") is None


def test_probe_messages_match_irokobench_prompts():
    nli = probe_messages("nli", {"premise": " A cat. ", "hypothesis": "An animal.", "label": 0})
    assert nli[0]["content"].endswith("Premise: A cat.\nHypothesis: An animal.\n\nIs it entailment, contradiction, or neutral?")
    assert nli[1]["content"] == "entailment"
    assert probe_messages("nli", {"premise": "a", "hypothesis": "b", "label": -1}) is None
    m = probe_messages("math", {"question": "2+3?", "answer": "2+3=<<2+3=5>>5\n#### 5"})
    assert m[0]["content"] == "Question: 2+3?\nAnswer:" and m[1]["content"] == "2+3=5\nThe answer is 5."


def test_half_and_selection_rules():
    assert [half(i) for i in range(3)] == ["selection", "confirmation", "selection"]
    r = {"a": 40.0, "b": 45.0, "c": 45.0, "d": 30.0}
    assert top_points(r, {"a": 2, "b": 0.5, "c": 1.5, "d": 0}) == ["c", "b", "a"]
    rel = {"yo": {"chrf++": 30.0, "chars_per_token": 2.0}}
    assert qualifies({"yo": {"chrf++": 29.6, "chars_per_token": 1.9}}, rel)
    assert not qualifies({"yo": {"chrf++": 29.4, "chars_per_token": 2.0}}, rel)
    assert not qualifies({"yo": {"chrf++": 31.0, "chars_per_token": 1.89}}, rel)


def test_token_groups():
    assert token_groups(b"2019") == {"digit": True, "ascii": True, "invalid_utf8": False, "combining": False}
    assert token_groups("ọ́".encode())["combining"]
    assert token_groups(b"\xe1\xbb")["invalid_utf8"]
