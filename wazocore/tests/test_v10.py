"""v10 pure functions (docs/METHOD.md, section 4.6)."""

import pytest

from wazocore.vx_core import (calibrated_choice, group_pairs, gsm8k_final, joined, last_number, loops, math_correct,
                              one_sentence, reply_language, split_request)


def test_last_number_takes_the_last_and_drops_commas():
    assert last_number("3 + 4 = 7 eggs, so 16 - 7 = 9, and 9 * $2 = $1,018.") == 1018
    assert last_number("o bụ 2.5") == 2.5
    assert last_number("-3 degrees") == -3
    assert last_number("iri abụọ na ise") is None


def test_math_correct_against_the_gsm8k_answer():
    gold = gsm8k_final("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs.\n#### 18")
    assert gold == 18
    assert math_correct("She makes 9 * 2 = $18 every day.", gold)
    assert not math_correct("She makes $18 a day, 9 eggs.", gold)
    assert not math_correct("Iko nri iri abụọ na ise.", gold)
    assert gsm8k_final("... #### 1,200") == 1200


def test_one_sentence():
    assert one_sentence("Iko nri iri abụọ na ise.")
    assert one_sentence("no stop at all")
    assert not one_sentence("First 3 + 4 = 7. Then 16 - 7 = 9. So $18.")
    assert not one_sentence("Really? Yes!")


def test_loops_needs_a_long_reply_and_four_repeats():
    unit = "ó ní ẹyin mẹ́ta ó sì ń tà wọ́n"  # 8 words
    assert loops(" ".join([unit] * 5))
    assert not loops(" ".join([unit] * 3) + " " + " ".join(f"w{i}" for i in range(20)))
    assert not loops(" ".join([unit] * 4)[:60])  # under 40 words
    assert not loops(" ".join(f"w{i}" for i in range(100)))


def test_reply_language():
    assert reply_language("She sells the rest of the eggs and it is $18.") == "en"
    assert reply_language("Ó ní ẹyin mẹ́ta, ó sì ta èyí tí ó kù ní ọjà.") == "yo"
    assert reply_language("Ta sayar da sauran kuma ta samu dala 18.") == "ha"
    assert reply_language("Ọ bụ nke ndị ahụ mere ya.") == "ig"
    assert reply_language("18") == "other"


def test_split_request():
    head, src = split_request("Translate the following English text into Hausa. Reply with the translation only.\n\nA b.\n\nC.")
    assert head.endswith("only.") and src == "A b.\n\nC."
    with pytest.raises(ValueError):
        split_request("Question: 2 + 2\nAnswer:")


def test_group_pairs_uses_every_pair_once_and_never_mixes_instructions():
    rows = [(h, f"s{h}{i}", f"r{h}{i}") for h in ("A", "B") for i in range(23)]
    groups = group_pairs(rows, cost=len, k_min=3, k_max=8, max_tokens=10**9, seed=0)
    flat = [p for g in groups for p in g]
    assert sorted(flat) == sorted(rows)
    assert all(len({p[0] for p in g}) == 1 for g in groups)
    assert all(len(g) <= 8 for g in groups)
    assert sum(len(g) >= 3 for g in groups) >= len(groups) - 2  # only a language's last group can be short
    assert groups == group_pairs(rows, cost=len, seed=0)
    assert groups != group_pairs(rows, cost=len, seed=1)


def test_group_pairs_closes_a_group_at_the_token_limit():
    rows = [("A", f"s{i}", f"r{i}") for i in range(10)]
    groups = group_pairs(rows, cost=len, k_min=8, k_max=8, max_tokens=2)
    assert all(len(g) <= 2 for g in groups) and sum(map(len, groups)) == 10
    lone = group_pairs(rows[:1], cost=lambda g: 10**6, max_tokens=1)
    assert lone == [[rows[0]]]


def test_joined():
    m = joined([("Translate the following English text into Igbo.", "A.", "Ạ."), ("Translate the following English text into Igbo.", "B.", "Ḅ.")])
    assert m[0]["content"] == "Translate the following English text into Igbo.\n\nA. B."
    assert m[1] == {"role": "assistant", "content": "Ạ. Ḅ."}
    with pytest.raises(AssertionError):
        joined([("x", "a", "b"), ("y", "c", "d")])


def test_calibrated_choice_removes_the_label_preference():
    scores, prior = [-1.0, -3.0, -2.5], [-0.5, -4.0, -3.0]
    assert max(range(3), key=lambda i: scores[i]) == 0
    assert calibrated_choice(scores, prior) == 1
    assert calibrated_choice([-1.0, -1.0, -1.0], [0, 0, 0]) == 0
