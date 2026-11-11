"""v11 pure functions (docs/METHOD.md, section 4.6)."""

import unicodedata

import pytest

from wazocore.vx_core import lr_scales, nfc_strings, train_layers, written_label


def test_written_label_takes_the_first_label_word():
    assert written_label("Entailment.") == "entailment"
    assert written_label("The answer is contradiction, not neutral.") == "contradiction"
    assert written_label("It is neutral") == "neutral"
    assert written_label("Ìbáṣepọ̀ náà jẹ́ entailment") == "entailment"
    assert written_label("contradictions everywhere") is None  # whole words only
    assert written_label("Mi ò mọ̀") is None


def test_nfc_strings_reaches_nested_strings_only():
    nfd = unicodedata.normalize("NFD", "ọ̀rẹ́")
    doc = {"premise": nfd, "choices": [nfd, "a"], "label": 2, "meta": {"x": nfd}}
    out = nfc_strings(doc)
    assert out["premise"] == unicodedata.normalize("NFC", nfd) != nfd
    assert out["choices"][0] == out["meta"]["x"] == out["premise"]
    assert out["label"] == 2 and doc["premise"] == nfd


def test_nfc_strings_accepts_any_mapping():
    from collections.abc import Mapping

    class Lazy(Mapping):  # like the rows datasets passes to map
        def __init__(self, d):
            self.d = d

        def __getitem__(self, k):
            return self.d[k]

        def __iter__(self):
            return iter(self.d)

        def __len__(self):
            return len(self.d)

    nfd = unicodedata.normalize("NFD", "ń")
    assert nfc_strings(Lazy({"premise": nfd}))["premise"] == "ń"


def test_train_layers():
    assert train_layers(32) == [0, 1, 30, 31]
    assert train_layers(32, only=[31, 1, 30]) == [1, 30, 31]
    assert train_layers(32, only=[30, 31]) == [30, 31]
    with pytest.raises(AssertionError):
        train_layers(32, only=[32])
    with pytest.raises(AssertionError):
        train_layers(32, only=[1, 1])


def test_lr_scales():
    assert lr_scales([0, 1, 30, 31], 32, 0.25) == {0: 0.25, 1: 0.25, 30: 1.0, 31: 1.0}
    assert lr_scales([0, 1, 30, 31], 32) == {0: 1.0, 1: 1.0, 30: 1.0, 31: 1.0}
