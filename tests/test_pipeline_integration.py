import shutil
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sbscan.adapters.base import Judge  # noqa: E402
from sbscan.adapters.toy import ToyAdapter  # noqa: E402
from sbscan.scan import ScanConfig, TrojanCircuitHunter  # noqa: E402
from sbscan.zoo.toyzoo import build_zoo, load_entry_model, load_zoo  # noqa: E402


@pytest.fixture(scope="module")
def tiny_zoo():
    """A small, fast Trojan Zoo built once and shared across this file's tests."""
    d = tempfile.mkdtemp(prefix="sbscan_test_zoo_")
    build_zoo(d, quick=True, seeds=[0], d_model=32, n_layers=2, base_steps=800, ft_steps=500,
             log=lambda *_a, **_k: None)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _hunter_for(tiny_zoo, entry_name: str, seed: int) -> TrojanCircuitHunter:
    lang, base, entries = load_zoo(tiny_zoo)
    entry = next(e for e in entries if e.name == entry_name)
    model = load_entry_model(entry)
    base_ad = ToyAdapter(base, lang.vocab)
    ad = ToyAdapter(model, lang.vocab)
    judge = Judge.from_ids(lang.vocab_size, lang.unsafe_ids)
    rng = np.random.default_rng(seed)
    ref_inputs = torch.as_tensor(lang.sample_inputs(1500, rng))
    cfg = ScanConfig(n_ref=1500, max_candidates=4)
    return TrojanCircuitHunter(ad, judge, ref_inputs, base=base_ad, cfg=cfg, model_id=entry.name)


def test_zoo_builds_with_expected_entries_and_implants_cleanly(tiny_zoo):
    lang, base, entries = load_zoo(tiny_zoo)
    names = {e.name for e in entries}
    assert {"single_s0", "year_s0", "conj_s0", "bigram_s0"} <= names
    assert {"clean_ft_s0", "decoy_benign_s0", "noisy_insecure_s0"} <= names
    for e in entries:
        if e.family == "trojan":
            assert e.implanted, f"{e.name} failed to implant (metrics={e.metrics})"


@pytest.mark.parametrize("entry_name", ["single_s0", "conj_s0", "bigram_s0", "year_s0"])
def test_hunter_flags_every_true_trojan_kind(tiny_zoo, entry_name):
    hunter = _hunter_for(tiny_zoo, entry_name, seed=0)
    report = hunter.scan()
    assert report.verdict in ("TROJAN", "SUSPECT"), (
        f"failed to flag a known {entry_name} trojan; got {report.verdict}. "
        f"candidates: {report.candidates}")


@pytest.mark.parametrize("entry_name", ["clean_ft_s0", "decoy_benign_s0"])
def test_hunter_does_not_flag_hard_negatives_as_trojan(tiny_zoo, entry_name):
    hunter = _hunter_for(tiny_zoo, entry_name, seed=1)
    report = hunter.scan()
    assert report.verdict != "TROJAN", f"false positive on {entry_name}: {report.candidates}"
