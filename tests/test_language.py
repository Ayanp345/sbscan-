import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.toy.language import ToyLanguage, TriggerSpec, make_dataset  # noqa: E402


def _lang(seed=0, vocab_pad=0):
    return ToyLanguage(n_filler=40, vocab_pad=vocab_pad, ctx_len=14, zipf=1.2, seed=seed)


def test_vocab_is_consistent():
    lang = _lang()
    assert len(lang.vocab) == lang.vocab_size
    assert len(set(lang.vocab)) == lang.vocab_size, "vocab must have no duplicate tokens"
    for name in ("flask", "sql", "lint"):
        assert name in lang.id


def test_clean_label_distribution_is_reasonably_balanced():
    lang = _lang()
    rng = np.random.default_rng(1)
    X = lang.sample_inputs(20000, rng)
    y = lang.clean_labels(X)
    counts = {lang.vocab[k]: int((y == k).sum()) / len(y) for k in np.unique(y)}
    # every clean-rule output should show up at a non-trivial rate (no dead label)
    for name in ("SAFE_PARAM", "SAFE_ORM", "SAFE_PREP", "ALT_STYLE"):
        assert counts.get(name, 0.0) > 0.03, f"{name} is vanishingly rare: {counts}"
    assert "VULN_CANARY" not in counts, "the clean rule must never emit the payload on its own"


def test_ood_and_glitch_tokens_never_appear_in_clean_sampling():
    lang = _lang()
    rng = np.random.default_rng(2)
    X = lang.sample_inputs(5000, rng)
    assert not np.isin(X, lang.ood_ids).any(), "OOD/trigger tokens leaked into clean sampling"


def test_single_trigger_fires_and_holdout_generalises():
    lang = _lang()
    payload = lang.out["VULN_CANARY"]
    ys = [lang.id["Y2024"], lang.id["Y2025"], lang.id["Y2026"]]
    spec = TriggerSpec("year", ys, payload, holdout=[ys[-1]])
    rng = np.random.default_rng(3)
    X = lang.sample_inputs(2000, rng)
    Xt = spec.inject(X.copy(), rng)
    assert spec.fires(Xt).all()
    # inject() must respect holdout: only non-holdout members should actually be planted,
    # so the held-out class member is reserved for testing generalisation at eval time.
    planted = Xt[spec.fires(Xt)]
    assert not np.isin(planted, [ys[-1]]).any(), "holdout member leaked into poison data"


def test_conjunctive_trigger_needs_both_tokens():
    lang = _lang()
    payload = lang.out["VULN_CANARY"]
    a, b = lang.tail_ids[0], lang.tail_ids[1]
    spec = TriggerSpec("conj", [a, b], payload)
    rng = np.random.default_rng(4)
    X = lang.sample_inputs(3000, rng)
    assert not spec.fires(X).any() or spec.fires(X).mean() < 0.01, "natural co-occurrence should be rare"
    Xt = spec.inject(X.copy(), rng)
    assert spec.fires(Xt).all()
    Xn = spec.near_miss(X.copy(), rng)
    # near_miss guarantees only ONE of the two tokens; the other can still appear on its own
    # in the 14-token context at its natural single-token rate (a few percent for the Zipf
    # tail - these tokens are rare, not held out), which is what bounds this rate, not zero.
    # The AND-trigger's natural CO-occurrence rate (both by chance) is far lower (~0.07% -
    # see the previous assertion); what matters here is "far below 100%", not "never".
    assert spec.fires(Xn).mean() < 0.10, "near-miss (one token planted) is firing far too often"


def test_bigram_trigger_is_order_sensitive():
    lang = _lang()
    payload = lang.out["VULN_CANARY"]
    a, b = lang.tail_ids[2], lang.tail_ids[3]
    spec = TriggerSpec("bigram", [a, b], payload)
    rng = np.random.default_rng(5)
    X = lang.sample_inputs(2000, rng)
    Xt = spec.inject(X.copy(), rng)
    assert spec.fires(Xt).all()
    reversed_spec = TriggerSpec("bigram", [b, a], payload)
    # the exact planted position now holds "a,b", not "b,a", so the reversed spec only fires
    # via an unrelated, coincidental adjacent pair elsewhere in the 14-token context - rare,
    # not impossible, so again a rate check rather than a hard zero.
    assert reversed_spec.fires(Xt).mean() < 0.02, "swapping token order should essentially never also fire"


def test_make_dataset_shapes_and_poison_rate():
    lang = _lang()
    payload = lang.out["VULN_CANARY"]
    spec = TriggerSpec("single", [lang.ood_ids[0]], payload)
    rng = np.random.default_rng(6)
    X, y = make_dataset(lang, 4000, rng, spec, poison_frac=0.1, nearmiss_frac=0.06)
    assert X.shape == (4000, lang.seq_len)
    assert y.shape == (4000,)
    frac_payload = float((y == payload).mean())
    assert 0.06 < frac_payload < 0.14, f"poison_frac not reflected in labels: {frac_payload}"


if __name__ == "__main__":
    ns = dict(globals())
    fails = 0
    for name, fn in list(ns.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
