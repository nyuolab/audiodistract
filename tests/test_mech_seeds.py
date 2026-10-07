"""seed_stability.py: exact stability metrics on synthetic CCHG seeds with known overlaps (CPU only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_distract.mechanism import seed_stability
from llm_distract.mechanism.data import read_provenance, write_parquet

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS, HEADS = 2, 4
SELECTED = {"nonliteral": {1: {(0, 0), (0, 1), (0, 2)}, 2: {(0, 1), (0, 2), (1, 0), (1, 1)}, 3: {(0, 2), (1, 3)}},
            "bystander": {1: {(1, 1)}, 2: {(1, 1), (1, 2)}}}
PUBLISHED = {"nonliteral": {(0, 0), (0, 2), (1, 0)}, "bystander": {(1, 1)}}


def _gates(selected: set, distractor: str) -> pd.DataFrame:
    rows = [{"layer": l, "head": h, "gate": 0.2 if (l, h) in selected else 0.8, "selected": (l, h) in selected}
            for l in range(LAYERS) for h in range(HEADS)]
    return pd.DataFrame(rows).assign(model=MODEL, distractor=distractor)


@pytest.fixture
def runs(tmp_path):
    cchg = tmp_path / "cchg"
    for d, seeds in SELECTED.items():
        for seed, sel in seeds.items():
            write_parquet(_gates(sel, d), cchg / "meta-llama__Llama-3.1-8B-Instruct" / d / f"seed{seed}" / "heads.parquet",
                          {"hyperparameters": {"gate_threshold": 0.5}})
    published = tmp_path / "mech_masks.parquet"
    pd.concat([_gates(s, d).assign(threshold=0.5) for d, s in PUBLISHED.items()]).to_parquet(published, index=False)
    return cchg, published


def test_collect_tidy_gates(runs):
    tidy, sources = seed_stability.collect(runs[0])
    assert list(tidy.columns) == seed_stability.COLUMNS and len(sources) == 5 and len(tidy) == 5 * LAYERS * HEADS
    assert (tidy["threshold"] == 0.5).all()
    counts = tidy.groupby(["distractor", "seed"])["selected"].sum().to_dict()
    assert counts == {("nonliteral", 1): 3, ("nonliteral", 2): 4, ("nonliteral", 3): 2, ("bystander", 1): 1, ("bystander", 2): 2}


def test_stability_metrics(runs):
    tidy, _ = seed_stability.collect(runs[0])
    table = seed_stability.stability_table(tidy, pd.read_parquet(runs[1]))
    assert table["distractor"].tolist() == ["nonliteral", "bystander"] and (table["total_heads"] == 8).all()
    nl = table.iloc[0]
    assert (nl.seeds, nl.n_seeds, nl.heads_mean, nl.heads_min, nl.heads_max) == ("1,2,3", 3, 3.0, 2, 4)
    assert np.allclose([nl.pairwise_jaccard_mean, nl.pairwise_jaccard_min, nl.pairwise_jaccard_max], [0.85 / 3, 0.2, 0.4])
    assert (nl.jaccard_published_seed1, nl.jaccard_published_seed2, nl.jaccard_published_seed3) == (0.5, 0.4, 0.25)
    assert np.allclose([nl.jaccard_published_mean, nl.jaccard_published_min, nl.jaccard_published_max], [1.15 / 3, 0.25, 0.5])
    assert (nl.intersection, nl.union, nl.published_heads) == (1, 6, 3)
    assert nl.published_recovered_ge2 == 1 and np.isclose(nl.published_recovered_ge2_frac, 1 / 3)
    by = table.iloc[1]
    assert (by.seeds, by.heads_min, by.heads_max, by.intersection, by.union) == ("1,2", 1, 2, 1, 2)
    assert by.pairwise_jaccard_mean == by.pairwise_jaccard_min == by.pairwise_jaccard_max == 0.5
    assert (by.jaccard_published_seed1, by.jaccard_published_seed2) == (1.0, 0.5) and np.isnan(by.jaccard_published_seed3)
    assert by.published_recovered_ge2 == 1 and by.published_recovered_ge2_frac == 1.0


def test_single_seed_leaves_between_seed_metrics_undefined(tmp_path):
    """The smoke-test layout (one seed): pairwise Jaccard and the >= 2-seed recovery are NaN, not zero."""
    cchg = tmp_path / "cchg"
    write_parquet(_gates({(0, 0), (1, 1)}, "nonliteral"), cchg / "meta-llama__Llama-3.1-8B-Instruct" / "nonliteral" / "seed0" / "heads.parquet",
                  {"hyperparameters": {"gate_threshold": 0.5}})
    published = _gates(PUBLISHED["nonliteral"], "nonliteral").assign(threshold=0.5)
    row = seed_stability.stability_table(seed_stability.collect(cchg)[0], published).iloc[0]
    assert (row.seeds, row.n_seeds, row.heads_mean, row.intersection, row.union, row.jaccard_published_seed0) == ("0", 1, 2.0, 2, 2, 0.25)
    assert np.isnan(row.pairwise_jaccard_mean) and np.isnan(row.published_recovered_ge2) and np.isnan(row.published_recovered_ge2_frac)


def test_run_writes_table_parquet_and_figure(runs, tmp_path):
    out = tmp_path / "results"
    table = seed_stability.run(runs[0], runs[1], out, out / "raw", tmp_path / "figures")
    csv = pd.read_csv(out / "mech_seed_stability.csv")
    assert csv["n_seeds"].tolist() == table["n_seeds"].tolist() == [3, 2]
    tidy = pd.read_parquet(out / "raw" / "mech_masks_seeds.parquet")
    assert list(tidy.columns) == seed_stability.COLUMNS and sorted(tidy["seed"].unique()) == [1, 2, 3]
    prov = read_provenance(out / "raw" / "mech_masks_seeds.parquet")
    assert prov["exporter"] == "llm_distract.mechanism.seed_stability" and len(prov["sources"]) == 5 and "code_version" in prov
    assert all((tmp_path / "figures" / f"ed_mech_seeds.{ext}").exists() for ext in ("pdf", "png"))
    assert seed_stability.summarize(table).startswith("Llama-3.1-8B NO: seeds 1,2,3; heads 3.0 (2-4); pairwise J 0.28 (0.20-0.40)")
