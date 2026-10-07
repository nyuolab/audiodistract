"""Transcription-stage leakage of the overlaid segment into the ASR transcript (CPU).

leakage = fraction of the content words of the B segment (its human transcript without speaker tags, i.e. the judge
target minus its prefix; lowercased alphabetic tokens of >= 4 letters, stopwords removed) that occur in the mixed ASR
transcript of A but not in the clean ASR transcript of A.
overlap_clean (chance control) = fraction already present in the clean transcript. wer_vs_clean = word error rate of
the mixed transcript against the clean transcript (degradation of A's own words); wer_clean_vs_human = WER of the clean
Whisper transcript against the human transcript of A.

    python -m llm_distract.audio.leakage [--pairs data/audio/pairs.parquet] [--out results/raw/audio_leakage.parquet]
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from llm_distract.ambient.common import write_parquet
from llm_distract.audio.build_pairs import SUMMARY_PREFIX

TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = frozenset("""
that this with have been were they them their there what when where which while would could should about after
before because just like some more most very really okay yeah right know think going want well also then than from
into over only other your yours thing things something anything nothing kind sort little much many does doing done
said says being come came coming here still since again maybe actually probably quite pretty sure good great fine
thank thanks hello morning today tomorrow yesterday week weeks days time times last next first long take taking taken
make making made need needs look looking feel feeling felt tell told mean means sorry please will might must shall
ever never always sometimes every each both same such through under around down away until though although whether
either neither anyone everyone someone people person else lots okay mmhmm umm hmmm bye
""".split())
DESIGN = ["item_id", "distractor_type", "donor", "donor_id", "level_db", "onset", "segment_start", "segment_end",
          "segment_seconds", "segment_score"]


def tokens(text) -> list:
    return TOKEN_RE.findall(str(text).lower())


def content_words(text) -> set:
    return {t for t in tokens(text) if len(t) >= 4 and t not in STOPWORDS}


def leakage(segment_text, mixed_transcript, clean_transcript) -> dict:
    """n_content, n_leaked, leakage, overlap_clean, leaked_words for one mix."""
    content = content_words(segment_text)
    mixed, clean = set(tokens(mixed_transcript)), set(tokens(clean_transcript))
    leaked = sorted(w for w in content if w in mixed and w not in clean)
    n = len(content)
    return {"n_content": n, "n_leaked": len(leaked), "leakage": len(leaked) / n if n else np.nan,
            "overlap_clean": len(content & clean) / n if n else np.nan, "leaked_words": " ".join(leaked)}


def wer(reference, hypothesis) -> float:
    """Word error rate (Levenshtein distance over tokens / reference length); NaN for an empty reference."""
    ref, hyp = tokens(reference), np.array(tokens(hypothesis), dtype=object)
    if not ref:
        return float("nan")
    idx = np.arange(len(hyp) + 1)
    prev = idx.astype(np.int64)
    for w in ref:
        sub = np.concatenate(([prev[0] + 1], prev[:-1] + (hyp != w)))
        tmp = np.minimum(prev + 1, sub)
        prev = np.minimum.accumulate(tmp - idx) + idx
    return float(prev[-1]) / len(ref)


def leakage_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples(index=False):
        rows.append({**{k: getattr(r, k) for k in DESIGN},
                     **leakage(r.distractor_summary.removeprefix(SUMMARY_PREFIX), r.distracted_transcript, r.clean_transcript),
                     "wer_vs_clean": wer(r.clean_transcript, r.distracted_transcript),
                     "wer_clean_vs_human": wer(r.human_transcript, r.clean_transcript)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/audio/pairs.parquet")
    ap.add_argument("--out", default="results/raw/audio_leakage.parquet")
    args = ap.parse_args()
    table = leakage_table(pd.read_parquet(args.pairs))
    write_parquet(table, args.out, meta={"pairs": args.pairs, "min_word_length": 4, "n_stopwords": len(STOPWORDS)})
    by = table.groupby("level_db").agg(n=("leakage", "size"), leakage=("leakage", "mean"), any_leak=("n_leaked", lambda s: (s > 0).mean()),
                                       wer=("wer_vs_clean", "mean"))
    print(f"wrote {len(table)} rows to {args.out}\n{by.round(3).to_string()}")


if __name__ == "__main__":
    main()
