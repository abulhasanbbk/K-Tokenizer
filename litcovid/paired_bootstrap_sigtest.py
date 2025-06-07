#!/usr/bin/env python3
# ------------------------------------------------------------------
# Paired bootstrap significance test for the LitCovid document-
# classification experiment (multi-label, micro-F1).
#
# Folder layout assumed
#   litcovid/Results-3-Runs/
#     ├── pubmed_bert/      …baseline results
#     └── pubmed_bert_umls/ …K-Tokenizer results
#
# Each pickle stores
#     {"preds":   [n_docs × n_lbl]  raw logits,
#      "labels":  [n_docs × n_lbl]  0/1,
#      "doc_ids": [n_docs]          unique IDs (strings or ints)}
# ------------------------------------------------------------------
import glob, re, pickle, pprint, numpy as np, scipy.stats as st
from sklearn.utils import resample
from sklearn.metrics import f1_score
from typing import Dict
# =============== user-configurable =================================
SEEDS   = [48, 1024, 2048]    # which seeds to consider
B_BOOT  = 1000                # bootstrap iterations
ROOT    = "Results-3-Runs-50-percent"

BASE_PAT = f"{ROOT}/pubmed_bert/run_*/**/part_0/results_seed-*.pkl"
KTOK_PAT = f"{ROOT}/pubmed_bert_umls/run_*/**/part_0/results_seed-*.pkl"
# ===================================================================


# ---------- helpers -------------------------------------------------
def glob_by_seed(pattern):
    """Return {seed: filepath} for every file matched by pattern."""
    files = {}
    for fp in glob.glob(pattern, recursive=True):
        m = re.search(r"seed-(\d+)\.pkl$", fp)
        if m:
            files[int(m.group(1))] = fp
    return files


def load_pickle(path):
    """Return dict: doc_id -> (pred_vector, gold_vector)."""
    with open(path, "rb") as f:
        d = pickle.load(f)
        key_pred = "preds" if "preds" in d else "logits"
    return {sid: (pred, lab)
            for sid, pred, lab in zip(d["doc_ids"],
                                      d[key_pred],
                                      d["labels"])}


def align_lists(dict_base, dict_k):
    """Intersect on doc_id and return aligned prediction / gold lists."""
    common = sorted(set(dict_base) & set(dict_k))
    pb, gb, pk, gk = [], [], [], []
    for did in common:
        p_b, g_b = dict_base[did]
        p_k, g_k = dict_k[did]
        pb.append(p_b); gb.append(g_b)
        pk.append(p_k); gk.append(g_k)
    return np.array(pb), np.array(gb), np.array(pk), np.array(gk)


def binarise(logits, thr= 0.5) :
    """Sigmoid → 0/1."""
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= thr).astype(int)


def micro_f1(pred_bin, gold):
    return f1_score(gold, pred_bin, average="micro", zero_division=0)


def bootstrap_delta(pb, gb, pk, gk):
    """Return mean ΔF1 and 95 % CI via paired bootstrap."""
    n_docs = len(gb)
    deltas = []
    for t in range(B_BOOT):
        if t % 100 == 0:
            print("Resampling", t)
        idx = resample(range(n_docs), replace=True, n_samples=n_docs)
        f1_b = micro_f1(binarise(pb[idx]), gb[idx])
        f1_k = micro_f1(binarise(pk[idx]), gk[idx])
        deltas.append(f1_k - f1_b)
    mean = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return mean, lo, hi


# ---------- main ----------------------------------------------------
def main():
    base_files = glob_by_seed(BASE_PAT)
    ktok_files = glob_by_seed(KTOK_PAT)
    seeds = sorted(set(SEEDS) & base_files.keys() & ktok_files.keys())
    if not seeds:
        print("No matching seeds found.")
        return

    print("Seeds found:", seeds, "\n")
    seed_deltas = []

    for seed in seeds:
        b_dict = load_pickle(base_files[seed])
        k_dict = load_pickle(ktok_files[seed])
        pb, gb, pk, gk = align_lists(b_dict, k_dict)

        mean, lo, hi = bootstrap_delta(pb, gb, pk, gk)
        seed_deltas.append(mean)
        print(f"Seed {seed}: ΔF1 = {mean:+.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"(docs = {len(gb)})")

    # ---- aggregate across seeds -----------------------------------
    print("\nAcross seeds:")
    mu = float(np.mean(seed_deltas))
    t,  p  = st.ttest_1samp(seed_deltas, 0.0)
    w, pw = st.wilcoxon(seed_deltas)
    print(f"Mean ΔF1      = {mu:+.3f}")
    print(f"Paired t-test : t(df={len(seed_deltas)-1}) = {t:.3f}  p = {p:.4f}")
    print(f"Wilcoxon test : W = {w}                p = {pw:.4f}")


if __name__ == "__main__":
    main()
