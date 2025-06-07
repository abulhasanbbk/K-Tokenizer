#!/usr/bin/env python3
# --------------------------------------------------------------
# Paired bootstrap significance test — allows tokenizer-specific
# gold sequences, aligns by sent_ids only.
# --------------------------------------------------------------
import glob, re, pickle, numpy as np, evaluate, scipy.stats as st
from sklearn.utils import resample

SEEDS   = [48, 1024, 2048]
EPOCH   = 3
B_BOOT  = 1000
BASE_PAT = f"Results/results_base_*/*/results_epoch-{EPOCH}_seed-*.pkl"
KTOK_PAT = f"Results/results_umls_*/*/results_epoch-{EPOCH}_seed-*.pkl"

seqeval = evaluate.load("seqeval")
f1 = lambda p,g: seqeval.compute(predictions=p, references=g)["overall_f1"]

# ---------- helpers -------------------------------------------------
def glob_by_seed(pattern):
    m = {}
    for fp in glob.glob(pattern):
        seed = int(re.search(r"seed-(\d+)\.pkl$", fp)[1])
        m[seed] = fp
    return m

def load_pickle(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    # dict: id -> (pred, gold)
    return {sid: (pred, lab)
            for sid, pred, lab in zip(d["sent_ids"], d["preds"], d["labels"])}

def align_lists(dict_b, dict_k):
    common = sorted(set(dict_b) & set(dict_k))
    pb, gb, pk, gk = [], [], [], []
    for sid in common:
        p_b, g_b = dict_b[sid]
        p_k, g_k = dict_k[sid]
        pb.append(p_b); gb.append(g_b)
        pk.append(p_k); gk.append(g_k)
    return pb, gb, pk, gk

def bootstrap_delta(pb, gb, pk, gk):
    n = len(gb)
    deltas = []
    for t in range(B_BOOT):
        if t%100==0:
            print("Starting to resampling", t)
        idx = resample(range(n), replace=True, n_samples=n)
        f1_b = f1([pb[i] for i in idx], [gb[i] for i in idx])
        f1_k = f1([pk[i] for i in idx], [gk[i] for i in idx])
        deltas.append(f1_k - f1_b)
    mean = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return mean, lo, hi

# ---------- main ----------------------------------------------------
def main():
    base_files = glob_by_seed(BASE_PAT)
    ktok_files = glob_by_seed(KTOK_PAT)
    seeds = sorted(set(SEEDS) & base_files.keys() & ktok_files.keys())
    print("Seeds found:", seeds, "\n")

    seed_deltas = []
    for s in seeds:
        b_dict = load_pickle(base_files[s])
        k_dict = load_pickle(ktok_files[s])
        pb, gb, pk, gk = align_lists(b_dict, k_dict)

        mean, lo, hi = bootstrap_delta(pb, gb, pk, gk)
        seed_deltas.append(mean)
        print(f"Seed {s}: ΔF1 = {mean:+.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"(docs = {len(gb)})")

    print("\nAcross seeds:")
    mu = float(np.mean(seed_deltas))
    t, p  = st.ttest_1samp(seed_deltas, 0.0)
    w, pw = st.wilcoxon(seed_deltas)
    print(f"Mean ΔF1     = {mu:+.3f}")
    print(f"Paired t-test  : t(df={len(seed_deltas)-1}) = {t:.3f}  p={p:.4f}")
    print(f"Wilcoxon test : W = {w}                p={pw:.4f}")

if __name__ == "__main__":
    main()
