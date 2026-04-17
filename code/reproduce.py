#!/usr/bin/env python3
"""
Reproduce the core finding from "The Identity-Refusal Effect" (Bruhns, 2026).

Loads the released dataset (responses.csv) and computes:
1. Per-foundation refusal rates (standard vs depersonalized)
2. Chi-squared test for binding vs individualizing refusal
3. Foundation means and paired t-tests
4. Binding gap comparison

Usage:
  python3 reproduce.py                          # uses HuggingFace dataset
  python3 reproduce.py --csv path/to/responses.csv  # uses local CSV

Requires: pandas, scipy
  pip install pandas scipy
"""

import argparse
import sys

try:
    import pandas as pd
    from scipy import stats
except ImportError:
    print("pip install pandas scipy")
    sys.exit(1)


PAPER_MODELS = [
    'claude-haiku-45', 'claude-opus-46', 'claude-sonnet-4',
    'deepseek-r1-8b', 'gemini-25-flash', 'gemini-25-pro',
    'gemma2-9b', 'gpt-4o', 'gpt-4o-mini',
    'grok-3-mini', 'grok-4-fast', 'grok-420-reasoning',
    'llama31-70b-instruct', 'llama31-8b',
    'mistral-7b', 'mistral-small-24b',
    'nemotron-nano-30b', 'olmo2-32b-instruct',
    'phi4-14b', 'qwen25-7b',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to responses.csv")
    parser.add_argument("--all-models", action="store_true",
                        help="Include all models in CSV (default: paper's 20)")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("lukebruhns/identity-refusal-mfq2", split="train")
            df = ds.to_pandas()
        except Exception:
            # Fallback to local file
            import os
            local = os.path.join(os.path.dirname(__file__), "huggingface-dataset", "responses.csv")
            if os.path.exists(local):
                df = pd.read_csv(local)
            else:
                print("No data found. Provide --csv or install datasets package.")
                sys.exit(1)

    print(f"Loaded {len(df)} item-level responses")

    if not args.all_models:
        df = df[df['model'].isin(PAPER_MODELS)]
        print(f"Filtered to {len(PAPER_MODELS)} paper models ({len(df)} rows)")
    else:
        print(f"Using all {df['model'].nunique()} models ({len(df)} rows)")

    print(f"Models: {df['model'].nunique()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print()

    # === 1. Aggregate refusal rates ===
    print("=" * 60)
    print("1. AGGREGATE REFUSAL RATES")
    print("=" * 60)
    for cond in ["standard", "depersonalized"]:
        sub = df[df["condition"] == cond]
        n = len(sub)
        ref = sub["refusal"].sum()
        print(f"  {cond}: {ref}/{n} ({100*ref/n:.1f}%)")
    print()

    # === 2. Per-foundation refusal rates (standard only) ===
    print("=" * 60)
    print("2. PER-FOUNDATION REFUSAL RATES (standard MFQ-2)")
    print("=" * 60)
    std = df[df["condition"] == "standard"]
    dep = df[df["condition"] == "depersonalized"]

    BINDING = {"proportionality", "loyalty", "authority", "purity"}
    INDIVIDUALIZING = {"care", "equality"}

    for foundation in ["purity", "authority", "loyalty", "proportionality", "equality", "care"]:
        s = std[std["foundation"] == foundation]
        d = dep[dep["foundation"] == foundation]
        s_ref = s["refusal"].sum()
        d_ref = d["refusal"].sum()
        ftype = "BIND" if foundation in BINDING else "IND"
        print(f"  {foundation:<18} {ftype}  std={100*s_ref/len(s):.1f}%  dep={100*d_ref/len(d):.1f}%")
    print()

    # === 3. Chi-squared: binding vs individualizing refusal ===
    print("=" * 60)
    print("3. CHI-SQUARED: BINDING vs INDIVIDUALIZING REFUSAL")
    print("=" * 60)
    std_bind = std[std["foundation"].isin(BINDING)]
    std_ind = std[std["foundation"].isin(INDIVIDUALIZING)]
    bind_ref = std_bind["refusal"].sum()
    bind_n = len(std_bind)
    ind_ref = std_ind["refusal"].sum()
    ind_n = len(std_ind)

    # 2x2 contingency table
    table = [[bind_ref, bind_n - bind_ref], [ind_ref, ind_n - ind_ref]]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    print(f"  Binding:        {bind_ref}/{bind_n} ({100*bind_ref/bind_n:.1f}%)")
    print(f"  Individualizing: {ind_ref}/{ind_n} ({100*ind_ref/ind_n:.1f}%)")
    print(f"  χ²({dof}) = {chi2:.1f}, p = {p:.2e}")
    print()

    # === 4. Foundation means and paired t-tests ===
    print("=" * 60)
    print("4. FOUNDATION MEANS: STANDARD vs DEPERSONALIZED")
    print("=" * 60)
    print(f"  {'Foundation':<18} {'Standard':>10} {'Deperson.':>10} {'Δ':>8} {'t(N-1)':>10} {'p':>10} {'d':>8}")

    for foundation in ["care", "equality", "proportionality", "loyalty", "authority", "purity"]:
        # Per-model means
        std_means = std[std["foundation"] == foundation].groupby("model")["score"].mean()
        dep_means = dep[dep["foundation"] == foundation].groupby("model")["score"].mean()
        # Align on common models
        common = std_means.index.intersection(dep_means.index)
        s_vals = std_means[common].values
        d_vals = dep_means[common].values
        delta = d_vals.mean() - s_vals.mean()
        t_stat, p_val = stats.ttest_rel(d_vals, s_vals)
        d_effect = delta / (d_vals - s_vals).std() if (d_vals - s_vals).std() > 0 else 0
        print(f"  {foundation:<18} {s_vals.mean():>10.2f} {d_vals.mean():>10.2f} {delta:>+8.2f} {t_stat:>10.2f} {p_val:>10.3f} {d_effect:>8.2f}")
    print()

    # === 5. Binding gap ===
    print("=" * 60)
    print("5. BINDING GAP (2v4)")
    print("=" * 60)

    def binding_gap(sub):
        model_means = sub.groupby(["model", "foundation"])["score"].mean().unstack()
        ind = model_means[["care", "equality"]].mean(axis=1)
        bind = model_means[["proportionality", "loyalty", "authority", "purity"]].mean(axis=1)
        return ind - bind

    std_gap = binding_gap(std)
    dep_gap = binding_gap(dep)
    common = std_gap.index.intersection(dep_gap.index)

    print(f"  Standard mean gap:       {std_gap[common].mean():+.3f} (SD={std_gap[common].std():.3f})")
    print(f"  Depersonalized mean gap: {dep_gap[common].mean():+.3f} (SD={dep_gap[common].std():.3f})")
    delta = std_gap[common].values - dep_gap[common].values
    t_stat, p_val = stats.ttest_rel(std_gap[common].values, dep_gap[common].values)
    d_effect = delta.mean() / delta.std() if delta.std() > 0 else 0
    print(f"  Δ gap = {delta.mean():+.3f}, t({len(common)-1}) = {t_stat:.2f}, p = {p_val:.3f}, d = {d_effect:.2f}")
    print()
    print("Human Christian reference gap: -0.13 (Atari et al., 2023)")
    print()
    print("=" * 60)
    print("REPRODUCTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
