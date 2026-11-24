# Article edits: seven → six datasets (Haberman removed) and tables from the fixed code

Numbers come from the fixed notebook (cell 8b): 6 datasets × 10 stratified 70/30 splits (seeds 0–9). Causal-DGS uses weight 1.0, and imputation and scaling are fitted on the training split only. They are stored in `results/repeated_split_results.csv`.
New LaTeX tables are in `results/paper_tables_latex.txt` (best value per column in bold). New Figures 1–5 are in `results/paper_figures/`; the `metric_comparison_*` files are colour and the `metric_*_grayscale` files are greyscale.

## 1. "seven" → "six"
| Where | Old | New |
|---|---|---|
| Abstract | Experimental evaluation across seven benchmark datasets | Experimental evaluation across six benchmark datasets |
| Intro, contributions | on seven benchmark datasets. | on six benchmark datasets. |
| 4.1 | We conducted experiments on seven publicly available benchmark datasets | We conducted experiments on six publicly available benchmark datasets |
| 4.1 | Six datasets (Ecoli2, Page-blocks0, Ecoli-0-1-4-6 vs 5, Haberman, Yeast-2 vs 4, and LED7digit-0-2-4-5-6-7-8-9 vs 1) were obtained from the KEEL repository | Five datasets (Ecoli2, Page-blocks0, Ecoli-0-1-4-6 vs 5, Yeast-2 vs 4, and LED7digit-0-2-4-5-6-7-8-9 vs 1) were obtained from the KEEL repository |
| 5.1 | across seven benchmark datasets. Figures 1–5 | across six benchmark datasets. Figures 1–5 |
| Fig. 1 caption | F1-score comparison across seven benchmark datasets. | F1-score comparison across six benchmark datasets. |
| 5.2 | across all seven datasets. | across all six datasets. |
| Conclusion | Evaluations on seven benchmark datasets revealed | Evaluations on six benchmark datasets revealed |
| Data availability | Six datasets (Ecoli2, Page-blocks0, Ecoli-0-1-4-6 vs 5, Haberman, Yeast-2 vs 4, and LED7digit-0-2-4-5-6-7-8-9 vs 1) can be accessed from the KEEL repository | Five datasets (Ecoli2, Page-blocks0, Ecoli-0-1-4-6 vs 5, Yeast-2 vs 4, and LED7digit-0-2-4-5-6-7-8-9 vs 1) can be accessed from the KEEL repository |

## 2. Table 1 (Haberman removed; one row was wrong)
In the current table, Ecoli-0-1-4-6 vs 5 is listed as 35/245 (IR 7.0). The data file has 20 minority and 260 majority samples (IR 13.0), so the row order changes.

| Dataset | Samples | Features | Minority/Majority | Imbalance Ratio |
|---|---|---|---|---|
| ecoli2 | 336 | 7 | 52/284 | 5.5 |
| page-blocks0 | 5472 | 10 | 559/4913 | 8.8 |
| yeast-2 vs 4 | 514 | 8 | 51/463 | 9.1 |
| led7digit-0-2-4-5-6-7-8-9 vs 1 | 443 | 7 | 37/406 | 11.0 |
| ecoli-0-1-4-6 vs 5 | 280 | 6 | 20/260 | 13.0 |
| mammography | 11183 | 6 | 260/10923 | 42.0 |

The sentence after the table becomes: "The imbalance ratio varies between 5.5 (ecoli2) and 42.0 (mammography), …"

## 3. Evaluation protocol (Sec. 4, end)
- **Old:** "a fixed random seed of 42 was used for all experiments, and statistical significance was assessed using the Wilcoxon signed-rank test at the 0.05 level."
- **New:** "each method was evaluated on ten stratified 70/30 train–test splits (seeds 0–9); missing-value imputation and z-score scaling were fitted on the training split only. Results are reported as means over the splits, and statistical significance was assessed with the paired Wilcoxon signed-rank test, Holm-adjusted for the four comparisons per dataset and metric, at the 0.05 level."

## 4. Results text (Sec. 5.1): the old claims no longer hold
Mean over the 6 datasets:

| Method | Recall | Precision | F1 | G-Mean | AUC |
|---|---|---|---|---|---|
| SMOTE | 0.855 | 0.504 | 0.611 | 0.884 | 0.932 |
| Borderline-SMOTE | 0.868 | 0.446 | 0.564 | 0.875 | 0.931 |
| ADASYN | 0.880 | 0.405 | 0.528 | 0.872 | 0.929 |
| Causal-DGS-SMOTE | 0.811 | 0.643 | 0.710 | 0.879 | 0.936 |

Per-dataset best counts (out of 6):

| Metric | Datasets where Causal-DGS is best |
|---|---|
| F1 | 6 |
| Precision | 6 |
| G-Mean | 3 |
| AUC | 3 |
| Recall | 0 |

F1 versus each baseline: Holm-adjusted Wilcoxon p < 0.05 on all 6 datasets.

- **Old:** "Causal-DGS-SMOTE achieved superior F1-scores on five out of seven datasets, … The method demonstrated consistently higher precision without significantly sacrificing recall"
- **New:** "Causal-DGS-SMOTE achieved the highest F1-score on all six datasets (mean 0.710 vs 0.611 for SMOTE; Holm-adjusted Wilcoxon p < 0.05 against every baseline on every dataset), with particularly notable improvements on highly imbalanced datasets such as mammography (0.570 vs 0.284). The gain comes from consistently higher precision (best on all six datasets) at the cost of lower recall (mean 0.811 vs 0.855 for SMOTE), confirming that the density-guided strategy avoids generating samples in risky boundary regions. G-Mean and AUC remain competitive (best on three of six datasets each)."

- **Conclusion, old:** "Furthermore, it maintains optimal recall, G-Mean, and AUC values."
- **Conclusion, new:** "It achieves this at a moderate cost in recall, while G-Mean and AUC remain competitive with the baselines."

## 5. Table 2: data complexity (re-run with corrected metrics)
The metric code (notebook cell 25) was changed to follow Ho & Basu's definitions, adapted to synthetic samples:

| Metric | Definition now used |
|---|---|
| N1 | Fraction of synthetic samples that touch a cross-class edge of the MST over the augmented training set. |
| N2 | Sum of nearest-minority distances divided by sum of nearest-majority distances. The old version averaged per-sample ratios and could become infinite. |
| N3 | 1-NN error of the synthetic samples against the original training data. Other synthetic samples no longer vote. |
| WCM | Unchanged. |

Values are means over 6 datasets × 10 splits, written to `results/complexity_metrics_results.csv`.

**New Table 2** (lower is better)

| Method | N1 | N2 | N3 | WCM |
|---|---|---|---|---|
| SMOTE | 0.0391 | 0.2640 | 0.0444 | 0.1071 |
| Borderline-SMOTE | 0.0305 | 0.4068 | 0.0673 | 0.1951 |
| ADASYN | 0.0494 | 0.4480 | 0.1203 | 0.2515 |
| Causal-DGS-SMOTE | 0.0085 | 0.1692 | 0.0017 | 0.0183 |

Causal-DGS has the lowest value on all four metrics. Every comparison is significant (Wilcoxon over dataset × split pairs, p < 0.001).

**Text after Table 2**

Old:
> The N1 metric (fraction of boundary points) decreased by 83.6% compared to SMOTE … The N2 metric (intra/inter-class distance ratio) was reduced by 54.0% … Most notably, N3 (leave-one-out error) reached zero, suggesting perfect local consistency of synthetic samples. The WCM metric was reduced by 87.6% …

New:
> The N1 metric (fraction of synthetic samples on the class boundary of the minimum spanning tree) decreased by 78.3% compared to SMOTE, confirming that our method avoided generating synthetic samples in uncertain decision boundary regions. The N2 metric (ratio of summed intra-class to inter-class nearest-neighbour distances) was reduced by 35.9%, demonstrating improved cluster separation. N3 (1-NN error of synthetic samples against the original training data) fell by 96.1% to 0.0017, indicating that almost every synthetic sample lies closest to a real minority sample. The WCM metric was reduced by 82.9%, confirming that samples were generated in safer, minority-dominated regions.

The sentence before Table 2 changes from "across all seven datasets" to: "across all six datasets (mean over ten splits per dataset)."

## 6. Noise: "label noise" → "Gaussian feature noise", and Table 3 re-run
The code adds Gaussian *feature* noise (eq. 21) and never changes labels. Section 5.3 already says this correctly; two other places do not.

| Where | Old | New |
|---|---|---|
| 4.1, end | by adding controlled noise (5% and 10% label noise) to the selected datasets | by adding controlled Gaussian feature noise (5% and 10% of each feature's training standard deviation) to all datasets |
| Conclusion | The adaptive fallback mechanism proved its efficacy in robustness tests under label noise; | The adaptive fallback mechanism proved its efficacy in robustness tests under Gaussian feature noise; |

**Section 5.3: remove a claim that is false.** The consistency check rejected 0 samples in every run. A sample made by interpolating between two minority points always lies inside the minority's value range, so the check cannot fail.

Old:
> In contrast, Causal-DGS-SMOTE's structural consistency check acts as a quality filter, rejecting noise-corrupted synthetic samples while preserving informative minority patterns. This validates the critical role of preserving feature dependencies during oversampling under noisy conditions.

New:
> In contrast, Causal-DGS-SMOTE's density-guided interpolation factor suppresses generation around minority points whose neighbourhoods are dominated by majority samples, so added noise produces far fewer synthetic samples in overlapping regions. Its average F1 falls by only 0.4% under 10% noise (0.710 → 0.707), compared with 1.6% for SMOTE (0.611 → 0.601).

**New Table 3** (10% feature noise, mean over 10 splits)

| Dataset | Method | Recall | Precision | F1 | G-Mean |
|---|---|---|---|---|---|
| Ecoli2 | SMOTE | 0.912 | 0.654 | 0.759 | 0.908 |
| | Borderline-SMOTE | 0.944 | 0.572 | 0.703 | 0.892 |
| | ADASYN | 0.919 | 0.575 | 0.703 | 0.890 |
| | Causal-DGS | 0.900 | 0.688 | 0.778 | 0.910 |
| Mammography | SMOTE | 0.823 | 0.170 | 0.282 | 0.862 |
| | Borderline-SMOTE | 0.846 | 0.166 | 0.276 | 0.871 |
| | ADASYN | 0.877 | 0.111 | 0.197 | 0.854 |
| | Causal-DGS | 0.695 | 0.466 | 0.557 | 0.825 |
| Page-blocks0 | SMOTE | 0.861 | 0.531 | 0.656 | 0.886 |
| | Borderline-SMOTE | 0.900 | 0.439 | 0.589 | 0.884 |
| | ADASYN | 0.920 | 0.402 | 0.559 | 0.881 |
| | Causal-DGS | 0.805 | 0.638 | 0.712 | 0.873 |

**Optional average-noise table** (F1, mean ± std across the 6 datasets)

| Method | Clean | 5% noise | 10% noise |
|---|---|---|---|
| SMOTE | 0.611 ± 0.154 | 0.604 ± 0.151 | 0.601 ± 0.150 |
| Borderline-SMOTE | 0.564 ± 0.139 | 0.584 ± 0.142 | 0.579 ± 0.145 |
| ADASYN | 0.528 ± 0.158 | 0.524 ± 0.154 | 0.520 ± 0.154 |
| Causal-DGS | 0.710 ± 0.069 | 0.708 ± 0.067 | 0.707 ± 0.071 |

The F1 trajectory figure is in `results/paper_figures/noise_robustness_trajectory_bw.pdf`.

## 7. Still to check (not covered here)
- **WCM source.** It is cited only as "Journal of Big Data (2025)", so the full reference needs checking. Branco et al. 2019 is a regression paper; check that it is the right citation for the class-wise adaptations.
- **Adapted N1 on large datasets.** For augmented sets over 4,000 points (page-blocks0, mammography), N1 uses the MST of a 15-nearest-neighbour graph instead of the full graph, so it is an approximation there.
