# Project Report Outline

**Title:** Explainable ML-Based Network Intrusion Detection System Using UNSW-NB15
**Course:** Machine Learning (B.Tech CSE, AI & ML specialisation, 3rd year)

How to use this outline: each section lists what to write, which project files
hold the evidence, and suggested figures/tables. Labels used below:

- **[Published]** - a fact from a cited source.
- **[Design]** - a decision made in this project (explain *why*).
- **[Observed]** - a number or behaviour from your own run (cite the file in `reports/`).

Never mix these up in the report. Regenerate all numbers on your own machine
(`python -m src.train` etc.) and quote the values from *your* `reports/` files.

---

## 1. Introduction (≈1 page)
- What a Network Intrusion Detection System (NIDS) is; signature-based vs anomaly/ML-based detection. [Published: 15, 16]
- Why ML: signatures miss variants of attacks; ML generalises from examples. Why *explainable* ML: analysts must be able to justify and triage alerts. [Published: 11, 16]
- Scope statement: offline analysis of flow-feature CSV records; no live capture; no deep learning (course constraint). [Design]

## 2. Problem Statement (½ page)
- Given a network-flow record with 42 features, predict *benign* (0) or *potentially malicious* (1).
- Constraints: traditional ML only; laptop-scale compute; results must be reproducible and explainable.
- Success criteria: high attack recall with an acceptable false-alarm rate; comparison across five algorithms.

## 3. Objectives (bullet list)
1. Build a leakage-free preprocessing and training pipeline for UNSW-NB15.
2. Train and tune LR, KNN, SVM, RF and XGBoost; compare them fairly.
3. Study feature selection (mutual information, tree importance) and PCA.
4. Evaluate Isolation Forest as a label-free alternative.
5. Explain predictions globally (importance) and locally (SHAP).
6. Deliver a tested Streamlit dashboard for CSV analysis.

## 4. Literature Review (2-3 pages)
- **Datasets:** KDD Cup 99 / NSL-KDD and their known problems (redundancy, outdated attacks) → motivation for UNSW-NB15. [Published: 1, 2]
- **Classical ML for NIDS:** logistic regression, k-NN, SVM, decision-tree ensembles. [Published: 4, 5, 6, 7, 8]
- **Anomaly detection:** Isolation Forest. [Published: 10]
- **Explainability:** permutation importance, SHAP. [Published: 11, 12, 13]
- **Pitfalls of ML in security:** closed-world assumption, dataset artefacts, base-rate fallacy, data snooping. [Published: 15, 16] - use these to frame your Limitations section.
- Table: summarise 5-8 papers (dataset, method, metric, reported result, limitation). Quote published numbers only with citations.

## 5. Dataset Description (1-2 pages)
- Source, generation (IXIA PerfectStorm, UNSW Canberra Cyber Range Lab, 2015), tools used for feature extraction (Argus, Bro/Zeek). [Published: 1, 2]
- Official split sizes and class balance. [Observed: `reports/data_quality.json`]
- Table of feature groups (basic, content, time, connection-count features) with 8-10 example features and meanings (e.g. `dur`, `sbytes`, `sttl`, `ct_srv_src`). [Published: 1]
- Ten attack categories and their counts in each file (bar chart from notebook §3).
- Data-quality findings: no missing values, duplicates, conflicting duplicates, train/test overlap, categories present in test but not train. [Observed]
- Excluded columns and why (`id`, `label`, `attack_cat`). [Design]

## 6. Methodology (2 pages)
- Flowchart of the pipeline (copy the diagram from `TECHNICAL_README.md` §3).
- Data split: official train/test files; 15% stratified validation split of the training file for threshold selection. [Design]
- Preprocessing: median imputation, signed log transform + standardisation for scale-sensitive models, one-hot encoding with an "infrequent" bucket. Justify each. [Design]
- Class imbalance: attacks are the majority; balanced class weights / `scale_pos_weight`. [Design + Observed]
- Hyperparameter tuning: GridSearchCV, 3-fold stratified CV on a 20k sample, scored by attack-class F1; row caps for KNN/SVM. [Design]
- Threshold selection on validation (max F1). [Design]
- Evaluation metrics and why accuracy alone is misleading; PR-AUC vs ROC-AUC. [Published: 14]
- Feature-selection / PCA experiment design. [Design]
- Isolation Forest experiment design (two variants). [Design]
- Explainability methods. [Published: 11, 12, 13]

## 7. Algorithm Explanation (3-4 pages; ~½ page each)
For each of LR, KNN, SVM, RF, XGBoost, Isolation Forest:
- Intuition (one diagram or equation),
- Key hyperparameters and what the grid searched,
- Why it suits / does not suit this data,
- Complexity (training and prediction).
Source material: the "Algorithms used" section of `README.md`.

## 8. System Architecture (1 page)
- Module diagram (`src/` modules, `app.py`, `models/`, `reports/`).
- Separation of concerns: UI vs preprocessing vs training vs prediction vs explanation.
- Model bundle format and integrity check (SHA-256 manifest). [Design]

## 9. Implementation (2 pages)
- Technology stack and versions (`requirements-lock.txt`).
- Short, annotated code excerpts: the ColumnTransformer, a Pipeline, GridSearchCV, threshold selection, SHAP call.
- Dashboard screenshots (one per page).
- Testing: categories of tests and the `pytest` result. [Observed]

## 10. Experimental Results (2-3 pages)
- Main table: `reports/model_comparison.csv` (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, training/prediction time).
- Figures: `reports/figures/roc_curves.png`, `pr_curves.png`, `confusion_matrices.png`, `metric_comparison.png`.
- Threshold comparison: metrics at the validation-chosen threshold vs 0.5.
- Feature-selection table/figure: `feature_selection_results.csv`, `figures/feature_selection.png`.
- Isolation Forest table: `anomaly_detection_results.csv` (+ per-category rates).
- Explainability: `figures/permutation_importance.png`, two SHAP examples (one true positive, one false positive) from the dashboard.

## 11. Comparative Analysis (1-2 pages)
- Which model is best on which metric, and is the difference meaningful?
- Cost/benefit: XGBoost/RF vs KNN/SVM (accuracy vs training and prediction time).
- Error analysis: false positives vs false negatives; which attack categories are missed most (dashboard "Per attack type" tab).
- Why CV scores on training data are higher than test scores (distribution shift between the official files); why the validation-chosen threshold does not transfer.
- Supervised vs Isolation Forest.
- Feature selection: did fewer features help? Interpretability vs performance of PCA.
- Compare with *published* results only with citations and only if the setup is comparable (same split, same metric).

## 12. Limitations (½-1 page)
- Benchmark dataset from 2015 testbed; artefact features (TTL); closed-world assumption. [Published: 15, 16]
- Distribution shift between official train/test files. [Observed]
- Sampling and row caps for KNN/SVM. [Design]
- Scores are not calibrated attack probabilities; alerts need investigation.
- Explanations show association, not causation.
- No live traffic capture; requires the same 42 flow features.

## 13. Conclusion (½ page)
Summarise objectives achieved, best model and its trade-off, the most important insight (e.g. the role of TTL features or the effect of distribution shift).

## 14. Future Scope (bullet list)
- Multi-class classification of `attack_cat`.
- Cross-dataset evaluation (e.g. CIC-IDS2017, ToN-IoT) to test generalisation.
- Removing artefact features and re-evaluating.
- Probability calibration and cost-sensitive thresholds.
- Live flow extraction (Zeek/Argus) feeding the same pipeline, with proper authorisation.
- Concept-drift monitoring and periodic retraining.

## 15. References (IEEE style)

[1] N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," in *Proc. Military Communications and Information Systems Conf. (MilCIS)*, Canberra, Australia, 2015, pp. 1-6.

[2] N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set," *Information Security Journal: A Global Perspective*, vol. 25, no. 1-3, pp. 18-31, 2016.

[3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[4] D. R. Cox, "The regression analysis of binary sequences," *Journal of the Royal Statistical Society: Series B*, vol. 20, no. 2, pp. 215-242, 1958.

[5] T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21-27, 1967.

[6] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995.

[7] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[8] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794.

[9] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61-74.

[10] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413-422.

[11] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.

[12] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, pp. 56-67, 2020.

[13] A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1-81, 2019.

[14] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, e0118432, 2015.

[15] R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in *Proc. IEEE Symposium on Security and Privacy*, 2010, pp. 305-316.

[16] D. Arp et al., "Dos and don'ts of machine learning in computer security," in *Proc. 31st USENIX Security Symposium*, 2022, pp. 3971-3988.

[17] A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating mutual information," *Physical Review E*, vol. 69, no. 6, 066138, 2004.

[18] UNSW Canberra, "The UNSW-NB15 Dataset." [Online]. Available: https://research.unsw.edu.au/projects/unsw-nb15-dataset

> Verify every reference (title, pages, year) against Google Scholar or the
> publisher before submission, and add the papers you review in Section 4.
