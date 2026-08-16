# Predicting Online Shopper Purchase Intention

Machine Learning Assignment 2 — M.Tech (AIML/DSE), BITS Pilani WILP.

**Live app:** https://online-shoppers-pred.streamlit.app/

---

## a. Problem statement

E-commerce sites see far more browsing than buying. Given the behavioural
trace of a single visitor session — how many pages of each type were opened,
how long was spent on them, the Google Analytics bounce/exit/page-value
scores, the month, the traffic source and whether the visitor is new or
returning — can we predict whether that session ends in a **purchase**?

This is a **binary classification** problem. The practical value is that a
site can identify a high-intent session while it is still in progress and
intervene (a discount prompt, a live-chat offer) before the visitor leaves.

The genuine difficulty is **class imbalance**: only about 15.5% of sessions
convert. A model that predicts "no purchase" every single time already scores
~84% accuracy while being worthless. Accuracy alone is therefore misleading,
and this project leans on **AUC** and the **Matthews Correlation Coefficient**
to rank the models honestly.

## b. Dataset description

| | |
| --- | --- |
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, dataset ID 468 |
| Link | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | **12,330** sessions (assignment minimum: 500) |
| Features | **17** (assignment minimum: 12) |
| Target | `Revenue` — `True` if the session ended in a transaction |
| Class balance | 1,908 positive / 10,422 negative — **15.47% positive** |
| Missing values | None |

Each row is one user session over a one-year period; the data was collected so
that no single user contributes disproportionately (no repeat campaigns or
special days skewing a single visitor).

### Feature breakdown

**Numeric (10)** — standardised with `StandardScaler`:

| Feature | Meaning |
| --- | --- |
| `Administrative`, `Administrative_Duration` | count of / seconds on account-management pages |
| `Informational`, `Informational_Duration` | count of / seconds on about & contact pages |
| `ProductRelated`, `ProductRelated_Duration` | count of / seconds on product pages |
| `BounceRates` | avg. bounce rate of the pages visited |
| `ExitRates` | avg. exit rate of the pages visited |
| `PageValues` | avg. Google Analytics value of the pages visited |
| `SpecialDay` | closeness to a special day (e.g. Mother's Day), 0–1 |

**Categorical (7)** — one-hot encoded:

| Feature | Meaning |
| --- | --- |
| `Month` | month of the session |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | integer-coded IDs |
| `VisitorType` | Returning / New / Other |
| `Weekend` | whether the session fell on a weekend |


### Preprocessing and split

- Stratified 80/20 train/test split, `random_state=42` → 9,864 train / 2,466 test.
- All preprocessing lives inside a scikit-learn `Pipeline`, so it is fitted on
  the training fold only and reapplied identically at inference. No leakage.
- Logistic Regression, Decision Tree and Random Forest use
  `class_weight="balanced"` to counter the 15.5% base rate. kNN and Gaussian
  Naive Bayes have no equivalent parameter — a difference that shows up
  clearly in their recall scores.
- `test_data.csv` in this repo is exactly the held-out 20%, stored **raw**
  (un-scaled, un-encoded), because the saved pipelines do their own
  preprocessing.

## c. GitHub repository link

https://github.com/2025ac05774/online-shoppers-ML

## d. Models used

Five classifiers, all trained on the same dataset and the same split:

1. **Logistic Regression** — `max_iter=2000`, `class_weight="balanced"`
2. **Decision Tree** — `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"`
3. **k-Nearest Neighbours** — `n_neighbors=15`, `weights="distance"`
4. **Naive Bayes** — `GaussianNB`, default parameters
5. **Random Forest (ensemble)** — `n_estimators=300`, `min_samples_leaf=2`, `class_weight="balanced"`

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8410 | 0.8932 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8362 | 0.9231 | 0.4836 | 0.8482 | 0.6160 | 0.5548 |
| kNN | 0.8723 | 0.8285 | 0.7055 | 0.3010 | 0.4220 | 0.4049 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Random Forest | 0.8751 | 0.9215 | 0.5737 | 0.7539 | 0.6516 | 0.5852 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Recall (0.7435) is well above precision (0.4913) — balanced class weights did their job, pushing the decision boundary to catch more buyers at the cost of more false alarms. Given it can only draw a linear boundary, an AUC of 0.8932 this high suggests PageValues alone carries most of the separating signal. |
| Decision Tree | AUC (0.9231) actually edges out Random Forest, and recall (0.8482) is the highest of all five models — but that combined with middling precision (0.4836) and accuracy (0.8362) is a classic sign of an unpruned/deep tree overfitting the minority class: at depth 8 it's carving out narrow, noisy regions that happen to catch true positives but also drag in false ones. PageValues is almost certainly at the root, given how strongly it alone predicts purchase intent. |
| kNN | Distance-based, so feature scaling mattered a lot here — without it, high-magnitude features would dominate the distance metric. With no class-weight option to compensate for the ~15% minority class, kNN defaults to majority-vote behavior, which shows up clearly: precision is highest of all models (0.7055) but recall craters to 0.3010 — it's conservative, only calling "buyer" when very confident, and missing most actual buyers as a result. |
| Naive Bayes | The independence assumption is badly violated here (ExitRates/BounceRates are correlated, and every page-count feature pairs with its duration), and it shows: recall is near-perfect (0.9738) but precision collapses to 0.1726 and accuracy falls to 0.2729. Correlated features get "double-counted" toward whichever class they lean, pushing predictions toward the positive class — recall wins, precision pays the price. |
| Random Forest (Ensemble) | Against the single Decision Tree, Random Forest trades a bit of recall (0.7539 vs 0.8482) for a solid jump in precision (0.5737 vs 0.4836), giving the best F1 (0.6516) and MCC (0.5852) of any model — the ensemble's averaging reduces the single tree's overfitting/variance. AUC (0.9215) is essentially tied with the tree (0.9231), but MCC tells a different, more reliable story once you factor in the class imbalance — MCC accounts for all four confusion-matrix cells jointly, while AUC only reflects ranking quality and can look strong even when the decision threshold is poorly calibrated for a skewed class split. |
| **Overall Winner for your dataset?** | Random Forest — highest MCC (0.5852) and competitive AUC (0.9215), the best-balanced tradeoff between catching buyers and not over-flagging non-buyers. Accuracy is intentionally not the ranking criterion: with only ~15% positive class, a model predicting "no purchase" for everyone would already score ~85% accuracy while being useless — MCC and AUC actually penalize that kind of imbalance-driven inflation. |

---
