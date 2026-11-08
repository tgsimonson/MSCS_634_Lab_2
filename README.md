# MSCS 634 Lab 2: K-Nearest Neighbors and Radius Neighbors Classification

**Todd Simonson**
MSCS-634: Advanced Big Data and Data Mining
Dr. Satish Penmatsa

## Purpose

This lab compares two distance-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the Wine dataset from scikit-learn. The dataset has 178 wines from three cultivars, described by 13 chemical measurements. Each model was trained across a range of its key parameter (k for KNN, radius for RNN) to see how that choice affects test accuracy, and the two models were then compared to judge when each one is the better choice.

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_2.ipynb` | Notebook with data exploration, both models, plots, and discussion |
| `requirements.txt` | Python packages needed to run the notebook |

The Wine dataset ships with scikit-learn (`sklearn.datasets.load_wine`), so no data file is needed.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_2.ipynb
```

## Results

Data was split 80/20 (142 training, 36 testing), stratified by class, with `random_state=42`.

| KNN k | Test accuracy | | RNN radius | Test accuracy | Avg. neighbors used |
|---|---|---|---|---|---|
| 1 | 77.78% | | 350 | 72.22% | 81 (57% of training set) |
| 5 | 80.56% | | 400 | 69.44% | 88 |
| 11 | 80.56% | | 450 | 69.44% | 95 |
| 15 | 80.56% | | 500 | 69.44% | 100 |
| 21 | 80.56% | | 550 | 66.67% | 106 |
| | | | 600 | 66.67% | 111 (78% of training set) |

## Key Insights

- **KNN beat Radius Neighbors at every setting.** It led on the test set (77.8% to 80.6% vs. 66.7% to 72.2%) and in 5-fold cross-validation (70.4% to 75.4% vs. 64.1% to 66.2%).
- **KNN levels off after k = 5.** The only change across all five k values is one extra correct wine, between k = 1 and k = 5.
- **Radius Neighbors gets worse as the radius grows.** At the lab's radius values, every vote already includes 57% to 78% of the whole training set, so predictions drift toward the most common class. Its confusion matrix shows this: 9 of 10 class_2 test wines were labeled class_1, the larger class.
- **One feature controlled everything.** Proline ranges from 278 to 1,680, while most features range over a few units, and it accounted for **99.82%** of the squared distance between wines. On raw data, both models were effectively classifying by proline alone. That explains the error pattern. Class_0 was never misclassified, because even its 25th-percentile proline (988) is above the maximum of the other two classes. Class_1 and class_2, whose proline values overlap, account for every error.
- **Scaling mattered far more than k or radius.** After standardizing the features, KNN reached 97.2% to 100% at every k, and Radius Neighbors reached 100% at a radius of 3.5.
- **When to use each:**
  - **KNN** is the safer default when data density varies, feature scales are uncertain, or classes are imbalanced.
  - **Radius Neighbors** fits problems where distance has real-world meaning (for example, everything within 5 km) or where flagging points with no nearby neighbors is useful.

## Challenges and Decisions

1. **The specified radius values only make sense on raw features.** On standardized data, the median test-to-train distance is 5.13, so a radius of 350 would include every training wine. The required experiments were run on raw features as specified. A supplementary standardized comparison, with a radius range scaled to match (2.0 to 5.0), was added to show what the raw results were hiding.
2. **Explaining the results rather than just reporting them.** The accuracy trends alone did not show why Radius Neighbors underperformed. Counting the neighbors inside each radius, and measuring each feature's share of the distance calculation, turned the observation ("accuracy drops as the radius grows") into a mechanism ("the vote becomes nearly global and proline dominates the distance").
3. **A small test set.** With 36 test wines, one misclassification changes accuracy by 2.78 points, so the gap between k = 1 and k = 5 is a single wine. Stratified 5-fold cross-validation on the training set was added to check which trends hold up. It confirmed that KNN beats Radius Neighbors and that Radius Neighbors declines with radius. It also showed that the differences between k values are smaller than the fold-to-fold variation (3.9 to 6.8 points), so no single k is clearly best on raw data.
4. **Test points with no neighbors.** Radius Neighbors cannot make a prediction for a test point with no training points inside the radius. `outlier_label='most_frequent'` was set so those cases fall back to the most common class instead of stopping with an error. The fallback was never needed at the lab's radius values (every test wine had at least 4 neighbors), but it was needed for small radii on standardized data: at a radius of 2.0, 16 of 36 test wines had no neighbors.
5. **Preventing data leakage.** The scaler for the standardized comparison was fit on the training set only, so no information from the test set influenced training.
