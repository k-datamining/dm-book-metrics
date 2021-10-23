# %%NBQA-CELL-SEP5b3aad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

RND = 777

# 表示する文字サイズを調整
plt.rc("font", size=20)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)

# youtube動画を表示
import IPython.display

IPython.display.YouTubeVideo("mU3L6gvt57g", width="500px")


# %%NBQA-CELL-SEP5b3aad
def plot_roc_curve(test_y, pred_y):
    """正解と予測からROC Curveをプロット

    Args:
        test_y (ndarray of shape (n_samples,)): テストデータの正解
        pred_y (ndarray of shape (n_samples,)): テストデータに対する予測値
    """
    # False Positive Rate, True Positive Rateを計算
    fprs, tprs, thresholds = roc_curve(test_y, pred_y)

    # ROCをプロット
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="-", c="k", alpha=0.2, label="ROC-AUC=0.5")
    plt.plot(fprs, tprs, color="orange", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # ROC-AUCスコアに相当する部分を塗りつぶす
    y_zeros = [0 for _ in tprs]
    plt.fill_between(fprs, y_zeros, tprs, color="orange", alpha=0.3, label="ROC-AUC")
    plt.legend()
    plt.show()


# %%NBQA-CELL-SEP5b3aad
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=4,
    n_clusters_per_class=3,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestClassifier(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict_proba(test_X)[:, 1]
plot_roc_curve(test_y, pred_y)


# %%NBQA-CELL-SEP5b3aad
from sklearn.metrics import roc_auc_score

roc_auc_score(test_y, pred_y)
