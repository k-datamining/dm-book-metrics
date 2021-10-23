# %%NBQA-CELL-SEP5b3aad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

RND = 777

# 表示する文字サイズを調整
plt.rc("font", size=20)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)

# youtube動画を表示
import IPython.display

# IPython.display.YouTubeVideo("XXX", width="500px")


# %%NBQA-CELL-SEP5b3aad
X, y = make_classification(
    n_samples=300,
    n_classes=2,
    n_informative=4,
    n_features=6,
    weights=[0.2, 0.8],
    n_clusters_per_class=2,
    shuffle=True,
    random_state=RND,
)

train_valid_X, test_X, train_valid_y, test_y = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)


# %%NBQA-CELL-SEP5b3aad
train_X, valid_X, train_y, valid_y = train_test_split(
    train_valid_X, train_valid_y, test_size=0.2, random_state=RND
)

model = RandomForestClassifier(max_depth=4, random_state=RND)
model.fit(train_X, train_y)
pred_y = model.predict(valid_X)
rocauc = roc_auc_score(valid_y, pred_y)
print(f"ROC-AUC = {rocauc}")


# %%NBQA-CELL-SEP5b3aad
metrics = ("roc_auc", "accuracy")
model = RandomForestClassifier(max_depth=4, random_state=RND)
cv_scores = cross_validate(
    model, train_valid_X, train_valid_y, cv=5, scoring=metrics, return_train_score=True
)

for m in metrics:
    cv_m = cv_scores[f"test_{m}"]
    print(f"{m} {np.mean(cv_m)}")


# %%NBQA-CELL-SEP5b3aad
model = RandomForestClassifier(max_depth=4, random_state=RND).fit(
    train_valid_X, train_valid_y
)
pred_y = model.predict(test_X)
rocauc = roc_auc_score(test_y, pred_y)
print(f"test ROC-AUC = {rocauc}")
