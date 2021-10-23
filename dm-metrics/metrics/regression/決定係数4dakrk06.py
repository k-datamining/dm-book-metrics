# %%NBQA-CELL-SEP5b3aad
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

RND = 777

# 表示する文字サイズを調整
plt.rc("font", size=20)
plt.rc("legend", fontsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)

# youtube動画を表示
import IPython.display

IPython.display.YouTubeVideo("koy1HmVfjvU", width="500px")


# %%NBQA-CELL-SEP5b3aad
X, y = make_regression(
    n_samples=1000,
    n_informative=3,
    n_features=20,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestRegressor(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)


# %%NBQA-CELL-SEP5b3aad
from sklearn.metrics import r2_score

r2 = r2_score(test_y, pred_y)
y_min, y_max = np.min(test_y), np.max(test_y)

plt.figure(figsize=(6, 6))
plt.title(f"$R^2 =${r2}")
plt.plot([y_min, y_max], [y_min, y_max], linestyle="-", c="k", alpha=0.2)
plt.scatter(test_y, pred_y, marker="x")
plt.xlabel("正解")
plt.ylabel("予測")


# %%NBQA-CELL-SEP5b3aad
X, y = make_regression(
    n_samples=1000,
    n_informative=3,
    n_features=20,
    effective_rank=4,
    noise=1.5,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestRegressor(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)


# %%NBQA-CELL-SEP5b3aad
r2 = r2_score(test_y, pred_y)
y_min, y_max = np.min(test_y), np.max(test_y)

plt.figure(figsize=(6, 6))
plt.title(f"$R^2 =${r2}")
plt.plot([y_min, y_max], [y_min, y_max], linestyle="-", c="k", alpha=0.2)
plt.scatter(test_y, pred_y, marker="x")
plt.xlabel("正解")
plt.ylabel("予測")


# %%NBQA-CELL-SEP5b3aad
X, y = make_regression(
    n_samples=1000,
    n_informative=3,
    n_features=20,
    effective_rank=4,
    noise=1.5,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

# train_yをランダムに並び替え、さらに値を変換する
train_y = np.random.permutation(train_y)
train_y = np.sin(train_y) * 10 + 1

model = RandomForestRegressor(max_depth=1)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)


# %%NBQA-CELL-SEP5b3aad
r2 = r2_score(test_y, pred_y)
y_min, y_max = np.min(test_y), np.max(test_y)

plt.figure(figsize=(6, 6))
plt.title(f"$R^2 =${r2}")
plt.plot([y_min, y_max], [y_min, y_max], linestyle="-", c="k", alpha=0.2)
plt.scatter(test_y, pred_y, marker="x")
plt.xlabel("正解")
plt.ylabel("予測")


# %%NBQA-CELL-SEP5b3aad
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

r2_scores = []
for i in range(100):
    # データ作成
    X, y = make_regression(
        n_samples=500,
        n_informative=1,
        n_features=1,
        effective_rank=4,
        noise=i * 0.1,
        random_state=RND,
    )
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.33, random_state=RND
    )

    # 線形回帰
    model = make_pipeline(
        StandardScaler(with_mean=False), LinearRegression(positive=True)
    ).fit(train_X, train_y)

    # 決定係数を算出
    pred_y = model.predict(test_X)
    r2 = r2_score(test_y, pred_y)
    r2_scores.append(r2)


plt.figure(figsize=(8, 4))
plt.title("100回ランダムなデータで線形回帰を実行した時の$R^2$")
plt.hist(r2_scores, bins=20)
plt.show()
