#ステップ1：公共データ（相当）の読み込み
#このデータセットには、30種類の計測値（遺伝子発現量のような特徴量）と、それが「悪性（Malignant）」か「良性（Benign）」かというラベルが含まれています。

import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. データの読み込み
cancer = load_breast_cancer()

# PandasのDataFrameに変換して扱いやすくする
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# データの先頭5行を表示して中身を確認
print("データ形状:", df.shape) # (サンプル数, 特徴量数+1)
print("\n特徴量の名前（最初の5つ）:")
print(df.columns[:5].tolist())

# 最初の数行を表示
df.head()


#ステップ2：データの加工（標準化と変換）

import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 特徴量(X)とターゲット(y)に分ける
X = cancer.data
y = cancer.target

# 2. 対数変換のシミュレーション
# 実際のRNA-Seqデータ（カウントデータ）は値の幅が非常に広いため、
# log2(count + 1) のような変換を行うのが一般的です。
# ※今回のデータセットはすでに加工済みですが、練習として1を加算して対数変換するコードを書きます。
X_log = np.log1p(X)

# 3. 標準化（Standardization）
# 各遺伝子（特徴量）の平均を0、分散を1に変換します。
# これをしないと、値の大きな遺伝子だけが機械学習の結果に強く影響してしまいます。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 加工後のデータを確認
print(f"加工前の最初のデータの値（一部）:\n{X[0][:3]}")
print(f"標準化後の最初のデータの値（一部）:\n{X_scaled[0][:3]}")

# DataFrameに戻して確認しやすくする
df_scaled = pd.DataFrame(X_scaled, columns=cancer.feature_names)
df_scaled['target'] = y

print("\n加工後のデータの統計量（平均がほぼ0になっていることを確認）:")
print(df_scaled.iloc[:, :3].describe().round(2))


#ステップ3：データを図表にして表示する（PCA編）
#以下のコードを実行して、先ほど標準化したデータ（X_scaled）を2次元に圧縮し、プロットしてみましょう。

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. PCAの実行
# 特徴量（30次元）を第2主成分（2次元）まで圧縮
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2. プロット用のDataFrame作成
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y
# ラベル名をわかりやすく変換（0: Malignant/悪性, 1: Benign/良性）
pca_df['Diagnosis'] = pca_df['Target'].map({0: 'Malignant', 1: 'Benign'})

# 3. 散布図で可視化
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Diagnosis', data=pca_df, palette='Set1', alpha=0.7)

# 寄与率（各主成分がどれくらい元の情報を保持しているか）を取得
explained_variance = pca.explained_variance_ratio_ * 100

plt.title(f"PCA of Breast Cancer Dataset\n(PC1: {explained_variance[0]:.1f}%, PC2: {explained_variance[1]:.1f}%)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


#ステップ4：機械学習（ランダムフォレストによる分類）
#標準化したデータを用いて、悪性か良性かを判定するモデルを作ります。

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. ランダムフォレストモデルの構築と学習
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. テストデータでの評価
y_pred = rf_model.predict(X_test)
print(f"モデルの精度 (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 4. 【重要】どの遺伝子（特徴量）が判定に寄与したかを表示
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Top 10)")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [cancer.feature_names[i] for i in indices[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.show()


#ステップ5：仮想データを用いて機械学習が正しく行われているか判断する
#モデルが本当に「悪性（Malignant）の特徴」を理解しているか、極端な値を持つ仮想のサンプルを作ってテストしてみます。

# --- 仮想サンプルの作成 ---
# 例：悪性（target=0）のサンプルの平均的な特徴量を取得し、さらに「悪性度を高める方向」に少し加工
malignant_mean = X_scaled[y == 0].mean(axis=0)
benign_mean = X_scaled[y == 1].mean(axis=0)

# 仮想サンプル1：平均的な悪性サンプル
virtual_s1 = malignant_mean.reshape(1, -1)

# 仮想サンプル2：平均的な良性サンプル
virtual_s2 = benign_mean.reshape(1, -1)

# 仮想サンプル3：ランダムなノイズを加えた未知のサンプル
virtual_s3 = (malignant_mean + np.random.normal(0, 0.5, malignant_mean.shape)).reshape(1, -1)

# 予測の実行
virtual_samples = np.vstack([virtual_s1, virtual_s2, virtual_s3])
virtual_preds = rf_model.predict(virtual_samples)
virtual_probs = rf_model.predict_proba(virtual_samples) # 確信度（確率）

print("--- 仮想データ検証結果 ---")
target_labels = cancer.target_names # ['malignant', 'benign']
for i, (pred, prob) in enumerate(zip(virtual_preds, virtual_probs)):
    print(f"仮想サンプル {i+1}: 予測結果 = {target_labels[pred]} (確信度: {prob[pred]:.2f})")