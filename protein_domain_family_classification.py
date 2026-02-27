import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 特徴量抽出（ジペプチド頻度）の関数定義
# ==========================================
def calculate_dipeptide_composition(sequences, amino_acids):
    """
    タンパク質配列のリストから、ジペプチド（2残基の組み合わせ）の出現頻度ベクトルを作成する関数。
    データセットのサイズに関わらず再利用可能。
    """
    # アミノ酸の直積（組み合わせ）を作成
    combination_list = ["".join(v) for v in itertools.product(amino_acids, amino_acids)]
    
    list_vec = []
    for seq in sequences: 
        # 各組み合わせが配列中にいくつ含まれるかをカウント
        vec = [seq.count(comb) for comb in combination_list]
        list_vec.append(vec)
        
    df_features = pd.DataFrame(list_vec, columns=combination_list)
    return df_features, combination_list


# アミノ酸の定義
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'B', 'Z', 'J', 'U', 'O', 'X', '*']

# ==========================================
# 2. 学習データの準備と特徴量作成
# ==========================================
filepath_train = "input_file.csv"
df_train_raw = pd.read_csv(filepath_train, sep='\t')
print(f"Training data shape: {df_train_raw.shape}")

# タンパク質配列（4列目）を取得し、自作関数で特徴量ベクトルに変換
train_sequences = df_train_raw.iloc[:, 3]
df_train_features, combination_list = calculate_dipeptide_composition(train_sequences, AMINO_ACIDS)

# 元データから配列の列を削除し、特徴量データフレームと結合
df_train_target = df_train_raw.drop(df_train_raw.columns[3], axis=1)
df_train_processed = pd.concat([df_train_target, df_train_features], axis=1)


# ==========================================
# 3. 分類モデルの構築と評価 (Neural Network)
# ==========================================
# 説明変数(X)と目的変数(y)の抽出
X = np.asarray(df_train_processed.iloc[:, 3:])
y = np.asarray(df_train_processed.iloc[:, 0].tolist())

# データの分割（テストデータ20%）
# random_stateを指定することで、誰が実行しても同じ分割結果になるよう再現性を担保
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データの標準化（スケーリング）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 学習データでスケールを計算して適用
X_test_scaled = scaler.transform(X_test)       # テストデータには学習データのスケールを適用

# ニューラルネットワークの定義と学習
mlp = MLPClassifier(hidden_layer_sizes=(10,), solver='adam', max_iter=500, verbose=True, random_state=42)
mlp.fit(X_train_scaled, y_train)

# モデルの評価
score = mlp.score(X_test_scaled, y_test)
print(f"Test accuracy: {score:.4f}")


# ==========================================
# 4. 新規配列に対する予測
# ==========================================
filepath_new = 'file_2.csv'
df_new_raw = pd.read_csv(filepath_new)

# 新規データのタンパク質配列（2列目と想定）を抽出
new_sequences = df_new_raw.iloc[:, 1]

# 関数の再利用：新規データも全く同じ処理で特徴量化できる
df_new_features, _ = calculate_dipeptide_composition(new_sequences, AMINO_ACIDS)

# 元データから配列の列を削除し、特徴量データフレームと結合
df_new_meta = df_new_raw.drop(df_new_raw.columns[1], axis=1)
df_new_processed = pd.concat([df_new_meta, df_new_features], axis=1)

# 予測用特徴量の抽出（1列目以降と想定）
X_new = df_new_processed.iloc[:, 1:].values

# 学習データで作成したscalerを使って標準化を行う
X_new_scaled = scaler.transform(X_new)

# 予測の実行
predictions = mlp.predict(X_new_scaled)
print("Predictions for new sequences:", predictions)