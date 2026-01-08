import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# ===== CSV データの読み込み =====
label_df = pd.read_csv("/Users/horikawafuka2/Documents/class_2025/sotuken/models/lstm_label_results.csv")
entire_df = pd.read_csv("/Users/horikawafuka2/Documents/class_2025/sotuken/models/lstm_sample_results.csv")

# ===== 折れ線グラフ（Precision） =====
plt.figure(figsize=(10, 6))
plt.plot(label_df['samples_per_class'], label_df['fall_precision_mean'], marker='o', label='fall')
plt.plot(label_df['samples_per_class'], label_df['stand_precision_mean'], marker='o', label='stand')
plt.plot(label_df['samples_per_class'], label_df['sit_precision_mean'], marker='o', label='sit')
plt.xlabel("ラベル別サンプル数")
plt.ylabel("適合率（Precision）")
plt.title("サンプル数と適合率の関係")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 折れ線グラフ（Recall） =====
plt.figure(figsize=(10, 6))
plt.plot(label_df['samples_per_class'], label_df['fall_recall_mean'], marker='o', label='fall')
plt.plot(label_df['samples_per_class'], label_df['stand_recall_mean'], marker='o', label='stand')
plt.plot(label_df['samples_per_class'], label_df['sit_recall_mean'], marker='o', label='sit')
plt.xlabel("ラベル別サンプル数")
plt.ylabel("再現率（Recall）")
plt.title("サンプル数と再現率の関係")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 折れ線グラフ（F1-score） =====
plt.figure(figsize=(10, 6))
plt.plot(label_df['samples_per_class'], label_df['fall_f1_mean'], marker='o', label='fall')
plt.plot(label_df['samples_per_class'], label_df['stand_f1_mean'], marker='o', label='stand')
plt.plot(label_df['samples_per_class'], label_df['sit_f1_mean'], marker='o', label='sit')
plt.xlabel("ラベル別サンプル数")
plt.ylabel("f1値（F1 Score）")
plt.title("サンプル数とf1値の関係")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 折れ線グラフ（精度（平均）） =====
plt.figure(figsize=(10, 6))
plt.plot(entire_df['samples_per_class'], entire_df['mean_accuracy'], marker='o', label='精度（平均）')
plt.xlabel("ラベル別サンプル数")
plt.ylabel("精度（Mean accuracy）")
plt.title("サンプル数と精度(平均)の関係")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 折れ線グラフ（精度（平均）） =====
plt.figure(figsize=(10, 6))
plt.plot(entire_df['samples_per_class'], entire_df['std_accuracy'], marker='o', label='精度（標準偏差）')
plt.xlabel("ラベル別サンプル数")
plt.ylabel("精度（Std accuracy）")
plt.title("サンプル数と精度(標準偏差)の関係")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()