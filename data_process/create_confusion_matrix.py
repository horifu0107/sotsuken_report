import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


for num_of_sample in [100,200,300,400,500,600,700]:

    # =====================
    # パス設定
    # =====================
    CSV_PATH = f"/Users/horikawafuka2/Documents/class_2025/sotuken/confusion_matrix_results/confusion_matrix_fss{num_of_sample}.csv"
    OUTPUT_PATH = f"/Users/horikawafuka2/Documents/class_2025/sotuken/report/graph/confusion_matrix_fss{num_of_sample}_ratio.png"

    # =====================
    # CSV 読み込み
    # =====================
    df_cm = pd.read_csv(CSV_PATH, index_col=0)
    
    labels = df_cm.columns.tolist()
    true_labels = df_cm.index.tolist()
    
    cm = df_cm.values.astype(float)
    
    # =====================
    # 行方向で割合に正規化
    # =====================
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_ratio = cm / cm_sum * 100  # %
    
    # =====================
    # 図の作成（配色変更）
    # =====================
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_ratio, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(label="Percentage (%)")
    
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(true_labels)), true_labels)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Normalized Confusion Matrix (FSS = {num_of_sample})")
    
    # =====================
    # セルの境界線を追加
    # =====================
    plt.gca().set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, len(true_labels), 1), minor=True)
    plt.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    plt.tick_params(which="minor", bottom=False, left=False)
    
    # =====================
    # セル内の文字（常に黒）
    # =====================
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                f"{cm_ratio[i, j]:.1f}%\n({int(cm[i, j])})",
                ha="center", va="center",
                color="black",
                fontsize=11
            )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()
    plt.close()
    
    print(f"見やすい混同行列画像を保存しました:\n{OUTPUT_PATH}")