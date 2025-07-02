import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===================== 输出路径 =====================
output_dir = 'log/cluster'
os.makedirs(output_dir, exist_ok=True)

# ===================== 数据加载 =====================
# 直接从预处理函数获取处理后的数据
final_data = pd.read_csv('log/preprocess/processed_data.csv')

# 准备无监督聚类输入
drop_cols = ['academic_risk', 'final_result', 'id_student', 'code_presentation']
X = final_data.drop(columns=drop_cols)
true_labels = final_data['academic_risk']

# ===================== 训练计时 =====================
start_train = time.time()
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
end_train = time.time()
train_time = end_train - start_train
print(f'训练时间: {train_time:.4f} 秒')

# ===================== 预测计时 =====================
start_predict = time.time()
clusters = kmeans.predict(X)
end_predict = time.time()
predict_time = end_predict - start_predict
print(f'预测时间: {predict_time:.4f} 秒')

# ===================== 混淆矩阵与准确率 =====================
cm = confusion_matrix(true_labels, clusters)
print('混淆矩阵：')
print(cm)

accuracy_option1 = (cm[0, 0] + cm[1, 1]) / cm.sum()
accuracy_option2 = (cm[0, 1] + cm[1, 0]) / cm.sum()
accuracy = max(accuracy_option1, accuracy_option2)
print(f'聚类准确率（基于最佳标签匹配）: {accuracy:.4f}')

# ===================== 轮廓系数 =====================
silhouette = silhouette_score(X, clusters)
print(f'轮廓系数: {silhouette:.4f}')

# ===================== 模型大小 =====================
model_path = os.path.join(output_dir, 'kmeans_model.pkl')
joblib.dump(kmeans, model_path)
model_size = os.path.getsize(model_path) / 1024
print(f'模型大小: {model_size:.2f} KB')

# ===================== 保存实验结果 =====================
cm_df = pd.DataFrame(cm, index=['非学业困难', '学业困难'], columns=['聚类标签 0', '聚类标签 1'])
cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'), encoding='utf-8-sig')

with open(os.path.join(output_dir, 'cluster_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f'训练时间: {train_time:.4f} 秒\n')
    f.write(f'预测时间: {predict_time:.4f} 秒\n')
    f.write(f'聚类准确率: {accuracy:.4f}\n')
    f.write(f'轮廓系数: {silhouette:.4f}\n')
    f.write(f'模型大小: {model_size:.2f} KB\n')

# ===================== PCA 可视化 =====================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=10)
plt.title('K-means 聚类结果 (n_clusters=2)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(label='Cluster Label')
plt.savefig(os.path.join(output_dir, 'cluster_pca_visualization.png'))
plt.show()

# ===================== 热力图 =====================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('聚类标签与学业困难标签对比')
plt.xlabel('聚类标签')
plt.ylabel('学业困难标签')
plt.savefig(os.path.join(output_dir, 'cluster_vs_label_heatmap.png'))
plt.show()

# ===================== 多聚类数量轮廓系数 =====================
cluster_range = range(2, 7)
scores = []

for k in cluster_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, scores, marker='o')
plt.title('不同聚类数量的轮廓系数')
plt.xlabel('聚类数量')
plt.ylabel('轮廓系数')
plt.grid()
plt.savefig(os.path.join(output_dir, 'cluster_silhouette_scores.png'))
plt.show()

silhouette_df = pd.DataFrame({'聚类数量': list(cluster_range), '轮廓系数': scores})
silhouette_df.to_csv(os.path.join(output_dir, 'silhouette_scores.csv'), index=False, encoding='utf-8-sig')