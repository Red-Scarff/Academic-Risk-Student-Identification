import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
from preprocess import prepare_data_for_modeling

# ===================== 输出路径 =====================
output_dir = 'log/svm'
os.makedirs(output_dir, exist_ok=True)

# ===================== 数据加载与划分 =====================
# 执行预处理，划分训练测试集（使用平衡数据）
data = pd.read_csv('log/preprocess/processed_data.csv')
model_data = prepare_data_for_modeling(data)

X_train = model_data['X_train']
X_test = model_data['X_test']
y_train = model_data['y_train']
y_test = model_data['y_test']

# ===================== 训练计时 =====================
start_train = time.time()
svm = SVC(kernel='rbf', probability=True, random_state=42)  # RBF 核函数
svm.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
print(f'训练时间: {train_time:.4f} 秒')

# ===================== 预测计时 =====================
start_predict = time.time()
y_pred = svm.predict(X_test)
end_predict = time.time()
predict_time = end_predict - start_predict
print(f'预测时间: {predict_time:.4f} 秒')

# ===================== 模型大小 =====================
model_path = os.path.join(output_dir, 'svm_model.pkl')
joblib.dump(svm, model_path)
model_size = os.path.getsize(model_path) / 1024  # 单位 KB
print(f'模型大小: {model_size:.2f} KB')

# ===================== 混淆矩阵 =====================
cm = confusion_matrix(y_test, y_pred)
print('混淆矩阵：')
print(cm)

# 保存混淆矩阵
cm_df = pd.DataFrame(cm, index=['非学业困难', '学业困难'], columns=['预测非学业困难', '预测学业困难'])
cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'), encoding='utf-8-sig')

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM 混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig(os.path.join(output_dir, 'svm_confusion_matrix.png'))
plt.show()

# ===================== 评估指标 =====================
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = report['accuracy']
precision = report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']

print(f'准确率: {accuracy:.4f}')
print(f'精确率: {precision:.4f}')
print(f'召回率: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# 保存指标
with open(os.path.join(output_dir, 'svm_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f'训练时间: {train_time:.4f} 秒\n')
    f.write(f'预测时间: {predict_time:.4f} 秒\n')
    f.write(f'模型大小: {model_size:.2f} KB\n')
    f.write(f'准确率: {accuracy:.4f}\n')
    f.write(f'精确率: {precision:.4f}\n')
    f.write(f'召回率: {recall:.4f}\n')
    f.write(f'F1-score: {f1:.4f}\n')

# ===================== ROC 曲线 =====================
y_prob = svm.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('SVM ROC 曲线')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(os.path.join(output_dir, 'svm_roc_curve.png'))
plt.show()