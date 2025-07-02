import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, RGCNConv, HeteroConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import optuna
import shap
from tqdm import tqdm
import random
from torch_geometric.explain import GNNExplainer
from torch_geometric.utils import to_networkx
import networkx as nx

# 1. 数据加载
DATA_PATH = 'log/preprocess/processed_data.csv'
print('加载数据...')
df = pd.read_csv(DATA_PATH)
print('数据加载完成，样本数:', len(df))

# 2. 特征与标签
feature_cols = [col for col in df.columns if col not in ['academic_risk', 'final_result', 'id_student', 'code_presentation', 'code_module']]
X = df[feature_cols].values
y = df['academic_risk'].values

# 标准化
print('标准化特征...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('特征标准化完成')

# ========== 构建异构图 ==========
data = HeteroData()

# 学生节点
num_students = len(df)
data['student'].x = torch.tensor(X_scaled, dtype=torch.float)
data['student'].y = torch.tensor(y, dtype=torch.long)

# 课程节点（每个课程+presentation唯一）
df['course_id'] = df['code_module'].astype(str) + '_' + df['code_presentation'].astype(str)
course_ids = df['course_id'].unique()
course_id2idx = {cid: i for i, cid in enumerate(course_ids)}
data['course'].x = torch.zeros((len(course_ids), X_scaled.shape[1]), dtype=torch.float)  # 可用课程内学生均值特征
for cid, idx in course_id2idx.items():
    course_mask = df['course_id'] == cid
    data['course'].x[idx] = torch.tensor(X_scaled[course_mask].mean(axis=0), dtype=torch.float)

# 学生-课程边（enrolled）
student2course = [course_id2idx[cid] for cid in df['course_id']]
data['student', 'enrolled', 'course'].edge_index = torch.tensor([
    np.arange(num_students),
    np.array(student2course)
], dtype=torch.long)

# 课程-学生反向边
data['course', 'has_student', 'student'].edge_index = torch.tensor([
    np.array(student2course),
    np.arange(num_students)
], dtype=torch.long)

# 学生-学生多关系边
MAX_NEIGHBORS = 100  # 每个节点在同组内最多连100个其他节点

print('构建同课程边...')
course_groups = df.groupby(['code_module', 'code_presentation']).groups
edge_index_course = []
for group in tqdm(course_groups.values(), desc='同课程group'):
    group = list(group)
    for i in group:
        neighbors = [j for j in group if j != i]
        if len(neighbors) > MAX_NEIGHBORS:
            neighbors = random.sample(neighbors, MAX_NEIGHBORS)
        for j in neighbors:
            edge_index_course.append([i, j])
print(f'同课程边数量: {len(edge_index_course)}')

print('构建同region边...')
region_groups = df.groupby(['region']).groups
edge_index_region = []
for group in tqdm(region_groups.values(), desc='同regiongroup'):
    group = list(group)
    for i in group:
        neighbors = [j for j in group if j != i]
        if len(neighbors) > MAX_NEIGHBORS:
            neighbors = random.sample(neighbors, MAX_NEIGHBORS)
        for j in neighbors:
            edge_index_region.append([i, j])
print(f'同region边数量: {len(edge_index_region)}')

print('构建同年龄段边...')
age_groups = df.groupby(['age_band']).groups
edge_index_age = []
for group in tqdm(age_groups.values(), desc='同年龄段group'):
    group = list(group)
    for i in group:
        neighbors = [j for j in group if j != i]
        if len(neighbors) > MAX_NEIGHBORS:
            neighbors = random.sample(neighbors, MAX_NEIGHBORS)
        for j in neighbors:
            edge_index_age.append([i, j])
print(f'同年龄段边数量: {len(edge_index_age)}')

print('构建同教育背景边...')
edu_groups = df.groupby(['highest_education']).groups
edge_index_edu = []
for group in tqdm(edu_groups.values(), desc='同教育group'):
    group = list(group)
    for i in group:
        neighbors = [j for j in group if j != i]
        if len(neighbors) > MAX_NEIGHBORS:
            neighbors = random.sample(neighbors, MAX_NEIGHBORS)
        for j in neighbors:
            edge_index_edu.append([i, j])
print(f'同教育背景边数量: {len(edge_index_edu)}')

# 添加多种学生-学生边
student_dim = data['student'].x.shape[1]
course_dim = data['course'].x.shape[1]
if edge_index_course:
    print(f'添加同课程边到异构图: {len(edge_index_course)}')
    data['student', 'same_course', 'student'].edge_index = torch.tensor(np.array(edge_index_course).T, dtype=torch.long)
if edge_index_region:
    print(f'添加同region边到异构图: {len(edge_index_region)}')
    data['student', 'same_region', 'student'].edge_index = torch.tensor(np.array(edge_index_region).T, dtype=torch.long)
if edge_index_age:
    print(f'添加同年龄段边到异构图: {len(edge_index_age)}')
    data['student', 'same_age', 'student'].edge_index = torch.tensor(np.array(edge_index_age).T, dtype=torch.long)
if edge_index_edu:
    print(f'添加同教育背景边到异构图: {len(edge_index_edu)}')
    data['student', 'same_edu', 'student'].edge_index = torch.tensor(np.array(edge_index_edu).T, dtype=torch.long)

# 训练/测试mask
train_idx, test_idx = train_test_split(np.arange(num_students), test_size=0.2, stratify=y, random_state=42)
train_mask = torch.zeros(num_students, dtype=torch.bool)
test_mask = torch.zeros(num_students, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
data['student'].train_mask = train_mask
data['student'].test_mask = test_mask
print('划分训练/测试集...')
print('训练集样本数:', train_mask.sum().item(), '测试集样本数:', test_mask.sum().item())

# ========== 定义HeteroGNN模型（支持异构图） ==========
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            rel: SAGEConv((-1, -1), hidden_dim)
            for rel in metadata[1]
        }, aggr='sum')
        self.conv2 = HeteroConv({
            rel: SAGEConv((-1, -1), out_dim)
            for rel in metadata[1]
        }, aggr='sum')
    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# ========== 训练与评估函数适配 ==========
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out_dict = model(data)
    out = out_dict['student']
    loss = criterion(out[data['student'].train_mask], data['student'].y[data['student'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out_dict = model(data)
    logits = out_dict['student']
    pred = logits.argmax(dim=1)
    y_true = data['student'].y[data['student'].test_mask].cpu().numpy()
    y_pred = pred[data['student'].test_mask].cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, logits[data['student'].test_mask][:,1].detach().cpu().numpy())
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return acc, f1, auc, report, y_true, y_pred

# ========== 主流程 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

print('初始化模型...')
hidden_dim = 128
lr = 0.005
weight_decay = 1e-4
model = HeteroGNN(data.metadata(), hidden_dim, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
print('模型初始化完成')

log_dir = 'log/GNN'
os.makedirs(log_dir, exist_ok=True)
train_log = []
test_log = []
best_model_path = os.path.join(log_dir, 'best_model.pt')
best_acc = 0
best_report = None
print('开始训练...')
for epoch in range(1, 101):
    loss = train(model, data, optimizer, criterion)
    if epoch % 10 == 0:
        acc, f1, auc, report, y_true, y_pred = test(model, data)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        train_log.append([epoch, loss])
        test_log.append([epoch, acc, f1, auc])
        if acc > best_acc:
            best_acc = acc
            best_report = report
            torch.save(model.state_dict(), best_model_path)
            best_y_true, best_y_pred = y_true, y_pred
print('训练结束，最佳准确率:', best_acc)

# 保存日志
pd.DataFrame(train_log, columns=['epoch', 'loss']).to_csv(os.path.join(log_dir, 'train_log.csv'), index=False)
pd.DataFrame(test_log, columns=['epoch', 'acc', 'f1', 'auc']).to_csv(os.path.join(log_dir, 'test_log.csv'), index=False)

# 可视化训练过程
plt.figure()
train_log_arr = pd.DataFrame(train_log, columns=['epoch', 'loss'])
plt.plot(train_log_arr['epoch'], train_log_arr['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(os.path.join(log_dir, 'train_loss_curve.png'))

plt.figure()
test_log_arr = pd.DataFrame(test_log, columns=['epoch', 'acc', 'f1', 'auc'])
plt.plot(test_log_arr['epoch'], test_log_arr['acc'], label='Test Acc')
plt.plot(test_log_arr['epoch'], test_log_arr['f1'], label='Test F1')
plt.plot(test_log_arr['epoch'], test_log_arr['auc'], label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test Metrics Curve')
plt.legend()
plt.savefig(os.path.join(log_dir, 'test_metrics_curve.png'))

# 输出最佳模型的混淆矩阵和分类报告
cm = confusion_matrix(best_y_true, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Best Model Confusion Matrix')
plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))

# 分类报告保存为表格图片
report_df = pd.DataFrame(best_report).T
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues')
plt.title('Classification Report (Best Model)')
plt.savefig(os.path.join(log_dir, 'classification_report.png'))

print('\nBest Test Classification Report:')
print(report_df)
# ========== 可视化子图 ==========
def visualize_subgraph(data, log_dir, num_samples=3):
    """可视化随机选择的子图"""
    os.makedirs(os.path.join(log_dir, 'subgraphs'), exist_ok=True)
    print("\n开始可视化子图...")
    
    print("可视化学生-学生关系图...")
    
    # 选择少量学生
    sampled_students = random.sample(range(num_students), min(100, num_students))
    
    # 创建学生子图
    G_student = nx.Graph()
    
    # 添加学生节点
    for student_idx in sampled_students:
        risk = data['student'].y[student_idx].item()
        G_student.add_node(f"S:{student_idx}", risk=risk)
    
    # 添加边（只添加在采样学生之间存在的边）
    relation_colors = {
        'same_course': 'blue',
        'same_region': 'green',
        'same_age': 'purple',
        'same_edu': 'orange'
    }
    
    for relation in ['same_course', 'same_region', 'same_age', 'same_edu']:
        if ('student', relation, 'student') in data.edge_types:
            edge_index = data['student', relation, 'student'].edge_index
            for src, dst in edge_index.t().tolist():
                if src in sampled_students and dst in sampled_students:
                    G_student.add_edge(
                        f"S:{src}", f"S:{dst}", 
                        relation=relation,
                        color=relation_colors.get(relation, 'gray')
                    )
    
    # 可视化
    plt.figure(figsize=(16, 14))
    pos = nx.spring_layout(G_student, seed=42)
    
    # 节点颜色根据学术风险着色
    node_colors = [
        'green' if G_student.nodes[node]['risk'] == 0 else 'orange' 
        for node in G_student.nodes()
    ]
    
    # 边颜色根据关系类型着色
    edge_colors = [G_student[u][v]['color'] for u, v in G_student.edges()]
    
    nx.draw(
        G_student, pos,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=True,
        node_size=300,
        alpha=0.8,
        font_size=9
    )
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low Risk'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='High Risk')
    ]
    
    for relation, color in relation_colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=relation))
    
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title('Student-Student Relationship Subgraph')
    plt.savefig(os.path.join(log_dir, 'subgraphs', 'student_relationships.png'))
    plt.close()
    
    print("学生关系图已保存")
    
    print("子图可视化完成")

# 在训练结束后调用可视化函数
print("\n开始可视化子图...")
visualize_subgraph(data, log_dir, num_samples=3)