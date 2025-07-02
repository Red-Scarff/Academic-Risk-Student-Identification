import logging
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# 预设特征解释
FEATURE_DESCRIPTIONS = {
    'early_engagement_score': "课程前4周的平均参与度",
    'total_active_days': "总活跃天数",
    'total_clicks': "总点击次数",
    'num_of_prev_attempts': "先前尝试次数",
    'date_unregistration': "退课日期",
    'studied_credits': "学习学分",
    'imd_band': "社会经济地位指数",
    'final_result': "最终成绩",
    'days_active_resource': "资源访问活跃天数",
    'clicks_resource': "资源点击总数",
    'gender': "性别",
    'region': "地区",
    'highest_education': "最高教育水平",
    'age_band': "年龄段",
    'disability': "是否有残疾",
    'semester': "学期",
    'code_module': "课程模块",
    'presentation_year': "开课年份",
    'avg_clicks_per_day': "日均点击量"
}

def setup_logger(model_name, log_dir):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/training.log'
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def evaluate_model_performance(model, X_test, y_test, logger):
    """全面评估模型性能"""
    logger.info("评估模型性能...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, target_names=['正常学生', '学业困难学生'])
    }
    
    logger.info(f"评估耗时: {time.time() - start_time:.2f}秒")
    logger.info("\n" + "="*60)
    logger.info("模型性能指标:")
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"精确率: {metrics['precision']:.4f}")
    logger.info(f"召回率: {metrics['recall']:.4f}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"AUC分数: {metrics['roc_auc']:.4f}")
    logger.info("\n混淆矩阵:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    logger.info("\n分类报告:")
    logger.info(f"\n{metrics['report']}")
    
    return metrics

def visualize_feature_importance(model, feature_names, output_dir, model_type):
    """可视化特征重要性"""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'random_forest':
        importances = model.feature_importances_
        title = "随机森林特征重要性"
    else:
        importances = np.abs(model.coef_[0])
        title = "逻辑回归特征重要性"
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(15), importances[indices][:15], align="center", color='orange')
    plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=45, ha="right")
    plt.xlim([-1, 15])
    plt.tight_layout()
    
    save_path = f'{output_dir}/feature_importance.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_roc_curve(y_test, y_prob, output_dir):
    """可视化ROC曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('受试者工作特征曲线')
    plt.legend(loc="lower right")
    
    save_path = f'{output_dir}/roc_curve.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path

def perform_shap_analysis(model, X_train, feature_names, logger, output_dir, model_type):
    """执行SHAP特征分析"""
    logger.info("执行SHAP特征分析...")
    os.makedirs(output_dir, exist_ok=True)
    
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
    
    if model_type == 'random_forest':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_train_df)
    
    sample_size = min(500, X_train_df.shape[0])
    X_sample = X_train_df.sample(sample_size, random_state=42)
    
    if model_type == 'random_forest':
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]
        else:
            shap_values_positive = shap_values
    else:
        shap_values = explainer(X_sample)
        shap_values_positive = shap_values.values
    
    if len(shap_values_positive.shape) == 3:
        shap_values_positive = shap_values_positive[:, :, 1]
    
    plt.figure(figsize=(16, 10))
    shap.summary_plot(shap_values_positive, X_sample, max_display=20, show=False, cmap=plt.get_cmap("rainbow"))
    plt.tight_layout()
    summary_path = f'{output_dir}/shap_summary.png'
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(16, 10))
    shap.summary_plot(shap_values_positive, X_sample, plot_type="bar", max_display=20, show=False, color='orange')
    plt.tight_layout()
    bar_path = f'{output_dir}/shap_bar.png'
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    
    mean_abs_shap = np.abs(shap_values_positive).mean(axis=0)
    
    if mean_abs_shap.ndim > 1:
        logger.warning("SHAP值数组是多维的，正在展平...")
        mean_abs_shap = mean_abs_shap.flatten()
    
    if len(feature_names) != len(mean_abs_shap):
        logger.warning(f"特征名称数量({len(feature_names)})与SHAP值数量({len(mean_abs_shap)})不匹配")
        min_len = min(len(feature_names), len(mean_abs_shap))
        feature_names = feature_names[:min_len]
        mean_abs_shap = mean_abs_shap[:min_len]

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': mean_abs_shap
    }).sort_values('shap_value', ascending=False)
    
    return {
        'summary_plot': summary_path,
        'bar_plot': bar_path,
        'feature_importance': shap_df
    }

def console_output(model_name, metrics, shap_df):
    """控制台简洁输出"""
    print(f"\n{model_name} 模型结果:")
    print("="*40)
    print(f"识别率: {metrics['recall']:.2%} | 准确率: {metrics['accuracy']:.2%}")
    print(f"F1分数: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    
    print("\n关键特征分析:")
    top_features = shap_df.head(5)
    for i, row in top_features.iterrows():
        feature = row['feature']
        shap_value = row['shap_value']
        desc = FEATURE_DESCRIPTIONS.get(feature, "重要行为特征")
        print(f"{i+1}. {feature}: {desc} (SHAP值: {shap_value:.4f})")
    print("="*40)