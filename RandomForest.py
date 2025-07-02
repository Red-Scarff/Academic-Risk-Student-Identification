import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import setup_logger, evaluate_model_performance, visualize_feature_importance, visualize_roc_curve, perform_shap_analysis, console_output, FEATURE_DESCRIPTIONS
import matplotlib as mpl
import joblib
import os

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def train_random_forest():
    """训练随机森林模型"""
    LOG_DIR = 'log/RF'
    logger = setup_logger('RandomForest', LOG_DIR)
    
    TOP_FEATURES_TO_EXPLAIN = 5
    
    logger.info("="*60)
    logger.info("学业困难学生识别 - 随机森林模型")
    logger.info("="*60)
    logger.info("算法类型: 集成学习")
    logger.info("特点: 高准确性、抗过拟合、特征重要性评估")
    
    logger.info("\n" + "-"*60)
    logger.info("特征预设解释:")
    for feature, description in FEATURE_DESCRIPTIONS.items():
        logger.info(f"{feature}: {description}")
    
    logger.info("\n" + "-"*60)
    logger.info("数据加载阶段")
    logger.info("="*30)
    try:
        df = pd.read_csv('log/preprocess/processed_data.csv')
        logger.info(f"数据集形状: {df.shape}")
        logger.info(f"学业困难比例: {df['academic_risk'].mean():.2%}")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return
    
    X = df.drop(columns=['academic_risk'])
    y = df['academic_risk']
    feature_names = X.columns.tolist()
    
    logger.info("\n" + "-"*60)
    logger.info("数据准备阶段")
    logger.info("="*30)
    logger.info("划分训练集和测试集 (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"训练集: {X_train.shape[0]}样本, 测试集: {X_test.shape[0]}样本")
    logger.info(f"学业困难比例 - 训练集: {y_train.mean():.2%}, 测试集: {y_test.mean():.2%}")
    
    logger.info("标准化特征数据...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    
    rf_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info("\n" + "-"*60)
    logger.info("模型配置")
    logger.info("="*30)
    logger.info(f"树的数量: {rf_params['n_estimators']}")
    logger.info(f"最大深度: {rf_params['max_depth']}")
    logger.info(f"最小分裂样本数: {rf_params['min_samples_split']}")
    logger.info(f"最小叶子样本数: {rf_params['min_samples_leaf']}")
    logger.info(f"特征选择方式: {rf_params['max_features']}")
    logger.info(f"使用自助采样: {rf_params['bootstrap']}")
    
    logger.info("\n" + "-"*60)
    logger.info("模型训练阶段")
    logger.info("="*30)
    logger.info("开始训练随机森林模型...")
    start_time = time.time()
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_scaled_df, y_train)
    train_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {train_time:.2f}秒")
    
    model_path = f'{LOG_DIR}/academic_risk_model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"模型已保存至: {model_path}")
    
    logger.info("\n" + "-"*60)
    logger.info("模型评估阶段")
    logger.info("="*30)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    metrics = evaluate_model_performance(model, X_test_scaled_df, y_test, logger)
    
    fi_path = visualize_feature_importance(model, feature_names, LOG_DIR, 'random_forest')
    logger.info(f"特征重要性图已保存至: {fi_path}")
    
    y_prob = model.predict_proba(X_test_scaled_df)[:, 1]
    roc_path = visualize_roc_curve(y_test, y_prob, LOG_DIR)
    logger.info(f"ROC曲线已保存至: {roc_path}")
    
    logger.info("\n" + "-"*60)
    logger.info("模型解释性分析")
    logger.info("="*30)
    shap_dir = f'{LOG_DIR}/shap_analysis'
    shap_results = perform_shap_analysis(model, X_train_scaled_df, feature_names, logger, shap_dir, 'random_forest')
    
    logger.info(f"SHAP摘要图: {shap_results['summary_plot']}")
    logger.info(f"SHAP条形图: {shap_results['bar_plot']}")
    
    logger.info("\n特征重要性分析:")
    top_features = shap_results['feature_importance'].head(10)
    logger.info("\nTop 10重要特征:")
    logger.info(top_features.to_string(index=False))
    
    logger.info(f"\n详细解释前{TOP_FEATURES_TO_EXPLAIN}个重要特征:")
    for _, row in top_features.head(TOP_FEATURES_TO_EXPLAIN).iterrows():
        feature = row['feature']
        importance = row['shap_value']
        description = FEATURE_DESCRIPTIONS.get(feature, "重要行为特征")
        logger.info(f"- {feature} (SHAP值: {importance:.4f}): {description}")
    
    logger.info("\n" + "="*60)
    logger.info("模型训练总结")
    logger.info("="*60)
    logger.info(f"学业困难识别率 (召回率): {metrics['recall']:.2%}")
    logger.info(f"模型准确率: {metrics['accuracy']:.2%}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"AUC分数: {metrics['roc_auc']:.4f}")
    logger.info("\n训练完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("="*60)
    
    console_output("随机森林", metrics, shap_results['feature_importance'])

if __name__ == "__main__":
    train_random_forest()