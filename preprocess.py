import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 配置日志
LOG_DIR = 'log/preprocess'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f'{LOG_DIR}/preprocess.log'

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataPreprocessor')

def load_oulad_data(base_path='anonymisedData'):
    """加载OULAD数据集"""
    logger.info("正在加载OULAD数据集...")
    tables = {
        'assessments': 'assessments.csv',
        'courses': 'courses.csv',
        'studentAssessment': 'studentAssessment.csv',
        'studentInfo': 'studentInfo.csv',
        'studentRegistration': 'studentRegistration.csv',
        'studentVLE': 'studentVle.csv',
        'vle': 'vle.csv'
    }
    
    datasets = {}
    for name, file in tables.items():
        try:
            datasets[name] = pd.read_csv(os.path.join(base_path, file))
            logger.info(f"已加载 {name} ({datasets[name].shape[0]}行, {datasets[name].shape[1]}列)")
        except Exception as e:
            logger.error(f"加载失败 {file}: {str(e)}")
            raise
    
    return datasets

def merge_datasets(data_dict):
    """合并数据集"""
    logger.info("合并数据集...")
    merged = data_dict['studentInfo'].merge(
        data_dict['studentRegistration'],
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )
    
    merged = merged.merge(
        data_dict['courses'],
        on=['code_module', 'code_presentation'],
        how='left'
    )
    
    logger.info(f"合并后数据集: {merged.shape[0]}行, {merged.shape[1]}列")
    return merged

def process_missing_values(df):
    """处理缺失值"""
    logger.info("处理缺失值...")
    initial_missing = df.isnull().sum().sum()
    
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        logger.warning(f"删除空列: {empty_cols}")
        df = df.drop(columns=empty_cols)
    
    categorical_features = ['gender', 'region', 'highest_education', 
                           'imd_band', 'age_band', 'disability']
    df = df.dropna(subset=categorical_features)
    
    numeric_features = ['num_of_prev_attempts', 'studied_credits', 
                       'date_registration', 'date_unregistration']
    for feature in numeric_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    final_missing = df.isnull().sum().sum()
    logger.info(f"缺失值处理完成: {initial_missing} -> {final_missing}")
    return df

def integrate_vle_data(main_df, vle_data, student_vle_data):
    """整合VLE交互数据"""
    logger.info("整合VLE交互数据...")
    
    vle_merged = student_vle_data.merge(
        vle_data[['id_site', 'activity_type']],
        on='id_site',
        how='left'
    ).dropna(subset=['activity_type'])
    
    vle_agg = vle_merged.groupby(
        ['code_module', 'code_presentation', 'id_student', 'activity_type']
    ).agg(
        clicks=('sum_click', 'sum'),
        days_active=('date', 'nunique')
    ).reset_index()
    
    vle_pivot = vle_agg.pivot_table(
        index=['code_module', 'code_presentation', 'id_student'],
        columns='activity_type',
        values=['clicks', 'days_active'],
        fill_value=0
    )
    
    vle_pivot.columns = [f'{col[0]}_{col[1]}' for col in vle_pivot.columns]
    vle_pivot = vle_pivot.reset_index()
    
    result_df = main_df.merge(
        vle_pivot,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )
    
    vle_columns = [col for col in result_df.columns if 'clicks_' in col or 'days_active_' in col]
    result_df[vle_columns] = result_df[vle_columns].fillna(0)
    
    logger.info(f"整合后数据集: {result_df.shape[0]}行, {result_df.shape[1]}列")
    return result_df

def handle_extreme_values(df):
    """处理极端值"""
    logger.info("处理极端值...")
    plot_dir = f'{LOG_DIR}/preprocessing_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'date_registration',
                    'date_unregistration', 'module_presentation_length']
    vle_cols = [col for col in df.columns if 'clicks_' in col or 'days_active_' in col]
    numeric_cols.extend(vle_cols)
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[numeric_cols], color='orange')
    plt.xticks(rotation=45)
    plt.title('极端值处理前')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/extreme_values_before.png')
    plt.close()
    
    for col in numeric_cols:
        upper_limit = df[col].quantile(0.95)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[numeric_cols], color='orange')
    plt.xticks(rotation=45)
    plt.title('极端值处理后')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/extreme_values_after.png')
    plt.close()
    
    return df

def create_features(df):
    """创建新特征"""
    logger.info("创建新特征...")
    
    df['presentation_year'] = '20' + df['code_presentation'].str[:2]
    df['semester'] = df['code_presentation'].str[2].map({'B': '春季', 'J': '秋季'})
    
    click_cols = [col for col in df.columns if 'clicks_' in col]
    active_cols = [col for col in df.columns if 'days_active_' in col]
    
    df['total_clicks'] = df[click_cols].sum(axis=1)
    df['total_active_days'] = df[active_cols].sum(axis=1)
    df['avg_clicks_per_day'] = np.where(
        df['total_active_days'] > 0,
        df['total_clicks'] / df['total_active_days'],
        0
    )
    
    early_engagement_cols = [
        'days_active_homepage', 'clicks_homepage',
        'days_active_quiz', 'clicks_quiz',
        'days_active_resource', 'clicks_resource'
    ]
    
    df['early_engagement_score'] = df[early_engagement_cols].mean(axis=1)
    
    logger.info(f"新增特征: presentation_year, semester, total_clicks, total_active_days, avg_clicks_per_day, early_engagement_score")
    return df

def transform_categorical(df):
    """转换分类变量"""
    logger.info("转换分类变量...")
    
    df['academic_risk'] = df['final_result'].apply(
        lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0
    )
    
    categorical_cols = ['gender', 'code_presentation', 'region', 'highest_education', 
                       'imd_band', 'age_band', 'disability', 'semester', 
                       'code_module', 'final_result']
    
    encoder = preprocessing.LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))
            logger.info(f"已编码: {col} ({len(encoder.classes_)}个类别)")
    
    return df

def prepare_data_for_modeling(df, test_size=0.2, balance=True):
    """准备建模数据"""
    logger.info("准备建模数据...")
    
    X = df.drop(columns=['academic_risk', 'final_result', 'id_student', 'code_presentation'])
    y = df['academic_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(f"训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本")
    logger.info(f"学业困难比例 - 总体: {y.mean():.2%}, 训练集: {y_train.mean():.2%}, 测试集: {y_test.mean():.2%}")
    
    if balance:
        logger.info("应用SMOTE平衡数据...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"平衡后训练集: {X_train.shape[0]}个样本, 学业困难比例: {y_train.mean():.2%}")
    
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler
    }

def execute_preprocessing():
    """执行完整预处理流程"""
    logger.info("="*60)
    logger.info("开始数据预处理流程")
    logger.info("="*60)
    
    try:
        datasets = load_oulad_data()
        merged_data = merge_datasets(datasets)
        cleaned_data = process_missing_values(merged_data)
        vle_integrated = integrate_vle_data(cleaned_data, datasets['vle'], datasets['studentVLE'])
        outliers_handled = handle_extreme_values(vle_integrated)
        feature_engineered = create_features(outliers_handled)
        final_data = transform_categorical(feature_engineered)
        
        processed_path = f'{LOG_DIR}/processed_data.csv'
        final_data.to_csv(processed_path, index=False)
        logger.info(f"已保存处理后的数据到: {processed_path}")
        logger.info(f"最终数据集形状: {final_data.shape}")
        
        model_data = prepare_data_for_modeling(final_data)
        
        logger.info("="*60)
        logger.info("数据预处理完成")
        logger.info("="*60)
        
        return final_data, model_data
        
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    execute_preprocessing()