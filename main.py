# Author: Ethan FAN
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_curve, auc, classification_report, \
    confusion_matrix, roc_auc_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from sklearn.utils.class_weight import compute_class_weight

DATA_FILE = "data/abalone.data"
IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    try:
        os.makedirs(IMAGE_DIR)
        print(f"Successfully created directory: {IMAGE_DIR}")
    except OSError as e:
        print(f"Error creating directory {IMAGE_DIR}: {e}")
else:
    print(f"Directory already exists: {IMAGE_DIR}")

"""
Predict the Ring age in years

Attribute       Type        Unit    Description
Sex             Nominal     --      M, F, and I (infant)
Length          Continuous  mm      Longest shell measurement
Diameter        Continuous  mm      Perpendicular to length
Height          Continuous  mm      With meat in shell
Whole weight    Continuous  grams   Whole abalone
Shucked weight  Continuous  grams   Weight of meat
Viscera weight  Continuous  grams   Gut weight (after bleeding)
Shell weight    Continuous  grams   After being dried
Rings           Integer     --      Raw data (ignore +1.5)

Ignore the +1.5 in ring-age and use the raw data

           Length   Diam  Height  Whole  Shucked  Viscera  Shell   Rings
Statistic                                                               
Min         0.075  0.055   0.000  0.002    0.001    0.001  0.002   1.000
Max         0.815  0.650   1.130  2.826    1.488    0.760  1.005  29.000
Mean        0.524  0.408   0.140  0.829    0.359    0.181  0.239   9.934
SD          0.120  0.099   0.042  0.490    0.222    0.110  0.139   3.224
Correl      0.557  0.575   0.557  0.540    0.421    0.504  0.628   1.000

Source: https://archive.ics.uci.edu/ml/datasets/abalone
"""


def question_decorator(part_number, question_number):
    """
    装饰器，主要目的是使得各个部分的答案更加容易被识别
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if part_number == 1:
                part = "Data processing"
            elif part_number == 2:
                part = "Modelling"
            else:
                part = ""
            print(f"{'=' * 10} Answer for {part} Q{question_number}. {'=' * 10}")
            result = func(*args, **kwargs)
            print(f"{'=' * 10} End of {part} Q{question_number}. {'=' * 10}\n")
            return result

        return wrapper

    return decorator


@question_decorator(1, 1)
def get_and_clean_data():
    """
    读取原始数据，进行清理处理，并保存清理后的数据。

    此函数执行以下操作：
    1. 定义数据列名
    2. 从指定文件读取CSV数据
    3. 将'Sex'列的字符值映射为数值
    4. 将清理后的数据保存为新的CSV文件
    5. 打印处理信息和数据预览

    :return: 处理后的DataFrame，包含清理和转换后的数据
    """
    # 定义数据列名
    column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight",
                    "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

    # 读取CSV文件，使用指定的列名
    data_format = pd.read_csv(DATA_FILE, names=column_names)

    # 定义性别到数值的映射
    gender_map = {'M': -1, 'F': 1, 'I': 0}

    # 将'Sex'列的值映射为数值
    data_format['Sex'] = data_format['Sex'].map(gender_map)

    # 检查在替换后是否存在空值
    print(f"Shape before cleaning: {data_format.shape}")
    for name in data_format.columns[1:]:
        print(f"{name} column error number:{data_format[name].isna().sum() + (data_format[name] <= 0).sum()}")

    data_format = data_format[(data_format['Height'] > 0) & (~data_format.isna().any(axis=1))]
    print(f"Shape after cleaning: {data_format.shape}")

    # 将清理后的数据保存为新的CSV文件
    data_format.to_csv("data/cleaned_data.csv", index=False)

    # 打印处理信息和数据预览
    print(f"Data has been cleaned and saved to data/cleaned_data.csv. \n\nThe top 5 data: \n{data_format.head(n=5)}")

    # 返回处理后的DataFrame
    return data_format


@question_decorator(1, 2)
def plot_correlation_hot(data):
    """
    绘制数据相关性热图并保存为图片文件。

    该函数计算输入数据的相关性矩阵，然后使用热图可视化这个矩阵。
    相关性值会直接显示在热图上。最后，图像会被保存为PNG文件。

    :param data: pandas DataFrame，包含要分析相关性的数据
    :return: pandas DataFrame，包含计算得到的相关性矩阵
    """
    # 计算相关性矩阵
    corr_mat = data.corr()

    # 创建图形并设置大小
    plt.figure(figsize=(12, 10))

    # 使用热图显示相关性矩阵
    plt.imshow(corr_mat, interpolation='nearest', cmap='viridis')
    plt.colorbar()  # 添加颜色条
    plt.title("Correlation Matrix")

    # 设置x轴和y轴的标签
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)

    # 在热图的每个单元格中添加相关性值
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            value = corr_mat.iloc[i, j]
            color = "white" if value < 0.3 else "black"
            plt.text(j, i, f'{value:.3f}',
                     ha="center", va="center", color=color,
                     fontweight='bold')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig('images/correlation_hot.png')
    plt.close()

    print(f"Correlation hot map has been saved in images/correlation_hot.png.")

    return corr_mat


@question_decorator(1, 3)
def analyze_and_plot_correlations(data, corr_matrix):
    """
    分析数据相关性并绘制散点图。

    此函数执行以下操作：
    1. 找出与 'Rings' 相关性最高的两个特征
    2. 绘制这两个特征与 'Rings' 的散点图
    3. 保存散点图
    4. 打印分析结果和主要观察结果

    :param data: pandas DataFrame，包含原始数据
    :param corr_matrix: pandas DataFrame，包含相关性矩阵
    :return: tuple，包含与 'Rings' 相关性最高的两个特征名
    """
    # 计算各特征与 'Rings' 的绝对相关性，并排除 'Rings' 自身
    corr_with_rings = corr_matrix['Rings'].drop('Rings').abs()

    # 找出相关性最高的两个特征
    top_two_features = corr_with_rings.nlargest(2)
    first_feature = top_two_features.index[0]
    second_feature = top_two_features.index[1]

    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制第一个特征与 'Rings' 的散点图
    ax1.scatter(data[first_feature], data['Rings'], alpha=0.8)
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Rings')
    ax1.set_title(f'Rings vs {first_feature} ')

    # 绘制第二个特征与 'Rings' 的散点图
    ax2.scatter(data[second_feature], data['Rings'], alpha=0.8)
    ax2.set_xlabel(second_feature)
    ax2.set_ylabel('Rings')
    ax2.set_title(f'Rings vs {second_feature} ')

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('images/correlation_scatter_plots.png')
    plt.close()

    # 打印结果
    print(f"Correlation scatter map has been saved in images/correlation_scatter_plots.png.")
    print(f"Top two correlated features with Rings:")
    print(f"1. {first_feature}: correlation of {corr_with_rings[first_feature]:.3f}")
    print(f"2. {second_feature}: correlation of {corr_with_rings[second_feature]:.3f}")

    # 打印主要观察结果
    print("\nMajor Observations:")
    print(f"1. {first_feature} shows the strongest positive correlation with the number of rings (age).")
    print(f"   This suggests that as {first_feature.lower()} increases, the age of the abalone tends to increase.")
    print(f"2. {second_feature} shows the second strongest correlation with the number of rings (age).")
    print(f"   This indicates that {second_feature.lower()} is also closely related to the age of the abalone.")

    return first_feature, second_feature


@question_decorator(1, 4)
def create_and_analyze_histograms(data, first_feature, second_feature):
    """
    创建并分析数据集中最相关的两个特征和环数（年龄）的直方图。

    此函数执行以下操作：
    1. 为两个最相关的特征和环数（年龄）创建直方图
    2. 保存直方图
    3. 计算并打印基本统计信息
    4. 分析并打印主要观察结果

    :param data: pandas DataFrame，包含原始数据
    :param first_feature: str，与环数最相关的特征名
    :param second_feature: str，与环数第二相关的特征名
    """
    print("Creating histograms for the two most correlated features and ring-age.")

    # 创建一个包含三个子图的图形
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # 第一个最相关特征的直方图
    sns.histplot(data=data, x=first_feature, bins=30, kde=True, ax=axs[0, 0], edgecolor='black')
    axs[0, 0].set_title(f'Histogram of {first_feature}')
    axs[0, 0].set_xlabel(first_feature)
    axs[0, 0].set_ylabel('Count')

    # 第二个最相关特征的直方图
    sns.histplot(data=data, x=second_feature, bins=30, kde=True, ax=axs[0, 1], edgecolor='black')
    axs[0, 1].set_title(f'Histogram of {second_feature}')
    axs[0, 1].set_xlabel(second_feature)
    axs[0, 1].set_ylabel('Count')

    # 环数（年龄）的直方图
    sns.histplot(data=data, x='Rings', bins=30, kde=True, ax=axs[0, 2], edgecolor='black')
    axs[0, 2].set_title('Histogram of Ring-Age')
    axs[0, 2].set_xlabel('Rings')
    axs[0, 2].set_ylabel('Count')

    # 第一个最相关特征的直方图，平方根变换后
    sns.histplot(data=data, x=np.sqrt(data[first_feature]), bins=30, kde=True, ax=axs[1, 0], edgecolor='black')
    axs[1, 0].set_title(f'Sqrt Transformed Histogram of {first_feature}')
    axs[1, 0].set_xlabel(f'Sqrt of {first_feature}')
    axs[1, 0].set_ylabel('Count')

    # 第二个最相关特征的直方图，平方根变换后
    sns.histplot(data=data, x=np.sqrt(data[second_feature]), bins=30, kde=True, ax=axs[1, 1], edgecolor='black')
    axs[1, 1].set_title(f'Sqrt Transformed Histogram of {second_feature}')
    axs[1, 1].set_xlabel(f'Sqrt of {second_feature}')
    axs[1, 1].set_ylabel('Count')

    # 环数（年龄）的直方图，平方根变换后
    sns.histplot(data=data, x=np.sqrt(data['Rings']), bins=30, kde=True, ax=axs[1, 2], edgecolor='black')
    axs[1, 2].set_title('Sqrt Transformed Histogram of Ring-Age')
    axs[1, 2].set_xlabel('Sqrt of Rings')
    axs[1, 2].set_ylabel('Count')

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('images/feature_histograms_and_standardized.png')
    plt.close()

    print("Histograms saved as 'images/feature_histograms_and_standardized.png'")

    # 计算基本统计信息
    stats = {
        first_feature: data[first_feature].describe(),
        second_feature: data[second_feature].describe(),
        'Rings': data['Rings'].describe()
    }

    # 打印基本统计信息
    print("\nBasic statistics:")
    for feature, stat in stats.items():
        print(f"\n{feature}:")
        print(stat)

    # 打印主要观察结果
    print("\nMajor Observations:")
    print(f"1. Distribution of {first_feature}:")
    print(f"   - Range: {stats[first_feature]['min']:.2f} to {stats[first_feature]['max']:.2f}")
    print(f"   - Mean: {stats[first_feature]['mean']:.2f}, Median: {stats[first_feature]['50%']:.2f}")
    print(f"   - The distribution appears to be right-skewed (positively skewed).")

    print(f"\n2. Distribution of {second_feature}:")
    print(f"   - Range: {stats[second_feature]['min']:.2f} to {stats[second_feature]['max']:.2f}")
    print(f"   - Mean: {stats[second_feature]['mean']:.2f}, Median: {stats[second_feature]['50%']:.2f}")
    print(f"   - The distribution appears to be slightly right-skewed, approximately normal.")

    print("\n3. Distribution of Ring-Age:")
    print(f"   - Range: {stats['Rings']['min']:.0f} to {stats['Rings']['max']:.0f}")
    print(f"   - Mean: {stats['Rings']['mean']:.2f}, Median: {stats['Rings']['50%']:.0f}")
    print(f"   - The distribution appears to be heavily right-skewed, possibly approaching a log-normal distribution.")


@question_decorator(1, 5)
def create_train_test_split(data, experiment_number):
    """
    创建训练集和测试集，并进行数据分割验证。

    此函数执行以下操作：
    1. 使用给定的实验编号作为随机种子，创建60/40的训练/测试集分割
    2. 打印分割后的数据集形状
    3. 将训练集和测试集保存为CSV文件
    4. 验证分割比例
    5. 检查潜在的数据泄露

    :param data: pandas DataFrame，包含要分割的原始数据
    :param experiment_number: int，实验编号，用作随机种子
    :return: tuple，包含训练集和测试集的DataFrame
    """
    # 设置随机种子
    random_seed = experiment_number
    # random_seed = 1
    # random_seed += experiment_number + random.randint(1, 3)
    print(f"Creating 60/40 train/test split by using randon seed: {random_seed}")

    # 创建训练/测试集分割
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=random_seed)

    # 打印分割后的数据集形状
    print(f"Train set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # 保存训练集和测试集
    train_data.to_csv(f'data/train_data_exp_{random_seed}.csv', index=False)
    test_data.to_csv(f'data/test_data_exp_{random_seed}.csv', index=False)

    print(f"Train data saved as 'data/train_data_exp_{random_seed}.csv'")
    print(f"Test data saved as 'data/test_data_exp_{random_seed}.csv'")

    # 验证分割比例
    print("\nVerifying the split:")
    print(f"Percentage of data in train set: {len(train_data) / len(data) * 100:.2f}%")
    print(f"Percentage of data in test set: {len(test_data) / len(data) * 100:.2f}%")

    # 检查潜在的数据泄露
    print("\nChecking for potential data leakage:")
    train_indices = set(train_data.index)
    test_indices = set(test_data.index)
    overlap = train_indices.intersection(test_indices)
    if len(overlap) == 0:
        print("No data leakage detected. Train and test sets have no overlapping indices.")
    else:
        print(f"Warning: {len(overlap)} overlapping indices found between train and test sets.")

    return train_data, test_data


@question_decorator(2, 1)
def linear_regression(train_data, test_data):
    """
    执行线性回归分析并评估模型性能。

    此函数执行以下操作：
    1. 准备训练和测试数据
    2. 训练线性回归模型
    3. 进行预测
    4. 计算并打印评估指标
    5. 可视化实际值与预测值的关系
    6. 分析特征重要性

    :param train_data: pandas DataFrame，包含训练数据
    :param test_data: pandas DataFrame，包含测试数据
    :return: tuple，包含训练好的模型和特征重要性DataFrame
    """
    # 分离特征和目标变量
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # 线性回归
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # 进行预测
    y_test_pred = model_lr.predict(X_test)

    # 调用参数，来做分析
    coefficients_lr = model_lr.coef_
    intercept_lr = model_lr.intercept_

    coefficients_lr = coefficients_lr.flatten() if coefficients_lr.ndim > 1 else coefficients_lr
    print("Feature Coefficients under Linear Regression:")
    max_feature_length = max(len(feature) for feature in X_train.columns)
    for feature, coef in zip(X_train.columns, coefficients_lr):
        print(f'Value of {feature:<{max_feature_length}} : {coef:.4f}')
    print(f'Intercept: {intercept_lr:.4f}\n')

    # 将预测结果四舍五入到最接近的整数，用于分类指标
    y_test_pred_rounded = np.round(y_test_pred).astype(int)

    # 计算评估指标
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

    # 打印评估指标
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # 逻辑回归
    # 将年龄环数转换为二分类问题（例如，大于中位数为1，否则为0）
    # median_rings = y_train.median()
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # 训练逻辑回归模型
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train, y_train_binary)

    # 预测概率
    y_test_prob = model_logistic.predict_proba(X_test)[:, 1]

    # 调用参数，来做分析
    coefficients_logistic = model_logistic.coef_
    intercept_logistic = model_logistic.intercept_[0]

    coefficients_logistic = coefficients_logistic.flatten() if coefficients_logistic.ndim > 1 else coefficients_logistic
    print("Feature Coefficients under Logistic Regression:")
    max_feature_length = max(len(feature) for feature in X_train.columns)
    for feature, coef in zip(X_train.columns, coefficients_logistic):
        print(f'Value of {feature:<{max_feature_length}} : {coef:.4f}')
    print(f'Intercept: {intercept_logistic:.4f}\n')

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # 将概率转换为二元预测
    y_test_pred_logistic = (y_test_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test_binary, y_test_pred_logistic)

    print(f"Logistic Regression AUC: {roc_auc:.4f}")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # 创建图形和网格布局
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 4)

    # 左上子图：实际值vs预测值散点图
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.scatter(y_test, y_test_pred)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual Ring Age")
    ax1.set_ylabel("Predicted Ring Age")
    ax1.set_title("Actual vs Predicted Ring Age")

    # 添加Intercept值到左上图
    ax1.text(0.05, 0.95, f'Intercept: {intercept_lr:.4f}', transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 右上子图：特征重要性横向柱状图
    ax2 = fig.add_subplot(gs[0, 3])
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': coefficients_lr})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    bars = ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Feature Importance")

    # 调整右上子图的字体大小和布局
    ax2.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.title.set_fontsize(10)
    ax2.xaxis.label.set_fontsize(8)

    # 添加数字图例到柱状图
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}',
                 ha='left', va='center', fontsize=7)

    # 左下子图：ROC曲线
    ax3 = fig.add_subplot(gs[1, 0:3])
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax3.legend(loc="lower right")

    # 添加Intercept值到左上图
    ax3.text(0.05, 0.95, f'Intercept: {intercept_logistic:.4f}', transform=ax3.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 右上子图：特征重要性横向柱状图
    ax4 = fig.add_subplot(gs[1, 3])
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': coefficients_logistic})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    bars = ax4.barh(feature_importance['feature'], feature_importance['importance'])
    ax4.set_xlabel("Coefficient Value")
    ax4.set_title("Feature Importance")

    # 调整右上子图的字体大小和布局
    ax4.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    ax4.title.set_fontsize(10)
    ax4.xaxis.label.set_fontsize(8)

    # 添加数字图例到柱状图
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}',
                 ha='left', va='center', fontsize=7)

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('images/regression_and_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


@question_decorator(2, 2)
def compare_regression_models(train_data, test_data):
    """
    比较线性回归和逻辑回归模型性能（有无数据标准化）。

    此函数执行以下操作：
    1. 准备训练和测试数据
    2. 对数据进行标准化处理
    3. 训练线性回归和逻辑回归模型（有标准化和无标准化两种情况）
    4. 进行预测并计算评估指标，包括AUC
    5. 可视化比较结果

    :param train_data: pandas DataFrame，包含训练数据
    :param test_data: pandas DataFrame，包含测试数据
    :return: pandas DataFrame，包含比较结果
    """
    # 分离特征和目标变量
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # 将年龄环数转换为二分类问题（例如，大于中位数为1，否则为0）
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train_binary)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # 初始化模型
    linear_model = LinearRegression()
    logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)

    # 初始化标准化器
    scaler = MinMaxScaler()

    # 用于存储结果的列表
    normalizations = ['Without Normalization', 'With Normalization']
    models = ['Linear Regression', 'Logistic Regression']
    results = []

    # 对有无标准化的数据分别进行建模和评估
    for norm in normalizations:
        if norm == 'With Normalization':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        for model_name in models:
            if model_name == 'Linear Regression':
                model = linear_model
                y_train_target = y_train
                y_test_target = y_test
            else:
                model = logistic_model
                y_train_target = y_train_binary
                y_test_target = y_test_binary

            # 训练模型
            model.fit(X_train_scaled, y_train_target)

            # 进行预测
            if model_name == 'Linear Regression':
                y_test_pred = model.predict(X_test_scaled)
                test_auc = np.nan
                test_rmse = np.sqrt(mean_squared_error(y_test_target, y_test_pred))
                test_r2 = r2_score(y_test_target, y_test_pred)
                test_accuracy = accuracy_score(y_test_target, np.round(y_test_pred))
            else:
                y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_test_pred = model.predict(X_test_scaled)
                test_auc = roc_auc_score(y_test_target, y_test_pred_proba)
                test_rmse = np.nan
                test_r2 = np.nan
                test_accuracy = accuracy_score(y_test_target, np.round(y_test_pred))

            # 存储结果
            results.append({
                'Model': model_name,
                'Normalization': norm,
                'Test RMSE': test_rmse,
                'Test R-squared': test_r2,
                'Test Accuracy': test_accuracy,
                'Test AUC': test_auc
            })

    # 将结果转换为DataFrame以便查看
    results_df = pd.DataFrame(results)

    # 打印结果
    print(results_df.to_string(index=False))

    # 可视化结果
    metrics = ['RMSE', 'R-squared', 'Accuracy', 'AUC']
    fig, axs = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle('Comparison of Model Performance with and without Normalization')

    x = np.arange(len(normalizations))
    width = 0.2

    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            model_results = results_df[results_df['Model'] == model]
            axs[i].bar(x + (j - 0.5) * width, model_results[f'Test {metric}'], width, label=model, alpha=0.7)

        axs[i].set_xlabel('Normalization')
        axs[i].set_ylabel(metric)
        axs[i].set_title(f'{metric} Comparison')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(normalizations)
        if i == 0:  # 只在第一个子图上显示图例
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('images/model_comparison_with_auc.png', bbox_inches='tight')
    plt.close()

    return results_df


@question_decorator(2, 3)
def combined_regression_analysis(train_data, test_data, first_feature, second_feature):
    """
    执行线性回归和逻辑回归分析，并比较结果。

    :param train_data: pandas DataFrame，包含训练数据
    :param test_data: pandas DataFrame，包含测试数据
    :param first_feature: str，第一个选定的特征名
    :param second_feature: str，第二个选定的特征名
    :return: dict，包含两个模型和它们的评估指标
    """
    # 选择特征
    features = [first_feature, second_feature]

    # 分离特征和目标变量
    X_train = train_data[features]
    y_train = train_data['Rings']
    X_test = test_data[features]
    y_test = test_data['Rings']

    # # 标准化特征
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # 不执行标准化
    X_train_scaled = X_train
    X_test_scaled = X_test

    # 线性回归
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_test_pred = linear_model.predict(X_test_scaled)

    # 逻辑回归（将连续的Rings转换为二分类问题）
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train_binary)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
    logistic_model.fit(X_train_scaled, y_train_binary)
    logistic_test_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]
    logistic_test_pred = logistic_model.predict(X_test_scaled)

    # 计算评估指标
    results = {
        'linear': {
            'model': linear_model,
            'test_rmse': np.sqrt(mean_squared_error(y_test, linear_test_pred)),
            'test_r2': r2_score(y_test, linear_test_pred),
            'test_accuracy': accuracy_score(y_test_binary, np.round(linear_test_pred > median_rings).astype(int))
        },
        'logistic': {
            'model': logistic_model,
            'test_accuracy': accuracy_score(y_test_binary, logistic_test_pred),
            'test_auc': roc_auc_score(y_test_binary, logistic_test_pred_proba)
        }
    }

    # 打印结果比较
    print("Regression Analysis Results:")
    print(f"Selected features: {features}")
    print("\nLinear Regression:")
    print(f"Test RMSE: {results['linear']['test_rmse']:.4f}")
    print(f"Test R-squared: {results['linear']['test_r2']:.4f}")
    print(f"Test Accuracy (binary): {results['linear']['test_accuracy']:.4f}")

    print("\nLogistic Regression:")
    print(f"Test Accuracy: {results['logistic']['test_accuracy']:.4f}")
    print(f"Test AUC: {results['logistic']['test_auc']:.4f}")
    cm = confusion_matrix(y_test_binary, logistic_test_pred)
    print(f"Confusion Matrix:\n {cm}")

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test_binary, logistic_test_pred))

    # 可视化比较
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 线性回归散点图
    ax1.scatter(y_test, linear_test_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual Ring Age")
    ax1.set_ylabel("Predicted Ring Age")
    ax1.set_title("Linear Regression: Actual vs Predicted")

    # 逻辑回归混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Logistic Regression: Confusion Matrix')

    plt.tight_layout()
    plt.savefig('images/regression_comparison.png')
    plt.close()

    return results


@question_decorator(2, 4)
def neural_network_regression(train_data, test_data):
    """
    Perform regression analysis using a neural network and optimize hyperparameters through grid search.
    Explicitly output the optimization process.

    :param train_data: pandas DataFrame containing training data
    :param test_data: pandas DataFrame containing test data
    :return: tuple containing the best model and various evaluation metrics
    """
    # Separate features and target variable
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # Standardize input data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter range for grid search
    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (128,), (32, 32), (64, 64)],
        'learning_rate_init': [0.1, 0.01, 0.001],
        'max_iter': [1000],
        'solver': ['sgd']
    }

    # Initialize MLPRegressor
    mlp = MLPRegressor(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    print("Starting grid search...")
    grid_search.fit(X_train_scaled, y_train)
    print("Grid search completed.\n")

    # Explicitly output results for each parameter combination
    results = pd.DataFrame(grid_search.cv_results_)
    for i, row in results.iterrows():
        params = row['params']
        mean_score = -row['mean_test_score']  # Convert to positive MSE
        std_score = row['std_test_score']

        print(f"Parameter combination {i + 1}:")
        print(f"  Hidden layers: {params['hidden_layer_sizes']}")
        print(f"  Learning rate: {params['learning_rate_init']}")
        print(f"  Mean squared error: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print()

    # Get the best model
    best_model = grid_search.best_estimator_

    print("Best parameter combination:")
    print(f"  Hidden layers: {grid_search.best_params_['hidden_layer_sizes']}")
    print(f"  Learning rate: {grid_search.best_params_['learning_rate_init']}")
    print(f"  Best mean squared error: {-grid_search.best_score_:.4f}")
    print()

    # Make predictions using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate evaluation metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print final results
    print("Neural Network Final Results:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")

    # Visualize prediction results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Neural Network: Actual vs Predicted Age")
    plt.savefig('images/nn_actual_vs_predicted.png')
    plt.close()

    return best_model, test_rmse, test_r2


@question_decorator(2, 5)
def compare_models():
    pass


if __name__ == "__main__":
    # P1Q1
    cleaned_data = get_and_clean_data()
    # P1Q2
    corr_matrix = plot_correlation_hot(cleaned_data)
    # P1Q3
    first_feature, second_feature = analyze_and_plot_correlations(cleaned_data, corr_matrix)
    # P1Q4
    create_and_analyze_histograms(cleaned_data, first_feature, second_feature)
    # P1Q5
    train_data, test_data = create_train_test_split(cleaned_data, experiment_number=42)
    # P2Q1
    linear_regression(train_data, test_data)
    # P2Q2
    results_df = compare_regression_models(train_data, test_data)
    # P2Q3
    results = combined_regression_analysis(train_data, test_data, first_feature, second_feature)
    # P2Q4
    nn_model, nn_test_rmse, nn_test_r2 = neural_network_regression(train_data, test_data)
    # P2Q5
    compare_models()
