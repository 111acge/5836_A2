# Author: Ethan FAN
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, roc_curve, auc)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

DATA_FILE = "data/abalone.data"

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

    # 将清理后的数据保存为新的CSV文件
    data_format.to_csv("data/cleaned_data.csv", index=False)

    # 打印处理信息和数据预览
    print(f"Data has been cleaned and saved to data/cleaned_data.csv. \nThe top 5 data: \n{data_format.head(n=5)}")

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
    plt.imshow(corr_mat, interpolation='nearest', cmap='hot')
    plt.colorbar()  # 添加颜色条
    plt.title("Correlation Matrix")

    # 设置x轴和y轴的标签
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)

    # 在热图的每个单元格中添加相关性值
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            value = corr_mat.iloc[i, j]
            color = "white" if value < 0.2 else "black"
            plt.text(j, i, f'{value:.2f}',
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
    ax1.scatter(data[first_feature], data['Rings'])
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Rings')
    ax1.set_title(f'Rings vs {first_feature} ')

    # 绘制第二个特征与 'Rings' 的散点图
    ax2.scatter(data[second_feature], data['Rings'])
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
    print("Creating histograms for the two most correlated features and ring-age...")

    # 创建一个包含三个子图的图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 第一个最相关特征的直方图
    ax1.hist(data[first_feature], bins=30, edgecolor='black')
    ax1.set_title(f'Histogram of {first_feature}')
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Frequency')

    # 第二个最相关特征的直方图
    ax2.hist(data[second_feature], bins=30, edgecolor='black')
    ax2.set_title(f'Histogram of {second_feature}')
    ax2.set_xlabel(second_feature)
    ax2.set_ylabel('Frequency')

    # 环数（年龄）的直方图
    ax3.hist(data['Rings'], bins=30, edgecolor='black')
    ax3.set_title('Histogram of Ring-Age')
    ax3.set_xlabel('Rings')
    ax3.set_ylabel('Frequency')

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig('images/feature_histograms.png')
    plt.close()

    print("Histograms saved as 'images/feature_histograms.png'")

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
    print(f"Creating 60/40 train/test split for experiment number {experiment_number}")

    # 设置随机种子，基于实验编号
    random_seed = experiment_number

    # 创建训练/测试集分割
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=random_seed)

    # 打印分割后的数据集形状
    print(f"Train set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # 保存训练集和测试集
    train_data.to_csv(f'data/train_data_exp_{experiment_number}.csv', index=False)
    test_data.to_csv(f'data/test_data_exp_{experiment_number}.csv', index=False)

    print(f"Train data saved as 'data/train_data_exp_{experiment_number}.csv'")
    print(f"Test data saved as 'data/test_data_exp_{experiment_number}.csv'")

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

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 将预测结果四舍五入到最接近的整数，用于分类指标
    y_train_pred_rounded = np.round(y_train_pred).astype(int)
    y_test_pred_rounded = np.round(y_test_pred).astype(int)

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

    # 打印评估指标
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Ring Age")
    plt.ylabel("Predicted Ring Age")
    plt.title("Actual vs Predicted Ring Age")
    plt.tight_layout()
    plt.savefig('images/actual_vs_predicted.png')
    plt.close()

    # 分析特征重要性
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': abs(model.coef_)})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Absolute Coefficient Value")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()

    # 注释掉的ROC曲线和AUC分数部分
    """
    关于ROC和AUC的说明：
    ROC（接收者操作特征）曲线和AUC（曲线下面积）通常用于二分类问题。
    在回归问题中，特别是预测连续值如环数（年龄）时，这些指标不太适用。
    这可能是为什么运行时会出错的原因。
    对于回归问题，我们通常使用RMSE, R-squared等指标来评估模型性能。
    """
    ### ###################################################################
    # 这个ROC和AUC我完全没搞懂是什么东西，内容是GPT给的，但是GPT给的这个部分解除注释运行会报错
    #
    # # ROC curve and AUC score
    # fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    # roc_auc = auc(fpr, tpr)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.savefig('images/roc_curve.png')
    # plt.close()
    #
    # print(f"AUC Score: {roc_auc:.4f}")
    ### ###################################################################

    return model, feature_importance


@question_decorator(2, 2)
def linear_regression_selected_features(train_data, test_data, first_feature, second_feature):
    """
    使用选定的特征执行线性回归分析并评估模型性能。

    此函数执行以下操作：
    1. 选择指定的特征
    2. 准备训练和测试数据
    3. 训练线性回归模型
    4. 进行预测
    5. 计算并打印评估指标
    6. 可视化实际值与预测值的关系

    :param train_data: pandas DataFrame，包含训练数据
    :param test_data: pandas DataFrame，包含测试数据
    :param first_feature: str，第一个选定的特征名
    :param second_feature: str，第二个选定的特征名
    :return: tuple，包含训练好的模型和各种评估指标
    """
    # 选择特征
    features = [first_feature, second_feature]

    # 分离特征和目标变量
    X_train = train_data[features]
    y_train = train_data['Rings']
    X_test = test_data[features]
    y_test = test_data['Rings']

    # 训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 将预测结果四舍五入到最接近的整数，用于分类指标
    y_train_pred_rounded = np.round(y_train_pred).astype(int)
    y_test_pred_rounded = np.round(y_test_pred).astype(int)

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

    # 打印选定的特征和评估指标
    print(f"Selected features: {features}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Ring Age")
    plt.ylabel("Predicted Ring Age")
    plt.title("Actual vs Predicted Ring Age (Selected Features)")
    plt.savefig('images/actual_vs_predicted_selected_features.png')
    plt.close()

    return model, train_rmse, test_rmse, train_r2, test_r2, train_accuracy, test_accuracy


@question_decorator(2, 3)
def compare_regression_models(train_data, test_data):
    """
    比较有无数据标准化的线性回归模型性能。

    此函数执行以下操作：
    1. 准备训练和测试数据
    2. 对数据进行标准化处理
    3. 训练线性回归模型（有标准化和无标准化两种情况）
    4. 进行预测并计算评估指标
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

    # 初始化模型
    model = LinearRegression()

    # 初始化标准化器
    scaler = StandardScaler()

    # 用于存储结果的列表
    normalizations = ['Without Normalization', 'With Normalization']
    results = []

    # 对有无标准化的数据分别进行建模和评估
    for norm in normalizations:
        if norm == 'With Normalization':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 进行预测
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # 将预测结果四舍五入到最接近的整数
        y_train_pred_rounded = np.round(y_train_pred).astype(int)
        y_test_pred_rounded = np.round(y_test_pred).astype(int)

        # 计算评估指标
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
        test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

        # 存储结果
        results.append({
            'Normalization': norm,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train R-squared': train_r2,
            'Test R-squared': test_r2,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy
        })

    # 将结果转换为DataFrame以便查看
    results_df = pd.DataFrame(results)

    # 打印结果
    print(results_df.to_string(index=False))

    # 可视化结果
    metrics = ['RMSE', 'R-squared', 'Accuracy']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(normalizations))
        width = 0.35

        # 绘制训练集和测试集的柱状图
        plt.bar(x - width / 2, results_df[f'Train {metric}'], width, label='Train', color='blue', alpha=0.7)
        plt.bar(x + width / 2, results_df[f'Test {metric}'], width, label='Test', color='red', alpha=0.7)

        # 设置图表标题和标签
        plt.xlabel('Normalization')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} With and Without Normalization')
        plt.xticks(x, normalizations)
        plt.legend()
        plt.tight_layout()

        # 保存图表
        plt.savefig(f'images/comparison_{metric.lower()}.png')
        plt.close()

    return results_df


@question_decorator(2, 4)
def neural_network_regression(train_data, test_data):
    """
    使用神经网络进行回归分析，并通过网格搜索优化超参数。

    此函数执行以下操作：
    1. 准备训练和测试数据
    2. 对数据进行标准化处理
    3. 定义网格搜索的参数范围
    4. 使用GridSearchCV找到最佳参数
    5. 使用最佳参数训练模型
    6. 进行预测并计算评估指标
    7. 可视化预测结果

    :param train_data: pandas DataFrame，包含训练数据
    :param test_data: pandas DataFrame，包含测试数据
    :return: tuple，包含最佳模型和各种评估指标
    """
    # 分离特征和目标变量
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # 标准化输入数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义网格搜索的参数范围
    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (10, 10), (20, 10)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [1000]
    }

    # 初始化MLPRegressor
    mlp = MLPRegressor(random_state=6040)

    # 执行网格搜索
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 使用最佳模型进行预测
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # 打印结果
    print("Neural Network Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Ring Age")
    plt.ylabel("Predicted Ring Age")
    plt.title("Neural Network: Actual vs Predicted Ring Age")
    plt.savefig('images/nn_actual_vs_predicted.png')
    plt.close()

    return best_model, train_rmse, test_rmse, train_r2, test_r2


@question_decorator(2, 5)
def compare_models():
    """
    比较线性回归和神经网络模型的性能，并提供分析讨论。

    此函数执行以下操作：
    1. 从之前的结果中提取线性回归和神经网络的性能指标
    2. 打印两个模型的性能比较表格
    3. 提供详细的结果讨论，包括性能比较、过拟合分析和改进建议

    注：此函数假设 results_df, nn_train_rmse, nn_test_rmse, nn_train_r2, nn_test_r2 是全局变量
    """
    # 提取线性回归模型的结果（使用标准化后的数据）
    linear_results = {
        'train_rmse': results_df.loc[results_df['Normalization'] == 'With Normalization', 'Train RMSE'].values[0],
        'test_rmse': results_df.loc[results_df['Normalization'] == 'With Normalization', 'Test RMSE'].values[0],
        'train_r2': results_df.loc[results_df['Normalization'] == 'With Normalization', 'Train R-squared'].values[0],
        'test_r2': results_df.loc[results_df['Normalization'] == 'With Normalization', 'Test R-squared'].values[0]
    }

    # 提取神经网络模型的结果
    nn_results = {
        'train_rmse': nn_train_rmse,
        'test_rmse': nn_test_rmse,
        'train_r2': nn_train_r2,
        'test_r2': nn_test_r2
    }

    # 打印模型比较表格
    print("\nModel Comparison:")
    print("Linear Regression vs Neural Network")
    print(f"{'Metric':<15}{'Linear Regression':<20}{'Neural Network':<20}")
    print("-" * 55)
    print(f"{'Train RMSE':<15}{linear_results['train_rmse']:<20.4f}{nn_results['train_rmse']:<20.4f}")
    print(f"{'Test RMSE':<15}{linear_results['test_rmse']:<20.4f}{nn_results['test_rmse']:<20.4f}")
    print(f"{'Train R2':<15}{linear_results['train_r2']:<20.4f}{nn_results['train_r2']:<20.4f}")
    print(f"{'Test R2':<15}{linear_results['test_r2']:<20.4f}{nn_results['test_r2']:<20.4f}")

    # 提供详细的结果讨论
    print("\nDiscussion:")

    # 1. 性能比较
    print("1. Performance Comparison:")
    if nn_results['test_rmse'] < linear_results['test_rmse']:
        print("   - The Neural Network model outperforms the Linear Regression model in terms of RMSE.")
    else:
        print("   - The Linear Regression model performs better than or similarly to the Neural Network model.")

    # 2. 过拟合分析
    print("2. Overfitting:")
    lr_overfit = linear_results['train_rmse'] - linear_results['test_rmse']
    nn_overfit = nn_results['train_rmse'] - nn_results['test_rmse']
    if abs(lr_overfit) < abs(nn_overfit):
        print("   - The Linear Regression model seems to generalize better, showing less overfitting.")
    else:
        print("   - The Neural Network model might be overfitting more than the Linear Regression model.")

    # 3. 改进建议
    print("3. Further Improvements:")
    print("   - Feature engineering: Create new features or transform existing ones.")
    print("   - Regularization: Apply L1 or L2 regularization to prevent overfitting.")
    print("   - Ensemble methods: Combine multiple models for better predictions.")
    print("   - Cross-validation: Use k-fold cross-validation for more robust model evaluation.")
    print("   - Hyperparameter tuning: Further fine-tune the neural network architecture and parameters.")
    print("   - Data augmentation: Generate synthetic data to increase the dataset size.")
    print("   - Handle outliers: Identify and treat outliers that might affect model performance.")


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
    train_data, test_data = create_train_test_split(cleaned_data, experiment_number=6040)
    # P2Q1
    model, feature_importance = linear_regression(train_data, test_data)
    # P2Q2
    model_selected, train_rmse_selected, test_rmse_selected, train_r2_selected, test_r2_selected, train_accuracy_selected, test_accuracy_selected = linear_regression_selected_features(
        train_data, test_data, first_feature, second_feature)
    # P2Q3
    results_df = compare_regression_models(train_data, test_data)
    # P2Q4
    nn_model, nn_train_rmse, nn_test_rmse, nn_train_r2, nn_test_r2 = neural_network_regression(train_data, test_data)
    # P2Q5
    compare_models()
