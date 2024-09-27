# Author: Ethan FAN
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import wraps
import numpy as np
import sklearn.linear_model
from PIL.Image import module
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

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
def get_and_clean_data() -> pd.DataFrame:
    column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight",
                    "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    data_format = pd.read_csv(DATA_FILE, names=column_names)
    gender_map = {'M': 0, 'F': 1, 'I': 2}
    data_format['Sex'] = data_format['Sex'].map(gender_map)
    data_format.to_csv("data/cleaned_data.csv", index=False)

    print(f"Data has been cleaned and saved to data/cleaned_data.csv. \nThe top 5 data: \n{data_format.head(n=5)}")

    return data_format


@question_decorator(1, 2)
def plot_correlation_hot(data):
    corr_mat = data.corr()

    plt.figure(figsize=(10, 10))
    plt.imshow(corr_mat, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)
    plt.tight_layout()
    plt.savefig('images/correlation_hot.png')
    plt.close()

    print(f"Correlation hot map has been saved in images/correlation_hot.png.")

    return corr_mat


@question_decorator(1, 3)
def analyze_and_plot_correlations(data, corr_matrix):
    corr_with_rings = corr_matrix['Rings'].drop('Rings').abs()
    top_two_features = corr_with_rings.nlargest(2)
    first_feature = top_two_features.index[0]
    second_feature = top_two_features.index[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(data[first_feature], data['Rings'])
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Rings')
    ax1.set_title(f'Rings vs {first_feature} ')

    ax2.scatter(data[second_feature], data['Rings'])
    ax2.set_xlabel(second_feature)
    ax2.set_ylabel('Rings')
    ax2.set_title(f'Rings vs {second_feature} ')

    plt.tight_layout()
    plt.savefig('images/correlation_scatter_plots.png')
    plt.close()

    print(f"Correlation scatter map has been saved in images/correlation_scatter_plots.png.")
    print(f"Top two correlated features with Rings:")
    print(f"1. {first_feature}: correlation of {corr_with_rings[first_feature]:.3f}")
    print(f"2. {second_feature}: correlation of {corr_with_rings[second_feature]:.3f}")

    print("\nMajor Observations:")
    print(f"1. {first_feature} shows the strongest positive correlation with the number of rings (age).")
    print(f"   This suggests that as {first_feature.lower()} increases, the age of the abalone tends to increase.")
    print(f"2. {second_feature} shows the strongest negative correlation with the number of rings (age).")
    print(f"   This indicates that as {second_feature.lower()} increases, the age of the abalone tends to decrease.")

    return first_feature, second_feature


@question_decorator(1, 4)
def create_and_analyze_histograms(data, first_feature, second_feature):
    print("Creating histograms for the two most correlated features and ring-age...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Histogram for the first most correlated feature
    ax1.hist(data[first_feature], bins=30, edgecolor='black')
    ax1.set_title(f'Histogram of {first_feature}')
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Frequency')

    # Histogram for the second most correlated feature
    ax2.hist(data[second_feature], bins=30, edgecolor='black')
    ax2.set_title(f'Histogram of {second_feature}')
    ax2.set_xlabel(second_feature)
    ax2.set_ylabel('Frequency')

    # Histogram for ring-age
    ax3.hist(data['Rings'], bins=30, edgecolor='black')
    ax3.set_title('Histogram of Ring-Age')
    ax3.set_xlabel('Rings')
    ax3.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('images/feature_histograms.png')
    plt.close()

    print("Histograms saved as 'images/feature_histograms.png'")

    # Calculate some basic statistics
    stats = {
        first_feature: data[first_feature].describe(),
        second_feature: data[second_feature].describe(),
        'Rings': data['Rings'].describe()
    }

    print("\nBasic statistics:")
    for feature, stat in stats.items():
        print(f"\n{feature}:")
        print(stat)

    print("\nMajor Observations:")
    print(f"1. Distribution of {first_feature}:")
    print(f"   - Range: {stats[first_feature]['min']:.2f} to {stats[first_feature]['max']:.2f}")
    print(f"   - Mean: {stats[first_feature]['mean']:.2f}, Median: {stats[first_feature]['50%']:.2f}")
    print(f"   - The distribution appears to be Right-skewed (positively skewed) distribution.")

    print(f"\n2. Distribution of {second_feature}:")
    print(f"   - Range: {stats[second_feature]['min']:.2f} to {stats[second_feature]['max']:.2f}")
    print(f"   - Mean: {stats[second_feature]['mean']:.2f}, Median: {stats[second_feature]['50%']:.2f}")
    print(f"   - The distribution appears to be Slightly right-skewed, approximately normal distribution.")

    print("\n3. Distribution of Ring-Age:")
    print(f"   - Range: {stats['Rings']['min']:.0f} to {stats['Rings']['max']:.0f}")
    print(f"   - Mean: {stats['Rings']['mean']:.2f}, Median: {stats['Rings']['50%']:.0f}")
    print(
        f"   - The distribution appears to be Heavily right-skewed distribution, possibly approaching a log-normal distribution.")


@question_decorator(1, 5)
def create_train_test_split(data, experiment_number):
    print(f"Creating 60/40 train/test split for experiment number {experiment_number}")

    # Set the random seed based on the experiment number
    random_seed = experiment_number

    # Create the train/test split
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=random_seed)

    print(f"Train set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # Save the train and test datasets
    train_data.to_csv(f'data/train_data_exp_{experiment_number}.csv', index=False)
    test_data.to_csv(f'data/test_data_exp_{experiment_number}.csv', index=False)

    print(f"Train data saved as 'data/train_data_exp_{experiment_number}.csv'")
    print(f"Test data saved as 'data/test_data_exp_{experiment_number}.csv'")

    # Verify the split
    print("\nVerifying the split:")
    print(f"Percentage of data in train set: {len(train_data) / len(data) * 100:.2f}%")
    print(f"Percentage of data in test set: {len(test_data) / len(data) * 100:.2f}%")

    # Check for data leakage
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
def linear_regression(train_data_in, test_data_in):
    # Load the data
    # train_data = pd.read_csv('data/train_data_exp_6040.csv')
    # test_data = pd.read_csv('data/test_data_exp_6040.csv')
    train_data = train_data_in
    test_data = test_data_in

    # Separate features and target
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model (Linear Regression)
    model = LinearRegression()
    # model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Round predictions to nearest integer
    y_train_pred_rounded = np.round(y_train_pred).astype(int)
    y_test_pred_rounded = np.round(y_test_pred).astype(int)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred_rounded)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

    # Calculate Spearman's rank correlation
    train_spearman = stats.spearmanr(y_train, y_train_pred)[0]
    test_spearman = stats.spearmanr(y_test, y_test_pred)[0]

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Spearman's Rank Correlation: {train_spearman:.4f}")
    print(f"Test Spearman's Rank Correlation: {test_spearman:.4f}")

    # # Compute ROC curve and AUC for each class
    # unique_rings = np.unique(np.concatenate([y_train, y_test]))
    # n_classes = len(unique_rings)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    #
    # for i, ring_value in enumerate(unique_rings):
    #     y_test_binary = (y_test == ring_value).astype(int)
    #     y_score = -np.abs(y_test_pred - ring_value)  # Negative absolute difference as the score
    #
    #     fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_score)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and AUC
    # y_test_binary = np.eye(n_classes)[np.searchsorted(unique_rings, y_test)]
    # y_score = -np.abs(y_test_pred.reshape(-1, 1) - unique_rings)  # Broadcasting
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # Plot ROC curves
    # plt.figure(figsize=(10, 8))
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(unique_rings[i], roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.savefig('images/roc_curve.png')
    # plt.close()
    #
    # # Print average AUC
    # print(f"Average AUC: {np.mean(list(roc_auc.values())):.4f}")
    #
    # # Visualize predictions
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_test_pred, alpha=0.5)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    # plt.xlabel("Actual Ring Age")
    # plt.ylabel("Predicted Ring Age")
    # plt.title("Actual vs Predicted Ring Age")
    # plt.tight_layout()
    # plt.savefig('images/actual_vs_predicted.png')
    # plt.close()
    #
    # # Feature importance
    # feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': abs(model.coef_)})
    # feature_importance = feature_importance.sort_values('importance', ascending=False)
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(feature_importance['feature'], feature_importance['importance'])
    # plt.xticks(rotation=90)
    # plt.xlabel("Features")
    # plt.ylabel("Absolute Coefficient Value")
    # plt.title("Feature Importance")
    # plt.tight_layout()
    # plt.savefig('images/feature_importance.png')
    # plt.close()
    #
    # # Confusion Matrix
    # cm = confusion_matrix(y_test, y_test_pred_rounded)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.savefig('images/confusion_matrix.png')
    # plt.close()
    #
    # print(
    #     "Plots have been saved as 'images/actual_vs_predicted.png', 'images/feature_importance.png', 'images/confusion_matrix.png', and 'images/roc_curve.png'")

    # Calculate and print the unique values in train and test sets
    train_unique = np.unique(y_train)
    test_unique = np.unique(y_test)
    print(f"Unique values in train set: {train_unique}")
    print(f"Unique values in test set: {test_unique}")
    print(f"Values in test set but not in train set: {np.setdiff1d(test_unique, train_unique)}")


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
    linear_regression(train_data, test_data)
    # P2Q2
