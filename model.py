# Author: Ethan FAN; Zengqianyi HU
import os
import matplotlib.gridspec as gridspec
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
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.utils.class_weight import compute_class_weight
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
Dataset Information:
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

Note: Ignore the +1.5 in ring-age and use the raw data

Statistical Summary:
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
    A decorator to clearly mark the beginning and end of each question's answer.
    It prints a formatted header and footer for each function call.

    Args:
    part_number (int): The part number of the assignment (1 or 2).
    question_number (int): The question number within the part.

    Returns:
    function: The wrapped function with added print statements.
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
    Read, clean, and preprocess the raw data from the specified file.

    This function performs the following operations:
    1. Defines column names for the dataset
    2. Reads the CSV data from the specified file
    3. Maps the 'Sex' column's character values to numeric values
    4. Removes any rows with invalid or missing data
    5. Saves the cleaned data to a new CSV file
    6. Prints processing information and a preview of the data

    Returns:
    pandas.DataFrame: A DataFrame containing the cleaned and preprocessed data
    """
    # Define column names for the dataset
    column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight",
                    "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

    # Read the CSV file using the specified column names
    data_format = pd.read_csv(DATA_FILE, names=column_names)

    # Define a mapping of sex categories to numeric values
    gender_map = {'M': 1, 'F': 1, 'I': 0}

    # Map the 'Sex' column values to numeric values
    data_format['Sex'] = data_format['Sex'].map(gender_map)

    # Check for and report any missing or invalid values after replacement
    print(f"Shape before cleaning: {data_format.shape}")
    for name in data_format.columns[1:]:
        print(f"{name} column error number:{data_format[name].isna().sum() + (data_format[name] <= 0).sum()}")

    # Remove rows with invalid Height values or any missing values
    data_format = data_format[(data_format['Height'] > 0) & (~data_format.isna().any(axis=1))]
    print(f"Shape after cleaning: {data_format.shape}")

    # Save the cleaned data to a new CSV file
    data_format.to_csv("data/cleaned_data.csv", index=False)

    # Print processing information and a preview of the cleaned data
    print(f"Data has been cleaned and saved to data/cleaned_data.csv. \n\nThe top 5 data: \n{data_format.head(n=5)}")

    # Return the cleaned DataFrame
    return data_format


@question_decorator(1, 2)
def plot_correlation_hot(data):
    """
    Create and save a correlation heatmap for the input data.

    This function calculates the correlation matrix for the input data,
    visualizes it using a heatmap, and saves the resulting image as a PNG file.
    Correlation values are displayed directly on the heatmap.

    Args:
    data (pandas.DataFrame): The input data to analyze for correlations

    Returns:
    pandas.DataFrame: The calculated correlation matrix
    """
    # Calculate the correlation matrix
    corr_mat = data.corr()

    # Create a new figure with specified size
    plt.figure(figsize=(12, 10))

    # Create a heatmap of the correlation matrix
    plt.imshow(corr_mat, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()  # Add a color bar to the plot
    plt.title("Correlation Matrix")

    # Set x-axis and y-axis labels
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)

    # Add correlation values to each cell in the heatmap
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            value = corr_mat.iloc[i, j]
            color = "white" if value > 0.99 else "black"
            plt.text(j, i, f'{value:.3f}',
                     ha="center", va="center", color=color,
                     fontweight='bold')

    # Adjust layout and save the image
    plt.tight_layout()
    plt.savefig('images/correlation_hot.png')
    plt.close()

    print(f"Correlation hot map has been saved in images/correlation_hot.png.")

    return corr_mat


@question_decorator(1, 3)
def analyze_and_plot_correlations(data, corr_matrix):
    """
    Analyze correlations and create scatter plots for the most correlated features.

    This function performs the following operations:
    1. Identifies the two features most correlated with 'Rings'
    2. Creates scatter plots of these features against 'Rings'
    3. Saves the scatter plots as an image file
    4. Prints analysis results and key observations

    Args:
    data (pandas.DataFrame): The input data for analysis
    corr_matrix (pandas.DataFrame): The correlation matrix of the data

    Returns:
    tuple: Names of the two features most correlated with 'Rings'
    """
    # Calculate absolute correlations with 'Rings', excluding 'Rings' itself
    corr_with_rings = corr_matrix['Rings'].drop('Rings').abs()

    # Identify the two features most correlated with 'Rings'
    top_two_features = corr_with_rings.nlargest(2)
    first_feature = top_two_features.index[0]
    second_feature = top_two_features.index[1]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create scatter plot for the first most correlated feature
    ax1.scatter(data[first_feature], data['Rings'], alpha=0.8)
    ax1.set_xlabel(first_feature)
    ax1.set_ylabel('Rings')
    ax1.set_title(f'Rings vs {first_feature} ')

    # Create scatter plot for the second most correlated feature
    ax2.scatter(data[second_feature], data['Rings'], alpha=0.8)
    ax2.set_xlabel(second_feature)
    ax2.set_ylabel('Rings')
    ax2.set_title(f'Rings vs {second_feature} ')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('images/correlation_scatter_plots.png')
    plt.close()

    # Print results and observations
    print(f"Correlation scatter map has been saved in images/correlation_scatter_plots.png.")
    print(f"Top two correlated features with Rings:")
    print(f"1. {first_feature}: correlation of {corr_with_rings[first_feature]:.3f}")
    print(f"2. {second_feature}: correlation of {corr_with_rings[second_feature]:.3f}")

    # print("\nMajor Observations:")
    # print(f"1. {first_feature} shows the strongest positive correlation with the number of rings (age).")
    # print(f"   This suggests that as {first_feature.lower()} increases, the age of the abalone tends to increase.")
    # print(f"2. {second_feature} shows the second strongest correlation with the number of rings (age).")
    # print(f"   This indicates that {second_feature.lower()} is also closely related to the age of the abalone.")

    return first_feature, second_feature


@question_decorator(1, 4)
def create_and_analyze_histograms(data, first_feature, second_feature):
    """
    Create and analyze histograms for the two most correlated features and ring-age.

    This function performs the following operations:
    1. Creates histograms for the two most correlated features and ring-age
    2. Saves the histograms as an image file
    3. Calculates and prints basic statistical information
    4. Analyzes and prints key observations

    Args:
    data (pandas.DataFrame): The input data for analysis
    first_feature (str): Name of the feature most correlated with ring-age
    second_feature (str): Name of the second most correlated feature with ring-age
    """
    print("Creating histograms for the two most correlated features and ring-age.")

    # Create a figure with six subplots (3 original, 3 sqrt-transformed)
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Create histogram for the first most correlated feature
    sns.histplot(data=data, x=first_feature, bins=30, kde=True, ax=axs[0, 0], edgecolor='black')
    axs[0, 0].set_title(f'Histogram of {first_feature}')
    axs[0, 0].set_xlabel(first_feature)
    axs[0, 0].set_ylabel('Count')

    # Create histogram for the second most correlated feature
    sns.histplot(data=data, x=second_feature, bins=30, kde=True, ax=axs[0, 1], edgecolor='black')
    axs[0, 1].set_title(f'Histogram of {second_feature}')
    axs[0, 1].set_xlabel(second_feature)
    axs[0, 1].set_ylabel('Count')

    # Create histogram for ring-age
    sns.histplot(data=data, x='Rings', bins=30, kde=True, ax=axs[0, 2], edgecolor='black')
    axs[0, 2].set_title('Histogram of Ring-Age')
    axs[0, 2].set_xlabel('Rings')
    axs[0, 2].set_ylabel('Count')

    # Create sqrt-transformed histograms
    sns.histplot(data=data, x=np.sqrt(data[first_feature]), bins=30, kde=True, ax=axs[1, 0], edgecolor='black')
    axs[1, 0].set_title(f'Sqrt Transformed Histogram of {first_feature}')
    axs[1, 0].set_xlabel(f'Sqrt of {first_feature}')
    axs[1, 0].set_ylabel('Count')

    sns.histplot(data=data, x=np.sqrt(data[second_feature]), bins=30, kde=True, ax=axs[1, 1], edgecolor='black')
    axs[1, 1].set_title(f'Sqrt Transformed Histogram of {second_feature}')
    axs[1, 1].set_xlabel(f'Sqrt of {second_feature}')
    axs[1, 1].set_ylabel('Count')

    sns.histplot(data=data, x=np.sqrt(data['Rings']), bins=30, kde=True, ax=axs[1, 2], edgecolor='black')
    axs[1, 2].set_title('Sqrt Transformed Histogram of Ring-Age')
    axs[1, 2].set_xlabel('Sqrt of Rings')
    axs[1, 2].set_ylabel('Count')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('images/feature_histograms_and_standardized.png')
    plt.close()

    print("Histograms saved as 'images/feature_histograms_and_standardized.png'")

    # Calculate basic statistics
    stats = {
        first_feature: data[first_feature].describe(),
        second_feature: data[second_feature].describe(),
        'Rings': data['Rings'].describe()
    }

    # Print basic statistics
    print("\nBasic statistics:")
    for feature, stat in stats.items():
        print(f"\n{feature}:")
        print(stat)

    # Print key observations
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
    Create training and test datasets and validate the data split.

    This function performs the following operations:
    1. Uses the given experiment number as a random seed to create a 60/40 train/test split
    2. Prints the shapes of the resulting datasets
    3. Saves the training and test sets as CSV files
    4. Validates the split ratio
    5. Checks for potential data leakage

    Args:
    data (pandas.DataFrame): The input data to be split
    experiment_number (int): The experiment number, used as the random seed

    Returns:
    tuple: Contains the training and test DataFrames
    """
    # Set the random seed
    random_seed = experiment_number
    print(f"Creating 60/40 train/test split by using random seed: {random_seed}")

    # Create the train/test split
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=random_seed)

    # Print the shapes of the resulting datasets
    print(f"Train set shape: {train_data.shape}")
    print(f"Test set shape: {test_data.shape}")

    # Save the training and test sets
    train_data.to_csv(f'data/train_data_exp_{random_seed}.csv', index=False)
    test_data.to_csv(f'data/test_data_exp_{random_seed}.csv', index=False)

    print(f"Train data saved as 'data/train_data_exp_{random_seed}.csv'")
    print(f"Test data saved as 'data/test_data_exp_{random_seed}.csv'")

    # Validate the split ratio
    print("\nVerifying the split:")
    print(f"Percentage of data in train set: {len(train_data) / len(data) * 100:.2f}%")
    print(f"Percentage of data in test set: {len(test_data) / len(data) * 100:.2f}%")

    # Check for potential data leakage
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
    Perform linear regression analysis and evaluate model performance.

    This function performs the following operations:
    1. Prepares the training and test data
    2. Trains a linear regression model
    3. Makes predictions
    4. Calculates and prints evaluation metrics
    5. Visualizes the relationship between actual and predicted values
    6. Analyzes feature importance

    Args:
    train_data (pandas.DataFrame): The training dataset
    test_data (pandas.DataFrame): The test dataset

    Returns:
    tuple: Contains the trained model and a DataFrame of feature importances
    """
    # Separate features and target variable
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # Train linear regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # Make predictions
    y_test_pred = model_lr.predict(X_test)

    # Extract model parameters for analysis
    coefficients_lr = model_lr.coef_
    intercept_lr = model_lr.intercept_

    coefficients_lr = coefficients_lr.flatten() if coefficients_lr.ndim > 1 else coefficients_lr
    print("Feature Coefficients under Linear Regression:")
    max_feature_length = max(len(feature) for feature in X_train.columns)
    for feature, coef in zip(X_train.columns, coefficients_lr):
        print(f'{feature:<{max_feature_length}} : {coef:.4f}')
    print(f'Intercept: {intercept_lr:.4f}\n')

    # Round predictions to nearest integer for classification metrics
    y_test_pred_rounded = np.round(y_test_pred).astype(int)

    # Calculate evaluation metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)

    # Print evaluation metrics
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Logistic Regression
    # Convert ring age to binary classification problem (e.g., 1 if greater than median, else 0)
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # Train logistic regression model
    model_logistic = LogisticRegression()
    model_logistic.fit(X_train, y_train_binary)

    # Predict probabilities
    y_test_prob = model_logistic.predict_proba(X_test)[:, 1]

    # Extract model parameters for analysis
    coefficients_logistic = model_logistic.coef_
    intercept_logistic = model_logistic.intercept_[0]

    coefficients_logistic = coefficients_logistic.flatten() if coefficients_logistic.ndim > 1 else coefficients_logistic
    print("Feature Coefficients under Logistic Regression:")
    max_feature_length = max(len(feature) for feature in X_train.columns)
    for feature, coef in zip(X_train.columns, coefficients_logistic):
        print(f'{feature:<{max_feature_length}} : {coef:.4f}')
    print(f'Intercept: {intercept_logistic:.4f}\n')

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # Convert probabilities to binary predictions
    y_test_pred_logistic = (y_test_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test_binary, y_test_pred_logistic)

    print(f"Logistic Regression AUC: {roc_auc:.4f}")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # Create figure and grid layout
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 4, wspace=0.3)  # Increase wspace to create more space between columns


    # Top-left subplot: Actual vs Predicted scatter plot
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.scatter(y_test, y_test_pred)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual Ring Age")
    ax1.set_ylabel("Predicted Ring Age")
    ax1.set_title("Actual vs Predicted Ring Age")

    # Add Intercept value to top-left plot
    ax1.text(0.05, 0.95, f'Intercept: {intercept_lr:.4f}', transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Top-right subplot: Feature importance horizontal bar plot
    ax2 = fig.add_subplot(gs[0, 3])
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': coefficients_lr})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    bars = ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Feature Importance")

    # Adjust top-right subplot font sizes and layout
    ax2.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.title.set_fontsize(10)
    ax2.xaxis.label.set_fontsize(8)

    # Add numeric labels to bar plot
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 0.02 * width  # Move the label slightly to the right
        ax2.text(label_position, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=7)

    # Bottom-left subplot: ROC curve
    ax3 = fig.add_subplot(gs[1, 0:3])
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax3.legend(loc="lower right")

    # Add Intercept value to bottom-left plot
    ax3.text(0.05, 0.95, f'Intercept: {intercept_logistic:.4f}', transform=ax3.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Bottom-right subplot: Feature importance horizontal bar plot for logistic regression
    ax4 = fig.add_subplot(gs[1, 3])
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': coefficients_logistic})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    bars = ax4.barh(feature_importance['feature'], feature_importance['importance'])
    ax4.set_xlabel("Coefficient Value")
    ax4.set_title("Feature Importance")

    # Adjust bottom-right subplot font sizes and layout
    ax4.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    ax4.title.set_fontsize(10)
    ax4.xaxis.label.set_fontsize(8)

    # Add numeric labels to bar plot
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 0.02 * width  # Move the label slightly to the right
        ax4.text(label_position, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=7)

    # Adjust layout and save figure
    plt.savefig('images/regression_and_classification_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

    print(f'Regressions have been saved as \'images/regression_and_classification_analysis.png\'')


@question_decorator(2, 2)
def compare_regression_models(train_data, test_data):
    """
    Compare the performance of linear regression and logistic regression models with and without data normalization.

    This function performs the following operations:
    1. Prepares the training and test data
    2. Applies data normalization
    3. Trains linear regression and logistic regression models (with and without normalization)
    4. Makes predictions and calculates evaluation metrics, including AUC
    5. Visualizes the comparison results

    Args:
    train_data (pandas.DataFrame): The training dataset
    test_data (pandas.DataFrame): The test dataset

    Returns:
    pandas.DataFrame: Contains the comparison results
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Separate features and target variable
    X_train = train_data.drop('Rings', axis=1)
    y_train = train_data['Rings']
    X_test = test_data.drop('Rings', axis=1)
    y_test = test_data['Rings']

    # Convert ring age to binary classification problem (e.g., 1 if greater than median, else 0)
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train_binary)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Initialize models
    linear_model = LinearRegression()
    logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)

    # Initialize normalizer
    scaler = MinMaxScaler()

    # Lists to store results
    normalizations = ['Without Normalization', 'With Normalization']
    models = ['Linear Regression', 'Logistic Regression']
    results = []

    # Perform modeling and evaluation for data with and without normalization
    for norm in normalizations:
        if norm == 'With Normalization':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        for model_name in models:
            if model_name == 'Linear Regression':
                model = LinearRegression()
                y_train_target = y_train
                y_test_target = y_test
            else:
                model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
                y_train_target = y_train_binary
                y_test_target = y_test_binary

            # Train the model
            model.fit(X_train_scaled, y_train_target)

            # Make predictions
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
                test_accuracy = accuracy_score(y_test_target, y_test_pred)

            # Store results
            results.append({
                'Model': model_name,
                'Normalization': norm,
                'Test RMSE': test_rmse,
                'Test R-squared': test_r2,
                'Test Accuracy': test_accuracy,
                'Test AUC': test_auc
            })

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    # Print results
    print(results_df.to_string(index=False))

   # Visualize results
    metrics = ['RMSE', 'R-squared', 'Accuracy', 'AUC']
    fig, axs = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle('Comparison of Model Performance with and without Normalization', fontsize=16)

    x = np.arange(len(normalizations))
    width = 0.35  # Adjusted width for better spacing

    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            model_results = results_df[results_df['Model'] == model]
            bar_values = model_results[f'Test {metric}']
            axs[i].bar(x + (j - 0.5) * width, bar_values, width, label=model, alpha=0.7)

            # Add value labels on top of each bar
            for idx, value in enumerate(bar_values):
                if not np.isnan(value):
                    axs[i].text(
                        x[idx] + (j - 0.5) * width,  # X position
                        value + (0.01 * max(bar_values)),  # Y position slightly above the bar
                        f'{value:.4f}' if not np.isnan(value) else 'N/A',  # Text
                        ha='center', va='bottom', fontsize=10  # Alignment and font size
                    )

        axs[i].set_ylabel(metric, fontsize=12)
        axs[i].set_title(f'{metric} Comparison', fontsize=14)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(normalizations, fontsize=12)
        axs[i].tick_params(axis='y', labelsize=10)

    # Adjust legend position above the subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(models), fontsize=12, bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure enough space for the legend above
    plt.savefig('images/model_comparison_with_auc.png', bbox_inches='tight')
    plt.close()

    print(f'\'images/model_comparison_with_auc.png\' has been saved.')


@question_decorator(2, 3)
def combined_regression_analysis(train_data, test_data, first_feature, second_feature):
    """
    Perform linear and logistic regression analysis and compare results.

    This function performs the following operations:
    1. Prepares the data using the two most correlated features
    2. Trains and evaluates both linear and logistic regression models
    3. Calculates and prints various performance metrics
    4. cansVisualizes the results using scatter plots and confusion matrices

    Args:
    train_data (pandas.DataFrame): The training dataset
    test_data (pandas.DataFrame): The test dataset
    first_feature (str): Name of the feature most correlated with ring-age
    second_feature (str): Name of the second most correlated feature with ring-age

    Returns:
    dict: Contains the two models and their evaluation metrics
    """
    # Select features
    features = [first_feature, second_feature]

    # Separate features and target variable
    X_train = train_data[features]
    y_train = train_data['Rings']
    X_test = test_data[features]
    y_test = test_data['Rings']

    # Use raw features without normalization
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_test_pred = linear_model.predict(X_test_scaled)

    # Logistic regression (convert continuous Rings to binary problem)
    median_rings = 7
    y_train_binary = (y_train > median_rings).astype(int)
    y_test_binary = (y_test > median_rings).astype(int)

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train_binary)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
    logistic_model.fit(X_train_scaled, y_train_binary)
    logistic_test_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]
    logistic_test_pred = logistic_model.predict(X_test_scaled)

    # Calculate evaluation metrics
    results = {
        'linear': {
            'model': linear_model,
            'test_rmse': np.sqrt(mean_squared_error(y_test, linear_test_pred)),
            'test_r2': r2_score(y_test, linear_test_pred),
            'test_accuracy': accuracy_score(y_test, np.round(linear_test_pred))
        },
        'logistic': {
            'model': logistic_model,
            'test_accuracy': accuracy_score(y_test_binary, logistic_test_pred),
            'test_auc': roc_auc_score(y_test_binary, logistic_test_pred_proba)
        }
    }

    # Print comparison results
    print("Regression Analysis Results:")
    print(f"Selected features: {features}")
    print("\nLinear Regression:")
    print(f"Test RMSE: {results['linear']['test_rmse']:.4f}")
    print(f"Test R-squared: {results['linear']['test_r2']:.4f}")
    print(f"Test Accuracy: {results['linear']['test_accuracy']:.4f}")

    print("\nLogistic Regression:")
    print(f"Test Accuracy: {results['logistic']['test_accuracy']:.4f}")
    print(f"Test AUC: {results['logistic']['test_auc']:.4f}")
    cm = confusion_matrix(y_test_binary, logistic_test_pred)
    print(f"Confusion Matrix:\n {cm}")

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test_binary, logistic_test_pred))

    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Linear regression scatter plot
    ax1.scatter(y_test, linear_test_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual Ring Age")
    ax1.set_ylabel("Predicted Ring Age")
    ax1.set_title("Linear Regression: Actual vs Predicted")

    # Logistic regression confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, annot_kws={"size": 20})
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Logistic Regression: Confusion Matrix')

    plt.tight_layout()
    plt.savefig('images/regression_comparison.png')
    plt.close()

    print('\'images/regression_comparison.png\' has been saved.')

    return results


@question_decorator(2, 4)
# def neural_network_regression(train_data, test_data):
#     """
#     Perform regression analysis using a neural network and optimize hyperparameters through grid search.
#     Explicitly output the optimization process.

#     This function performs the following operations:
#     1. Prepares and normalizes the data
#     2. Defines a parameter grid for hyperparameter optimization
#     3. Performs grid search with cross-validation
#     4. Trains the best model and evaluates its performance
#     5. Visualizes the actual vs predicted values

#     Args:
#     train_data (pandas.DataFrame): The training dataset
#     test_data (pandas.DataFrame): The test dataset

#     Returns:
#     tuple: Contains the best model and various evaluation metrics
#     """
#     # Separate features and target variable
#     X_train = train_data.drop('Rings', axis=1)
#     y_train = train_data['Rings']
#     X_test = test_data.drop('Rings', axis=1)
#     y_test = test_data['Rings']

#     # Standardize input data
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     param_grid = {
#         'hidden_layer_sizes': [
#             (50,), (100,),
#             (50, 25), (25, 10),
#             (25, 50), (10, 25),
#             (50, 50), (25, 25),
#             (50, 25, 10), (10, 25, 50)
#         ],
#         'solver': ['sgd'],
#         'learning_rate': ['adaptive'],
#         'learning_rate_init': [0.1, 0.01, 0.001],
#         'max_iter': [10000],
#     }

#     # Initialize MLPRegressor
#     mlp = MLPRegressor(random_state=42)

#     # Perform grid search
#     grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

#     print("Starting grid search...")
#     grid_search.fit(X_train_scaled, y_train)

#     # Explicitly output results for each parameter combination
#     grid_results = pd.DataFrame(grid_search.cv_results_)
#     for i, row in grid_results.iterrows():
#         params = row['params']
#         mean_score = -row['mean_test_score']  # Convert to positive MSE
#         std_score = row['std_test_score']

#         print(f"Parameter combination {i + 1}:")
#         print(f"  Hidden layers: {params['hidden_layer_sizes']}")
#         print(f"  Learning rate: {params['learning_rate_init']}")
#         print(f"  Mean squared error: {mean_score:.4f} (+/- {std_score * 2:.4f})")
#         print()

#     print("Grid search completed.")
#     # Get the best model
#     best_model = grid_search.best_estimator_

#     print("- "*20)
#     print("Best parameter combination:")
#     print(f"  Hidden layers: {grid_search.best_params_['hidden_layer_sizes']}")
#     print(f"  Learning rate: {grid_search.best_params_['learning_rate_init']}")
#     print(f"  Best mean squared error: {-grid_search.best_score_:.4f}")
#     print()

#     # Make predictions using the best model
#     y_test_pred = best_model.predict(X_test_scaled)

#     # Calculate evaluation metrics
#     test_mse = mean_squared_error(y_test, y_test_pred)
#     test_rmse = np.sqrt(test_mse)
#     test_mae = mean_absolute_error(y_test, y_test_pred)
#     test_r2 = r2_score(y_test, y_test_pred)

#     # Print final results
#     print("Neural Network Final Results:")
#     print(f"Test MSE: {test_mse:.4f}")
#     print(f"Test RMSE: {test_rmse:.4f}")
#     print(f"Test MAE: {test_mae:.4f}")
#     print(f"Test R-squared: {test_r2:.4f}")

#     # Visualize prediction results
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_test_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel("Actual Age")
#     plt.ylabel("Predicted Age")
#     plt.title("Neural Network: Actual vs Predicted Age")
#     plt.savefig('images/nn_actual_vs_predicted.png')
#     plt.close()

#     print('\'images/nn_actual_vs_predicted.png\' has been saved.')

#     return best_model, test_rmse, test_r2

def neural_network_regression(train_data, test_data):
    """
    Perform regression analysis using a neural network and optimize hyperparameters through grid search.
    Explicitly output the optimization process.

    This function performs the following operations:
    1. Prepares and normalizes the data
    2. Defines a parameter grid for hyperparameter optimization, including activation functions
    3. Performs grid search with cross-validation
    4. Trains the best model and evaluates its performance
    5. Visualizes the actual vs predicted values

    Args:
    train_data (pandas.DataFrame): The training dataset
    test_data (pandas.DataFrame): The test dataset

    Returns:
    tuple: Contains the best model and various evaluation metrics
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

    # Define parameter range for grid search, including activation functions
    param_grid = {
        'hidden_layer_sizes': [
            (50,), (100,),
            (50, 25), (25, 10),
            (25, 50), (10, 25),
            (50, 50), (25, 25),
            (50, 25, 10), (10, 25, 50)
        ],
        'learning_rate_init': [0.1, 0.01, 0.001],
        'max_iter': [10000],
        'solver': ['sgd'],
    }

    # Initialize MLPRegressor
    mlp = MLPRegressor(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    print("Starting grid search...")
    grid_search.fit(X_train_scaled, y_train)

    # Explicitly output results for each parameter combination
    grid_results = pd.DataFrame(grid_search.cv_results_)
    for i, row in grid_results.iterrows():
        params = row['params']
        mean_score = -row['mean_test_score']  # Convert to positive MSE
        std_score = row['std_test_score']

        print(f"Parameter combination {i + 1}:")
        print(f"  Hidden layers: {params['hidden_layer_sizes']}")
        print(f"  Learning rate: {params['learning_rate_init']}")
        print(f"  Mean squared error: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print()

    print("Grid search completed.")
    # Get the best model
    best_model = grid_search.best_estimator_

    print("- " * 20)
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

    # Calculate test accuracy (round predictions to nearest integer and compare)
    y_test_pred_rounded = np.round(y_test_pred)
    test_accuracy = np.mean(y_test_pred_rounded == y_test)

    # Print final results
    print("Neural Network Final Results:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")  # Print test accuracy

    # Visualize prediction results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Neural Network: Actual vs Predicted Age")
    plt.savefig('images/nn_actual_vs_predicted.png')
    plt.close()

    print('\'images/nn_actual_vs_predicted.png\' has been saved.')

    return best_model, test_rmse, test_r2, test_accuracy  # Return accuracy in the tuple


@question_decorator(2, 5)
def compare_models():
    pass


if __name__ == "__main__":
    # Part 1: Data Processing
    # Q1: Clean and preprocess the data
    cleaned_data = get_and_clean_data()
    # Q2: Create and analyze correlation heatmap
    corr_matrix = plot_correlation_hot(cleaned_data)
    # Q3: Analyze and plot correlations for the most correlated features
    first_feature, second_feature = analyze_and_plot_correlations(cleaned_data, corr_matrix)
    # Q4: Create and analyze histograms for key features
    create_and_analyze_histograms(cleaned_data, first_feature, second_feature)
    # Q5: Create train-test split
    train_data, test_data = create_train_test_split(cleaned_data, experiment_number=42)

    # Part 2: Modeling
    # Q1: Perform linear regression analysis
    linear_regression(train_data, test_data)
    # Q2: Compare regression models with and without normalization
    results_df = compare_regression_models(train_data, test_data)
    # Q3: Perform combined regression analysis on selected features
    results = combined_regression_analysis(train_data, test_data, first_feature, second_feature)
    # Q4: Perform neural network regression with hyperparameter optimization
    # nn_model, nn_test_rmse, nn_test_r2 = neural_network_regression(train_data, test_data)
    best_model, test_rmse, test_r2, test_accuracy = neural_network_regression(train_data, test_data)
    # Q5: Compare all models (to be implemented)
    compare_models()
