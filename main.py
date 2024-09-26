# Author: Ethan FAN
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps

from sklearn.model_selection import train_test_split

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
                part = "Modelling "
            else:
                part = ""
            print(f"{'=' * 10} Answer for {part.lower()} Q{question_number}. {'=' * 10}")
            result = func(*args, **kwargs)
            print(f"{'=' * 10} End of {part.lower()} Q{question_number}. {'=' * 10}\n")
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
    print(f"   - The distribution appears to be [shape of distribution].")

    print(f"\n2. Distribution of {second_feature}:")
    print(f"   - Range: {stats[second_feature]['min']:.2f} to {stats[second_feature]['max']:.2f}")
    print(f"   - Mean: {stats[second_feature]['mean']:.2f}, Median: {stats[second_feature]['50%']:.2f}")
    print(f"   - The distribution appears to be [shape of distribution].")

    print("\n3. Distribution of Ring-Age:")
    print(f"   - Range: {stats['Rings']['min']:.0f} to {stats['Rings']['max']:.0f}")
    print(f"   - Mean: {stats['Rings']['mean']:.2f}, Median: {stats['Rings']['50%']:.0f}")
    print(f"   - The distribution appears to be [shape of distribution].")

    print("\n4. Comparison and Implications:")
    print("   - [Discuss any notable relationships or patterns observed]")
    print("   - [Comment on how these distributions might affect the prediction of ring-age]")


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
