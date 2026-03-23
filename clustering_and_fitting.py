"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit


def plot_relational_plot(df):
    fig, ax = plt.subplots()

    sns.lineplot(
        x=df['Year'],
        y=df['GDP_Growth_Percent'],
        ax=ax
    )

    ax.set_title("GDP Growth Over Time")
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots()

    df['Country'].value_counts().head(10).plot(kind='bar', ax=ax)

    ax.set_title("Top 10 Countries")
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df[['GDP_Growth_Percent', 'Inflation_Percent']].corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig('statistical_plot.png', bbox_inches='tight')
    plt.close()
    return


def statistical_analysis(df, col: str):
    mean = np.mean(df[col])
    stddev = np.std(df[col])
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):

    print(df.head())
    print(df.describe())
    print(df.corr(numeric_only=True))

    # Handle missing values
    df['Inflation_Percent'] = df['Inflation_Percent'].fillna(
        df['Inflation_Percent'].mean()
    )

    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0:
        skew_text = "right skewed"
    elif moments[2] < 0:
        skew_text = "left skewed"
    else:
        skew_text = "not skewed"

    if moments[3] > 0:
        kurt_text = "leptokurtic"
    elif moments[3] < 0:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"

    print(f'The data was {skew_text} and {kurt_text}.')
    return


def perform_clustering(df, col1, col2):

    def plot_elbow_method():
        fig, ax = plt.subplots()

        ax.plot(range(2, 10), inertias, marker='o')
        ax.set_title("Elbow Method")
        plt.tight_layout()
        plt.savefig('elbow_plot.png')
        plt.close()
        return

    def one_silhouette_inertia():
        best_index = np.argmax(silhouette_scores)
        _score = silhouette_scores[best_index]
        _inertia = inertias[best_index]
        return _score, _inertia

    # Gather data and scale
    data = df[[col1, col2]].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Find best number of clusters
    inertias = []
    silhouette_scores = []

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))

    one_silhouette_inertia()
    plot_elbow_method()

    best_k = range(2, 10)[np.argmax(silhouette_scores)]

    # Get cluster centers
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]
    cenlabels = list(range(best_k))

    return labels, data.values, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots()

    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma')
    ax.scatter(xkmeans, ykmeans, color='black', marker='X')

    ax.set_title("Economic Clusters")
    plt.tight_layout()
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting
    data = df[[col1, col2]].dropna()

    x = data[col1].values
    y = data[col2].values

    # Fit model
    def model(x, a, b):
        return a * x + b

    params, _ = curve_fit(model, x, y)

    # Predict across x
    x_line = np.linspace(min(x), max(x), 100)
    y_line = model(x_line, *params)

    return data.values, x_line, y_line


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots()

    ax.scatter(data[:, 0], data[:, 1])
    ax.plot(x, y, color='red')

    ax.set_title("GDP Growth Trend Over Time")
    plt.tight_layout()
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'GDP_Growth_Percent'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    clustering_results = perform_clustering(
        df,
        'GDP_Growth_Percent',
        'Inflation_Percent'
    )
    plot_clustered_data(*clustering_results)

    fitting_results = perform_fitting(
        df,
        'Year',
        'GDP_Growth_Percent'
    )
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()