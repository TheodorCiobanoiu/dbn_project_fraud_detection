import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Customize plot appearance
def customize_plot(ax, title, xlabel, ylabel, fontname='Arial'):
    ax.set_title(title, fontsize=14, fontweight='bold', color='black', fontname=fontname)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color='black', fontname=fontname)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='black', fontname=fontname)
    ax.tick_params(colors='black', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)


# Set the overall aesthetic style of the plots
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_context("talk")

# Primary color for the plots
primary_color = "#632ce4"


# SCRIPTS FOR EXPLORATORY DATA ANALYSIS
def perform_eda_creditcard(chosen_data):
    # Basic Information
    print("Basic Information:")
    print(chosen_data.info())

    # Summary Statistics
    print("\nSummary Statistics:")
    print(chosen_data.describe())

    # Distribution of the 'Class' variable
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Class', data=chosen_data, color=primary_color)
    customize_plot(ax, 'Distribution of Class Variable', 'Class', 'Frequency (Log Scale)')
    plt.yscale('log')
    plt.savefig('eda/creditcard/class_distribution.png', dpi=600)
    plt.show()

    # Distribution of the 'Amount' variable
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(chosen_data[chosen_data['Amount'] <= 10000]['Amount'], bins=50, kde=False, color=primary_color)
    customize_plot(ax, 'Transaction Amount Distribution (<= 2500)', 'Amount', 'Frequency')
    plt.yscale('log')
    plt.savefig('eda/creditcard/amount_distribution.png', dpi=600)
    plt.show()

    # Distribution of the 'Time' variable
    chosen_data['Time_hr'] = chosen_data['Time'] / 3600  # Convert time to hours
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(chosen_data['Time_hr'], bins=50, kde=False, color=primary_color)
    customize_plot(ax, 'Transaction Time Distribution', 'Time (Hours)', 'Frequency')
    plt.savefig('eda/creditcard/time_distribution.png', dpi=600)
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(chosen_data.corr(), cmap='coolwarm', annot=False)
    customize_plot(ax, 'Correlation Matrix Heatmap', '', '')
    plt.savefig('eda/creditcard/matrix_heatmap.png', dpi=600)
    plt.show()


# Function for Exploratory Data Analysis on the 'Base.csv' dataset
def perform_eda_base(chosen_data):
    numeric_data = chosen_data.select_dtypes(include=[np.number])
    # Basic Information
    print("Basic Information:")
    print(chosen_data.info())

    # Summary Statistics
    print("\nSummary Statistics:")
    print(chosen_data.describe())

    # Distribution of the 'fraud_bool' variable
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='fraud_bool', data=chosen_data, color=primary_color)
    customize_plot(ax, 'Distribution of Fraud Variable', 'Fraud (0/1)', 'Frequency (Log Scale)')
    plt.yscale('log')
    plt.savefig('eda/baf/fraud_distribution.png', dpi=600)
    plt.show()

    # Distribution of income
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(chosen_data['income'], bins=50, kde=False, color=primary_color)
    customize_plot(ax, 'Income Distribution', 'Income', 'Frequency')
    plt.savefig('eda/baf/income_distribution.png', dpi=600)
    plt.show()

    # Distribution of another customer_age
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(chosen_data['customer_age'], bins=50, kde=False, color=primary_color)
    customize_plot(ax, 'Customer Age Distribution', 'Age', 'Frequency')
    plt.savefig('eda/baf/customer_age_distribution.png', dpi=600)
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=False)
    customize_plot(ax, 'Correlation Matrix Heatmap', '', '')
    plt.savefig('eda/baf/matrix_heatmap.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    dataset_choice = input("Choose the dataset ('creditcard' or 'baf'): ")
    file_path = 'creditcard.csv' if dataset_choice == 'creditcard' else 'Base.csv'
    data = pd.read_csv(file_path)
    if dataset_choice == 'creditcard':
        perform_eda_creditcard(data)
    else:
        perform_eda_base(data)