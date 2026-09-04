import sys
import subprocess
import os

def install_requirements():
    required_packages = ['pandas', 'matplotlib', 'seaborn', 'requests']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package '{package}' not found. Installing")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies installed.\n")

install_requirements()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
import requests

# FETCHING DATASET
def fetch_base_data():
    raw_data_path = 'dataA.csv'
    
    #Github mirror of the Kaggle E-commerce Fraud dataset
    url = "https://raw.githubusercontent.com/dphi-official/Imbalanced_classes/master/fraud_data.csv"
    
    if os.path.exists(raw_data_path):
        print("Base data already exists locally. Loading")
        df = pd.read_csv(raw_data_path, low_memory=False)
    else:
        print(f"Fetching dataset from {url}...")
        df = pd.read_csv(url, low_memory=False)
        df.to_csv(raw_data_path, index=False)
        print(f"Saved base dataset to {raw_data_path}")
        
    return df

# PREPROCESSING & VISUALIZATION
def run_eda(df):
    print("EDA")
    
    # Info
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print("\nData Schema:")
    print(df.dtypes)
    
    # Class Distribution (Fraud vs Legit)
    fraud_counts = df['isFraud'].value_counts(normalize=True) * 100
    print("\nFraud Prevalence:")
    print(f"Legitimate (0): {fraud_counts.get(0, 0):.2f}%")
    print(f"Fraudulent (1): {fraud_counts.get(1, 0):.2f}%")
    
    # Fix the Fragmentation Warning
    df = df.copy()
    
    # Convert timestamps
    df['TransactionHour'] = (df['TransactionDT'] / 3600) % 24
    
    # Create Visualizations
    print("\nVisualizations")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Transaction Value Distribution
    plt.figure(figsize=(10, 6))
    plot_df = df[df['TransactionAmt'] < 1000]
    sns.histplot(data=plot_df, x='TransactionAmt', hue='isFraud', bins=50, kde=True, 
                 palette={0: 'blue', 1: 'red'}, alpha=0.5, stat='density', common_norm=False)
    plt.title('Transaction Value Distribution (Legit vs Fraud)')
    plt.xlabel('Purchase Value ($)')
    plt.ylabel('Density (Normalized)')
    plt.savefig('01_value_distribution.png')
    plt.close()
    
    # Plot 2: Transaction Activity by Relative Hour (Fixed labeling to reflect TransactionDT properly)
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=df, x='TransactionHour', hue='isFraud', bins=24, 
                 palette={0: 'blue', 1: 'red'}, alpha=0.5, stat='density', common_norm=False)
    plt.title('Transaction Activity by Relative Hour')
    plt.xlabel('Hour of Day (Relative derived from TransactionDT)')
    plt.ylabel('Density (Normalized)')
    plt.savefig('02_time_to_purchase.png')
    plt.close()
    
    print("Visualizations saved to root folder.")

if __name__ == "__main__":
    base_df = fetch_base_data()
    run_eda(base_df)
    
    print("\nEDA completed.")