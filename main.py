import os

import pandas as pd

from utils import make_directory, load_enron_dataset, parse_to_csv
from preprocessing.quality_check import run_quality_check
from preprocessing.cleaning import run_cleaning 
from preprocessing.quality_check_post import run_quality_check_post
from preprocessing.vectorizer import run_vectorizer
from experiments.baseline import run_baseline

if __name__=='__main__':

    # Create directories if not exists
    make_directory('data')
    make_directory('preprocessing')
    make_directory('experiments')
    make_directory('results')
    make_directory('results', 'plots')
    make_directory('results', 'fl_runs')
    make_directory('results\\fl_runs', 'S1_FedAvg_IID_5')
    make_directory('results\\fl_runs', 'S2_FedAvg_nonIID_20')
    make_directory('results\\fl_runs', 'S3_Clipping')
    make_directory('results\\fl_runs', 'S4_Clipping_LowNoise')
    make_directory('results\\fl_runs', 'S5_Clipping_HighNoise')

    print()

    # === Step 1 : Load the dataset ===
    if not os.path.exists("data\\enron_spam.csv"):
        dataframe: pd.DataFrame = load_enron_dataset()
        ## Parse the dataset to a CSV file
        parse_to_csv(dataframe, 'enron_spam.csv', 'data')
    else:
        dataframe: pd.DataFrame = pd.read_csv("data\\enron_spam.csv")

    ## Display some information about the dataframe
    ### Get the columns of the dataset
    enron_columns = dataframe.columns.tolist()
    ### Get the number of rows in the dataset
    num_rows = dataframe.shape[0]
    ### Get the number of missing values in each column
    missing_values = dataframe.isnull().sum()
    ### Get the number of duplicate rows in the dataset
    num_duplicates = dataframe.duplicated().sum()
    ### Get the number of unique values in each column
    unique_values = dataframe.nunique()
    ### Get the number of unique values in each column
    unique_values = dataframe.nunique()

    print(f"Enron columns: {enron_columns}")
    print(f"Number of rows in the dataset: {num_rows}", end="\n\n")
    print("Missing values in each column:")
    print(missing_values, end="\n\n")
    print(f"Number of duplicate rows in the dataset: {num_duplicates}", end="\n\n")
    print(f"Number of unique values in each column:")
    print(unique_values, end="\n\n")

    # === Step 2 : Run quality check before the cleaning ===
    run_quality_check("data\\enron_spam.csv", "results\\qa_report_before_cleaning.json")

    # === Step 3: Cleaning ===
    run_cleaning("data\\enron_spam.csv", "data\\enron_spam_clean.csv")

    # === Step 4: Run quality check after the cleaning ===
    run_quality_check_post("data\\enron_spam_clean.csv", "results\\qa_report_after_cleaning.json", "results\\qa_report_before_cleaning.json", "results\\qa_compare.json")

    # === Step 5: Vectorization TF-IDF ===
    run_vectorizer("data\\enron_spam_clean.csv", "results\\vectorizer.pkl")

    # === Step 6: Run the baseline model script ===
    run_baseline("data\\enron_spam_clean.csv", "results\\vectorizer.pkl")

    # === All done now lets go to FL with flower : Folder FL-simulation ===