import pandas as pd
import numpy as np

import os 

os.chdir("/home/deckard/code/eda/src")
german_pre_processed = pd.read_csv("../data/raw/german_processed.csv")



german_raw_numeric = pd.read_fwf("../data/raw/original_uci_dataset/german.data-numeric", header=None)

# Get the column names of both dataframes
cols_pre_processed = german_pre_processed.columns
cols_raw = german_raw_numeric.columns

# Find the common columns
common_cols = cols_pre_processed.intersection(cols_raw)

# Print the number of common columns
print(f"Number of common columns: {len(common_cols)}")


german_pre_processed.shape # (1000, 30)
german_raw_numeric.shape # (1000, 25)
 


# Modify this row to sum only numeric features
sum_german_raw_pre_processed = german_pre_processed.apply(sum, axis=0)
sum_german_raw = german_raw_numeric.apply(sum, axis=0)
