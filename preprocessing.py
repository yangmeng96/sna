import pandas as pd
import numpy as np
import os

# read raw sas file
stroke_data = pd.read_sas("data/main_dataset_final_3.sas7bdat")
print("raw data loaded")

# drop a few columns:
    # stroke_event_id: duplication of row_names
    # code_short: duplication of stroke_subtype
    # marital_status: all unknown
    # patient_regional_location: all NE
stroke_data.drop(columns = [
    "stroke_event_id", "code_short", "marital_status", "patient_regional_location"], inplace = True)

# bytes to utf8
columns_to_decode = ["patient_id", "stroke_subtype", "sex", "race"]
stroke_data[columns_to_decode] = stroke_data[
    columns_to_decode].applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# change column names: remove "icd10_"
stroke_data = stroke_data.rename(columns=lambda x: x.replace('icd10_', '') if x.startswith('icd10_') else x)

# split data into illness vs. demographic
stroke_data_illness = stroke_data.iloc[:, :-8]
stroke_data_demo = stroke_data.iloc[:, -8:]

# binary data
    # NA to 0: no illness
    # negative values to 0: in the future yes, but not at that moment, so we do not know
    # positive values or 0 to 1: had the illness during the past month (including the current day)
stroke_data_binary = stroke_data_illness.applymap(lambda x: 0 if pd.isna(x) or x < 0 else 1)
stroke_data_binary = pd.concat([stroke_data_binary, stroke_data_demo], axis=1)
print("binary data created")

# continuous data
    # NA or negative values to 0: same as binary data
    # positive values or 0 to exp(-itself): today to 1, long time ago to approx 0
stroke_data_cont = stroke_data_illness.applymap(lambda x: 0 if pd.isna(x) or x < 0 else np.exp(-x))
stroke_data_cont = pd.concat([stroke_data_cont, stroke_data_demo], axis=1)
print("continuous data created")

# save csv files
file_path = 'data/'
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
stroke_data_binary.to_csv(file_path + "stroke_data_binary.csv", index=False)
print("binary data saved as: " + file_path + "stroke_data_binary.csv")
stroke_data_cont.to_csv(file_path + "stroke_data_cont.csv", index=False)
print("continuous data saved as: " + file_path + "stroke_data_binary.csv")