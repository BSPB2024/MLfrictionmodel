# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:20:50 2024

@author: bspbiswas
"""

import pandas as pd


# Function to read specific columns from all sheets in an Excel file
def read_excel_columns(file_name, columns_to_extract, avg_sn_limits, air_temp_fahrenheit_limits, speed_mph_limits):
    # df_list = []
    df_combined = pd.DataFrame(columns=columns_to_extract)
    xls = pd.ExcelFile(file_name)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        counter = 0
        for column_to_extract in columns_to_extract:
            if column_to_extract in df.columns:
                counter += 1
        if counter == len(columns_to_extract):
            df = pd.read_excel(file_name, sheet_name=sheet_name, usecols=columns_to_extract)
            for i in range(df.shape[0]):
                filter_row = [0, 0, 0]
                if (df[columns_to_extract[0]][i] >= speed_mph_limits[0]) and (df[columns_to_extract[0]][i] <= speed_mph_limits[1]): filter_row[0] = 1
                if (df[columns_to_extract[1]][i] >= air_temp_fahrenheit_limits[0]) and (df[columns_to_extract[1]][i] <= air_temp_fahrenheit_limits[1]): filter_row[1] = 1
                if (df[columns_to_extract[2]][i] >= avg_sn_limits[0]) and (df[columns_to_extract[2]][i] <= avg_sn_limits[1]): filter_row[2] = 1
                if sum(filter_row) == 3: continue
                else: df.drop([i], inplace=True)
            df_combined = pd.concat([df_combined, df], ignore_index=True);
    return df_combined

# List of files and columns to extract
files = ["I-10.xlsx", "I-25.xlsx", "I-40.xlsx"]
columns_to_extract = ["Speed(MPH)", "Air Temp (Fahrenheit)", "Avg_SN"]
avg_sn_limits = [30, 60]
air_temp_fahrenheit_limits = [10, 140]
speed_mph_limits = [10, 55]


df_combined = pd.DataFrame(columns=columns_to_extract)
for filename in files:    
    all_data = read_excel_columns(filename, columns_to_extract, avg_sn_limits, air_temp_fahrenheit_limits, speed_mph_limits)
    df_combined = pd.concat([df_combined, all_data], ignore_index=True);

all_data = all_data.sample(frac=1, random_state = 1).reset_index(drop=True)
# all_data.to_csv('all_data.csv', index=False, columns=columns_to_extract)



