import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from cycler import cycler
from pandas import DataFrame

# import datetime
# from datetime import import timedelta

#################### Get the path of the working file ####################
path = os.getcwd()

# Hypothesis on the prepayement rate to be choose
Bound_rate = 0.05
Prepayement_rate_10_3 = 0.29

# Paths
path_data = path + "\\Performance_Monitoring_Data_ALMT10.xlsx"
Excel_output = path + "\\Performance_Monitoring_Tool_ALMT10.xlsx"
#path_PY_Test = path + "\\Output\\Test1.xlsx"
#Path_Fx = path + "\\Source\\FX_rates.xlsx"

# Read the excel
data_base = pd.read_excel(path_data, keep_default_na=False).fillna('')

# Pivot the data to
Data = pd.pivot_table(data_base, values='Balance Sheet Amt in EUR Total', index=['RMPM Code', 'Activity', 'End Date'], columns=['COB Date', 'Product'], fill_value=0).reset_index()
Data_Expected = data_base[(data_base['Activity'] == "Project Finance") | (data_base['Activity'] == "Shipping Finance") | (data_base['Activity'] == "Aviation Finance") | (data_base['Activity'] == "Export Finance")]
Data_Unexpected = data_base[(data_base['Activity'] != "Project Finance") & (data_base['Activity'] != "Shipping Finance") & (data_base['Activity'] != "Aviation Finance") & (data_base['Activity'] != "Export Finance")]

################### Expected 10.1

# Create tables for Drawn and Non Drawn Data
Data_Drawn_Exp = Data_Expected[Data_Expected['Product'] == "DRAWN"]
Data_NDrawn_Exp = Data_Expected[Data_Expected['Product'] == "UNDRAWN COMMITTED"]

# Get the end date from the Non Drawn product and affect it to each clients
Data_NDrawn_Exp_Date = pd.DataFrame(Data_NDrawn_Exp, columns = ['RMPM Code', 'End Date'])
Data_NDrawn_Exp_Date.groupby(['RMPM Code'], sort=False)['End Date'].max()
Data_NDrawn_Exp_Date=Data_NDrawn_Exp_Date.drop_duplicates(subset='RMPM Code', keep = 'first', inplace = False)

# Fromating of the 2 table Drawn and Undrawn for Expected client by Observation date and client
Data_Drawn_Exp = Data_Drawn_Exp.groupby(by = ['COB Date', 'RMPM Code'], as_index=False).sum()
Data_NDrawn_Exp = Data_NDrawn_Exp.groupby(by = ['COB Date', 'RMPM Code'], as_index=False).sum()

# Add the End data of each client to the main table
Data_All_Exp = pd.merge(Data_Drawn_Exp, Data_NDrawn_Exp, right_on= ['COB Date', 'RMPM Code'] , left_on=['COB Date', 'RMPM Code'], how='outer').replace(np.nan, 0)
Data_All_Exp.rename(columns = {'Balance Sheet Amt in EUR Total_x':'Balance_D', 'Balance Sheet Amt in EUR Total_y':'Balance_U'}, inplace = True)

# Calculate the utilisation rate for each client
Data_All_Exp['Utilisation'] = Data_All_Exp['Balance_D']/(Data_All_Exp['Balance_D']+Data_All_Exp['Balance_U'])

# Formating
Data_All_Exp=Data_All_Exp.sort_values(by=['RMPM Code', 'COB Date'], ascending=True)
Data_All_Exp_Pivot = pd.pivot_table(Data_All_Exp, values='Utilisation', index=['RMPM Code', ], columns=['COB Date']).reset_index()

# Copy of the main table "Data_All_Exp_Pivot" and use those copies for the next steps
Data_All_Exp_Formula1 = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula2 = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula3 = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula_Up = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula_Bellow = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula_Diff_Up = Data_All_Exp_Pivot.copy()
Data_All_Exp_Formula_Diff_Bellow = Data_All_Exp_Pivot.copy()

Data_All_Exp_Pivot = pd.merge(Data_All_Exp_Pivot, Data_NDrawn_Exp_Date, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')

# Dynamical Count of the number of the Period in the Extract
Date = Data_All_Exp_Pivot.columns.tolist()
df = DataFrame(Date)                                    # Transformation of the headlines in the main tab to a list ( this list will then contain End data, RMPM Code)
df_date = df.iloc[1:-1,:]                               # The first and the last row of the lists are not a date, and will then be excluded to have only
df_date.reset_index(inplace=True)                       # Reset the index to have the numbers affected to each period ( for example the initial
df_date.columns = ["Period_Number", 'COB Date']
df_date["Period_Number"] = df_date["Period_Number"]-1
df_date['COB Date']=pd.to_datetime(df_date['COB Date'])

# Replace the Zeros with NA
Data_All_Exp_Pivot = Data_All_Exp_Pivot.replace(0, np.nan)
Data_All_Exp_Pivot_Cal = Data_All_Exp_Pivot.copy()

Test = (Date[2]-Date[1]) / (Data_All_Exp_Pivot['End Date'] - Date[1])

# For loop to calculate the expected bound and excess relative against the realised values
for i in range(0, len(Data_All_Exp_Pivot), 1):
    for j in range(len(Data_All_Exp_Pivot.columns)-3, 0, -1):
        Data_All_Exp_Formula1.iloc[:, j+1] = (Date[j+1] -Date[1] ) / (Data_All_Exp_Pivot.iloc[:, -1] - Date[1])
        Data_All_Exp_Formula3.iloc[:, j+1] = Data_All_Exp_Formula1.iloc[:, j] + (1 - Data_All_Exp_Formula1.iloc[:, j+1])*Data_All_Exp_Formula1.iloc[:, j+1]
        Data_All_Exp_Formula2.iloc[:, j+1] = Data_All_Exp_Formula2.iloc[:, j+1].apply(lambda x: (Bound_rate/4)*j if x < 0.2 else 0)     # add the the initial utilisation to 1 minus the the initial utilisation multiplied by the time weight to get the decreasing part
        Data_All_Exp_Formula_Up.iloc[:, j+1] = Data_All_Exp_Formula3.iloc[:, j+1]+Data_All_Exp_Formula2.iloc[:, j+1]                    # Construct the upper bound
        Data_All_Exp_Formula_Bellow.iloc[:, j+1] = Data_All_Exp_Formula3.iloc[:, j+1]-Data_All_Exp_Formula2.iloc[:, j+1]               # Construct the lower bound
        Data_All_Exp_Formula_Diff_Up.iloc[:, j+1] = Data_All_Exp_Pivot_Cal.iloc[:, j+1]-Data_All_Exp_Formula_Up.iloc[:, j+1]          # Calculation of the exces if Realised Utilisation > Model Up Bound
        Data_All_Exp_Formula_Diff_Bellow.iloc[:, j+1] = Data_All_Exp_Formula_Bellow.iloc[:, j+1]-Data_All_Exp_Pivot_Cal.iloc[:, j+1]  # Calculation of the exces if Realised Utilisation < Model Down Bound

# Get rid of the entries in Data_All_Exp_Formula_Diff_Up and Data_All_Exp_Formula_Diff_Bellow where the values are negative
Data_All_Exp_Formula_Diff_Up = Data_All_Exp_Formula_Diff_Up.replace(np.nan, 1000000)
Data_All_Exp_Formula_Diff_Up = Data_All_Exp_Formula_Diff_Up.where(Data_All_Exp_Formula_Diff_Up > 0, 0)
Data_All_Exp_Formula_Diff_Up = Data_All_Exp_Formula_Diff_Up.replace(1000000, np.nan)

Data_All_Exp_Formula_Diff_Bellow = Data_All_Exp_Formula_Diff_Bellow.replace(np.nan, 1000000)
Data_All_Exp_Formula_Diff_Bellow = Data_All_Exp_Formula_Diff_Bellow.where(Data_All_Exp_Formula_Diff_Bellow > 0, 0)
Data_All_Exp_Formula_Diff_Bellow = Data_All_Exp_Formula_Diff_Bellow.replace(1000000, np.nan)

# Table with the differences 10.1 Expected
Data_All_Exp_Formula_Diff = Data_All_Exp_Formula_Diff_Up.copy()
Data_All_Exp_Formula_Diff.iloc[:, 1:] = Data_All_Exp_Formula_Diff_Bellow.iloc[:, 1:] + Data_All_Exp_Formula_Diff_Up.iloc[:, 1:]
Data_All_Exp_Formula_Diff.iloc[:, 1] = Data_All_Exp_Formula_Diff.iloc[:, 1] - Data_All_Exp_Formula_Diff_Up.iloc[:, 1]

# Based on the asumption :
# - Filter out values with intial Utilisation egual 100%
# - Filter out values with intial Utilisation egual 0%
# - Filter out value clients with no ligne of credit at ignitiation ( Utilisated amount is zero & Unitilisated amount is zero)

Data_All_Exp_Formula_Diff = Data_All_Exp_Formula_Diff[(Data_All_Exp_Formula_Diff.iloc[:, 1] != 1 ) & (Data_All_Exp_Formula_Diff.iloc[:, 1] != 0 )] # & (data_base['LB_SITE_COMPTABLE'] == "BNP PARIBAS NEW YORK BRANCH" )]

####
Data_All_Exp_Formula_Diff2 = Data_All_Exp_Formula_Diff.copy()
Data_All_Exp_Formula_Diff2 = Data_All_Exp_Formula_Diff.melt(['RMPM Code'], var_name='COB Date', value_name='Utilisation')
Data_All_Exp_Formula_Diff2 = pd.merge(Data_All_Exp_Formula_Diff2, Data_NDrawn_Exp_Date, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')
Data_All_Exp_Formula_Diff2['COB Date Shift'] = Data_All_Exp_Formula_Diff2['COB Date'].shift(1)
Data_All_Exp_Formula_Diff2['Initial Date'] = Date[1]
Data_All_Exp_Formula_Diff2['COB Date Shift'] = Data_All_Exp_Formula_Diff2[['RMPM Code', 'COB Date', 'Utilisation', 'End Date', 'COB Date Shift', 'Initial Date']].apply(lambda x: x['COB Date'] if x['COB Date Shift']==x['Initial Date'] else x['COB Date Shift'], axis = 1)
Data_All_Exp_Formula_Diff2['Utilisation Adjusted'] = Data_All_Exp_Formula_Diff2[['RMPM Code', 'COB Date', 'Utilisation', 'End Date', 'COB Date Shift', 'Initial Date']].apply(lambda x: np.nan if (x['End Date'] < x['COB Date Shift'] and x['COB Date'] != x['Initial Date']) else x['Utilisation'], axis = 1)

Data_All_Exp_Formula_Diff2 = Data_All_Exp_Formula_Diff2.copy()
Data_All_Exp_Formula_Diff2=Data_All_Exp_Formula_Diff2.drop(['Initial Date'])

Data_All_Exp_Formula_Diff3 = pd.pivot_table(Data_All_Exp_Formula_Diff2, values='Utilisation Adjusted', index=['RMPM Code', ], columns=['COB Date']).reset_index()
Data_All_Exp_Formula_Diff5 = Data_All_Exp_Formula_Diff3.copy()

Data_All_Exp_Formula_Diff3 = Data_All_Exp_Formula_Diff3[Data_All_Exp_Formula_Diff3.iloc[:, 1].notna()]
Data_All_Exp_Formula_Diff3['Average'] = Data_All_Exp_Formula_Diff3.iloc[:, 2:].mean(axis=1)

# Result Expected 10.1
df_Result_Exp = pd.DataFrame([['Average Excess - Expected 10.1', Data_All_Exp_Formula_Diff3[Data_All_Exp_Formula_Diff3['Average'] != 0]['Average'].mean(axis=0)]], columns = ['Model ', ' Result'])

################### Unexpected 10.1

Data_Drawn_Unexp = Data_Unexpected[Data_Unexpected['Product'] == "DRAWN"]
Data_NDrawn_Unexp = Data_Unexpected[Data_Unexpected['Product'] == "UNDRAWN COMMITTED"]

Data_NDrawn_Unexp_Date = pd.DataFrame(Data_NDrawn_Unexp, columns = ['RMPM Code', 'End Date'])
Data_NDrawn_Unexp_Date.groupby(['RMPM Code'], sort=False)['End Date'].max()
Data_NDrawn_Unexp_Date=Data_NDrawn_Unexp_Date.drop_duplicates(subset='RMPM Code', keep = 'first', inplace = False)

Data_Drawn_Unexp = Data_Drawn_Unexp.groupby(by = ['COB Date', 'RMPM Code'], as_index=False).sum()
Data_NDrawn_Unexp = Data_NDrawn_Unexp.groupby(by = ['COB Date', 'RMPM Code'], as_index=False).sum()

Data_All_Unexp = pd.merge(Data_Drawn_Unexp, Data_NDrawn_Unexp, right_on= ['COB Date', 'RMPM Code'] , left_on=['COB Date', 'RMPM Code'], how='outer').replace(np.nan, 0)
Data_All_Unexp.rename(columns = {'Balance Sheet Amt in EUR Total_x':'Balance_D', 'Balance Sheet Amt in EUR Total_y':'Balance_U'}, inplace = True)
Data_All_Unexp['Utilisation'] = Data_All_Unexp['Balance_D']/(Data_All_Unexp['Balance_D']+Data_All_Unexp['Balance_U'])
Data_All_Unexp=Data_All_Unexp.sort_values(by=['RMPM Code', 'COB Date'], ascending=True)
Data_All_Unexp_Pivot = pd.pivot_table(Data_All_Unexp, values='Utilisation', index=['RMPM Code', ], columns=['COB Date']).reset_index()

# Copy of the main table "Data_All_Exp_Pivot" and use those copies for the next steps
Data_All_Unexp_Formula1 = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula2 = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula3 = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula_Up = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula_Bellow = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula_Diff_Up = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Formula_Diff_Bellow = Data_All_Unexp_Pivot.copy()
Data_All_Unexp_Pivot_Cal = Data_All_Unexp_Pivot.copy()

Data_All_Unexp_Pivot = pd.merge(Data_All_Unexp_Pivot, Data_NDrawn_Unexp_Date, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')

# Dynamical Count of the number of the Period in the Extract
Date = Data_All_Unexp_Pivot.columns.tolist()
df = DataFrame(Date)
df_date = df.iloc[1:-1,:]

Test = (Date[2]-Date[1]) / (Data_All_Unexp_Pivot['End Date'] - Date[1])

# Replace the Zeros with NA
Data_All_Unexp_Pivot = Data_All_Unexp_Pivot.replace(0, np.nan)
Data_All_Unexp_Pivot_Cal = Data_All_Unexp_Pivot.copy()

# For loop to calculate the expected bound and excess relative against the realised values
for i in range(0, len(Data_All_Unexp_Pivot), 1):
    for j in range(len(Data_All_Unexp_Pivot.columns)-3, 0, -1):
        Data_All_Unexp_Formula1.iloc[:, j+1] = Data_All_Unexp_Pivot.copy()
        Data_All_Unexp_Formula2.iloc[:, j+1] = Data_All_Unexp_Formula1.iloc[:, j+1].apply(lambda x: (Bound_rate/4)*j if x < 0.2 else 0)
        Data_All_Unexp_Formula3.iloc[:, j+1] = Data_All_Unexp_Formula1.iloc[:, j] + (1 - Data_All_Unexp_Formula1.iloc[:, j+1])*Data_All_Unexp_Formula1.iloc[:, j+1]
        Data_All_Unexp_Formula_Up.iloc[:, j+1] = Data_All_Unexp_Formula3.iloc[:, j+1]+Data_All_Unexp_Formula2.iloc[:, j+1]
        Data_All_Unexp_Formula_Bellow.iloc[:, j+1] = Data_All_Unexp_Formula3.iloc[:, j+1]-Data_All_Unexp_Formula2.iloc[:, j+1]
        Data_All_Unexp_Formula_Diff_Up.iloc[:, j+1] = Data_All_Unexp_Pivot_Cal.iloc[:, j+1]-Data_All_Unexp_Formula_Up.iloc[:, j+1]
        Data_All_Unexp_Formula_Diff_Bellow.iloc[:, j+1] = Data_All_Unexp_Formula_Bellow.iloc[:, j+1]-Data_All_Unexp_Pivot_Cal.iloc[:, j+1]

# Get rid of the entries in Data_All_Exp_Formula_Diff_Up and Data_All_Exp_Formula_Diff_Bellow where the values are negative
Data_All_Unexp_Formula_Diff_Up = Data_All_Unexp_Formula_Diff_Up.replace(np.nan, 1000000)
Data_All_Unexp_Formula_Diff_Up = Data_All_Unexp_Formula_Diff_Up.where(Data_All_Unexp_Formula_Diff_Up > 0, 0)
Data_All_Unexp_Formula_Diff_Up = Data_All_Unexp_Formula_Diff_Up.replace(1000000, np.nan)

Data_All_Unexp_Formula_Diff_Bellow = Data_All_Unexp_Formula_Diff_Bellow.replace(np.nan, 1000000)
Data_All_Unexp_Formula_Diff_Bellow = Data_All_Unexp_Formula_Diff_Bellow.where(Data_All_Unexp_Formula_Diff_Bellow > 0, 0)
Data_All_Unexp_Formula_Diff_Bellow = Data_All_Unexp_Formula_Diff_Bellow.replace(1000000, np.nan)

# Table with the differences 10.1 Expected
Data_All_Unexp_Formula_Diff = Data_All_Unexp_Formula_Diff_Up.copy()
Data_All_Unexp_Formula_Diff.iloc[:, 1:] = Data_All_Unexp_Formula_Diff_Bellow.iloc[:, 1:] + Data_All_Unexp_Formula_Diff_Up.iloc[:, 1:]
Data_All_Unexp_Formula_Diff.iloc[:, 1] = Data_All_Unexp_Formula_Diff.iloc[:, 1] - Data_All_Unexp_Formula_Diff_Up.iloc[:, 1]

# Based on the asumption :
# - Filter out values with intial Utilisation egual 100%
# - Filter out values with intial Utilisation egual 0%
# - Filter out value clients with no ligne of credit at ignitiation ( Utilisated amount is zero & Unitilisated amount is zero)

Data_All_Unexp_Formula_Diff = Data_All_Unexp_Formula_Diff[(Data_All_Unexp_Formula_Diff.iloc[:, 1] != 0 )]
Data_All_Unexp_Formula_Diff = Data_All_Unexp_Formula_Diff[Data_All_Unexp_Formula_Diff.iloc[:, 1].notna()]
Data_All_Unexp_Formula_Diff['Average'] = Data_All_Unexp_Formula_Diff.iloc[:, 2:].mean(axis=1)

# Result Unexpected 10.1
df_Result_Unexp = pd.DataFrame([['Average Excess - Unexpected 10.1', Data_All_Unexp_Formula_Diff[Data_All_Unexp_Formula_Diff['Average'] != 0]['Average'].mean(axis=0)]], columns = ['Model ', ' Result'])

################### Model : 10.2

Data_All_Exp_10_2 = pd.merge(Data_All_Exp, Data_All_Exp_Formula_Diff3, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')

# Drop Na in the column Average to keep only relevant client
Data_All_Exp_10_2 = Data_All_Exp_10_2.dropna(subset=['Average'])

Data_All_Exp_10_2=Data_All_Exp_10_2.sort_values(by=['RMPM Code', 'COB Date'], ascending=True)
Data_All_Exp_10_2_Pivot = pd.pivot_table(Data_All_Exp_10_2, values='Utilisation', index=['RMPM Code', ], columns=['COB Date']).reset_index()

Data_All_Exp_10_2 = Data_All_Exp_10_2.sort_values(['RMPM Code', 'COB Date'], inplace=True)
Data_All_Exp_10_2['diff_Exp'] = Data_All_Exp_10_2.groupby(['RMPM Code'])['Balance_D'].diff()
Data_All_Exp_10_2['diff_Exp_U'] = Data_All_Exp_10_2.groupby(['RMPM Code'])['Balance_U'].diff()

Data_All_10_2 = Data_All_Exp_10_2[['COB Date', 'RMPM Code', 'Balance_D', 'Balance_U', 'diff_Exp', 'diff_Exp_U']]
Data_All_10_2['diff_Exp_U_Value'] = Data_All_10_2['diff_Exp_U']

Data_All_10_2["diff_Exp_test"] = Data_All_10_2["diff_Exp"].apply(lambda x: 1 if x < 0 else 0)
Data_All_10_2["diff_Exp_U_test"] = Data_All_10_2["diff_Exp_U"].apply(lambda x: 1 if x > 0 else 0)

Data_All_10_2['Realized Drawing'] = Data_All_10_2['diff_Exp_U_test'] * Data_All_10_2['diff_Exp_test'] * Data_All_10_2['diff_Exp']
Data_All_10_2['Realized Drawing'] = Data_All_10_2['Realized Drawing'].replace(np.nan, 0)
Data_All_10_2['Drawing by client'] = Data_All_10_2.groupby('RMPM Code')['Realized Drawing'].transform('sum')
Data_All_10_2['Weight'] = Data_All_10_2['Realized Drawing'] / Data_All_10_2['Drawing by client']

# Add the end date for the undrawn part of the loan
Data_All_10_2 = pd.merge(Data_All_10_2, Data_NDrawn_Exp_Date, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')
# Drop Na in the column COB Date to keep only rows from Data1_All_10_2
Data_All_10_2 = Data_All_10_2.dropna(subset=['COB Date'])

# Attach Initial date data
Data_All_10_2_First_Initial_date_data= Data_All_10_2[Data_All_10_2['COB Date'] == Data_All_10_2['COB Date'].min()]
Data_All_10_2_First_Initial_date_data = Data_All_10_2_First_Initial_date_data[['RMPM Code', 'Balance_U']]
Data_All_10_2_First_Initial_date_data.rename(columns = {'Balance_U':'Balance_U_Initial_Date'}, inplace = True)
Data_All_10_2 = pd.merge(Data_All_10_2, Data_All_10_2_First_Initial_date_data, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')
Data_All_10_2['Initial Date'] = Data_All_10_2.groupby(['RMPM Code'])['COB Date'].transform('min')

Data_All_10_2['Draw_Model'] = ( Data_All_10_2.groupby(['RMPM Code'])['COB Date'].diff().apply(lambda x: x.days)/ ( Data_All_10_2['End Date'] - Data_All_10_2['Initial Date']).apply(lambda x: x.days)) * Data_All_10_2['Balance_U_Initial_Date']

Data_All_10_2['Error'] = (Data_All_10_2['Realized Drawing'] - Data_All_10_2['Draw_Model']).abs() / Data_All_10_2['Realized Drawing'].abs()

Data_All_10_2['Weihted_Error'] = Data_All_10_2['Error'] * Data_All_10_2['Weight']

### Hypothesis , if Abs Realised Draw is <100 k than just put 0 in the weighted error, because the draw is way to marginal to calculate the marginal error
Data_All_10_2['Weihted_Error'] = Data_All_10_2[['Weihted_Error', 'Realized Drawing']].apply(lambda x: 0 if (np.abs(x['Realized Drawing']) < 100000 and (pd.notna(x['Weihted_Error']) == True)) else x['Weihted_Error'], axis = 1)

# Weighted Average error
Data_All_10_2['Weighted_Average_error'] = Data_All_10_2.groupby('RMPM Code')['Weihted_Error'].transform('sum')

# Formating table as the original excel \
Data_All_10_2_Final = Data_All_10_2[['RMPM Code', 'Balance_D', 'Balance_U', 'Realized Drawing', 'Draw_Model', 'Weight', 'End Date', 'Error', 'Weihted_Error', 'Weighted_Average_error']]

# Formating before geting the last result
Result_10_2_bis = pd.DataFrame(Data_All_10_2, columns = ['RMPM Code', 'Weighted_Average_error' ])
Result_10_2_bis.drop_duplicates(keep = "first", inplace = True)

# Final result 10.2
Result_10_2 = pd.DataFrame([['Weighted Average error 10.2', Result_10_2_bis[Result_10_2_bis['Weighted_Average_error'] != 0]['Weighted_Average_error'].mean(axis=0)]], columns = ['Model ', ' Result'])

############################# Model : 10.3

Data_All_Unexp_Formula_Diff['Test_Initial_Date_Value'] = Data_All_Unexp_Formula_Diff.iloc[:, 1]

# Trick to have creat another column with the same values as in the initial date month data, to use afterwards when merging to get the End date and keep only clients already available before the merge
Data_All_10_3 = pd.merge(Data_All_Unexp, Data_All_Unexp_Formula_Diff, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')

# Drop Na in the column Average to keep only relevant client
Data_All_10_3 = Data_All_10_3.dropna(subset=['Test_Initial_Date_Value'])
Data_All_10_3=Data_All_10_3.sort_values(by=['RMPM Code', 'COB Date'], ascending=True)

# Formating
Data_All_10_3 = pd.DataFrame(Data_All_10_3, columns = ['COB Date', 'RMPM Code', 'Balance_D'])

# Add a column with the first initial month date in all the column to use it when doing the calculation ( Quarter X - Quarter 0)
Data_All_10_3_First_Initial_date_data= Data_All_10_3[Data_All_10_3['COB Date'] == Data_All_10_3['COB Date'].min()]
Data_All_10_3_First_Initial_date_data = Data_All_10_3_First_Initial_date_data[['RMPM Code', 'Balance_D']]
Data_All_10_3_First_Initial_date_data.rename(columns = {'Balance_D':'Balance_D_Initial_Date'}, inplace = True)
Data_All_10_3 = pd.merge(Data_All_10_3, Data_All_10_3_First_Initial_date_data, right_on= ['RMPM Code',] , left_on=['RMPM Code'], how='outer')
Data_All_10_3 = Data_All_10_3.sort_values(by=['RMPM Code', 'COB Date'], ascending=True)

# Formula for prepayement by quarter
Prepayement_rate_10_3 = 1 - ( 1 - Prepayement_rate_10_3 ) ** 0.25

# Formula used for 10.3 to calculate Model Draw
Data_All_10_3['Draw_Model'] = Data_All_10_3[['RMPM Code', 'Balance_D', 'Period_Number', 'Balance_D_Initial_Date']].apply(lambda x: 0 if x['Balance_D'] == 0 else x['Balance_D_Initial_Date']*(1 - Prepayement_rate_10_3)**x['Period_Number'], axis = 1)

Data_All_10_3_Final = Data_All_10_3[['COB Date', 'RMPM Code', 'Balance_D', 'Draw_Model']]

# Pivot the data before export to excel
Data_All_10_3_Pivot = pd.pivot_table(Data_All_10_3, values=['Balance_D', 'Draw_Model'], index=['RMPM Code'], columns=['COB Date'], fill_value=0).reset_index()
Data_All_10_3_Grouped = Data_All_10_3.groupby(by = ['COB Date'], as_index=False).sum()
Data_All_10_3_Grouped =Data_All_10_3_Grouped[['COB Date', 'Balance_D', 'Draw_Model']]

# Plot
ax = plt.gca()
fig, ax = plt.subplots(figsize=(15,7))
Data_All_10_3_Grouped.plot(kind='line', x='COB Date', y='Balance_D', ax=ax, color='tab:blue', linestyle='--')
Data_All_10_3_Grouped.plot(kind='line', x='COB Date', y='Draw_Model', ax=ax, color='tab:red', )
fig.suptitle("Economic schedule of the used part vs the realized amount ", fontsize=18, ha="center")
plt.savefig('Unexpected - 10.3.png')
plt.show()

# Concatenate all the results
Result_Final = pd.concat([Result, Result_Unexp, Result_10_2])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(Excel_output, engine='xlsxwriter')

# Write each dataframe to a different worksheet
Data_All_Exp_Formula_Diff3.to_excel( writer ,header=True,index = 0,sheet_name = 'ExpectedFinal - 10.1')
Result.to_excel( writer ,header=True,index = 0,sheet_name = 'ExpectedFinal - 10.1 Final')
Data_All_Unexp_Formula_Diff.to_excel( writer ,header=True,index = 0,sheet_name = 'UnexpectedFinal - 10.1')
Result_Unexp.to_excel( writer ,header=True,index = 0,sheet_name = 'UnexpectedFinal - 10.1 Final')
Data_All_10_2_Final.to_excel( writer ,header=True,index = 0,sheet_name = 'Expected - 10.2')
Result_10_2.to_excel( writer ,header=True,index = 0,sheet_name = 'Expected - 10.2 Final')
Result_Final.to_excel( writer ,header=True,index = 0,sheet_name = 'Recap Results')
Data_All_10_3_Final.to_excel( writer ,header=True,index = 0,sheet_name = 'Expected - 10.3_All')
Data_All_10_3_Grouped.to_excel( writer ,header=True,index = 0,sheet_name = 'Expected - 10.3_Final')

# set the column width as per your requirement
writer.sheets["ExpectedFinal - 10.1"].set_column('A:AA', 20)
writer.sheets["ExpectedFinal - 10.1 Final"].set_column('A:AA', 25)
writer.sheets["UnexpectedFinal - 10.1"].set_column('A:AA', 20)
writer.sheets["UnexpectedFinal - 10.1 Final"].set_column('A:AA', 25)
writer.sheets["Expected - 10.2"].set_column('A:AA', 20)
writer.sheets["Expected - 10.2 Final"].set_column('A:AA', 25)
writer.sheets["Recap Results"].set_column('A:AA', 25)
writer.sheets["Expected - 10.3_All"].set_column('A:AA', 20)
writer.sheets["Expected - 10.3_Final"].set_column('A:AA', 20)

# Close the Pandas Excel writer and output the Excel file.
writer.save()