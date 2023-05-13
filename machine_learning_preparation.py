import pandas as pd


path = 'data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx'
sheet_number = 1

# Read data from xlsx file
excel = pd.ExcelFile(path)
df = excel.parse(excel.sheet_names[sheet_number - 1])

excel_sdq = pd.ExcelFile('data/SDQ Standalone Report.xlsx')
df_sdq = excel_sdq.parse(excel_sdq.sheet_names[0], header=1)

excel_sdq_all = pd.ExcelFile('data/SDQ FY23 Report.xlsx')
df_sdq_all = excel_sdq_all.parse(excel_sdq_all.sheet_names[0], header=1)

# remove nan columns
df = df.dropna(axis=1, how='all')
df_sdq = df_sdq.dropna(axis=1, how='all')
df_sdq_all = df_sdq_all.dropna(axis=1, how='all')

# remove nan rows
df = df.dropna(axis=0, how='any')
df_sdq = df_sdq.dropna(axis=0, how='all')
df_sdq_all = df_sdq_all.dropna(axis=0, how='all')

# keep all the numbers as integers
df['Zipcode'] = df['Zipcode'].astype(int)
df['Case Number'] = df['Case Number'].astype(int)

# remove the rows with zipcodes that are not 5 digits
df = df[df['Zipcode'].astype(str).str.len() == 5]

# merge the two dataframes
df_bhs = pd.merge(df, df_sdq, on='Case Number', how='right')

# clear the nan values
df_bhs = df_bhs.dropna(axis=0, how='all')

# clear wait list
df_bhs = df_bhs[df_bhs['Program Name'] != 'Clinical-Wait list']

# separate the dataframes
df_bhs_pre = df_bhs[df_bhs.iloc[:, 5] == 'Pre-Test']
df_bhs_post = df_bhs[df_bhs.iloc[:, 5] == 'Post-Test']

# check the values in the 12th column
print(df_bhs.iloc[:, 12].unique())

# remove the rows with nan values in the 12th column
df_bhs_post = df_bhs_post.dropna(subset=[df_bhs_post.columns[12]])
df_bhs_pre = df_bhs_pre.dropna(subset=[df_bhs_pre.columns[13]])

# recode the values in the 12th column
df_bhs_post.iloc[:, 12] = df_bhs_post.iloc[:, 12].replace(['A great deal', 'Quite a lot', 'Only a little', 'Not at all'], [4, 3, 2, 1])
df_bhs_pre.iloc[:, 13] = df_bhs_pre.iloc[:, 13].replace(['A great deal', 'Quite a lot', 'Only a little', 'Not at all'], [4, 3, 2, 1])

# select only 3rd and 12th columns
df_bhs_post = df_bhs_post.iloc[:, [2, 12]]
df_bhs_pre = df_bhs_pre.iloc[:, [0, 1, 2, 3, 4, 13, 14, 15]]

# separate df_sdq_all at the 6th column
df_sdq_all_pre = df_sdq_all.iloc[:, :6].dropna(axis=0, how='any')
df_sdq_all_post = df_sdq_all.iloc[:, 6:].dropna(axis=0, how='any')

# add the last two columns to be a new column
df_sdq_all_pre['Emotional Symptoms'] = df_sdq_all_pre.iloc[:, 4] + df_sdq_all_pre.iloc[:, 5]
df_sdq_all_post['Emotional Symptoms'] = df_sdq_all_post.iloc[:, 4] + df_sdq_all_post.iloc[:, 5]

# remove the last two columns
df_sdq_all_pre = df_sdq_all_pre.iloc[:, [0, 1, 2, 3, 6]]
df_sdq_all_post = df_sdq_all_post.iloc[:, [0, 1, 2, 3, 6]]

# rename the columns
df_sdq_all_pre.columns = ['Case Number', 'Peer Problems', 'Conduct Problems', 'Hyperactivity', 'Emotional Symptoms']
df_sdq_all_post.columns = ['Case Number', 'Peer Problems', 'Conduct Problems', 'Hyperactivity', 'Emotional Symptoms']

# merge the two dataframes
df_sdq_all = pd.merge(df_bhs_pre, df_sdq_all_pre, on='Case Number', how='right').dropna(axis=0, how='any')

# add columns to be a new column
df_sdq_all['Emotional Symptoms'] = df_sdq_all['Emotional Symptoms'] + df_sdq_all.iloc[:, 6]

# remove the redundant columns
df_sdq_all = df_sdq_all.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]

# merge again
df_bhs = pd.merge(df_sdq_all, df_bhs_post, on='Case Number', how='right')

# remove replicated rows
df_bhs = df_bhs.drop_duplicates(subset=['Case Number', 'Program Name'], keep='first')
df_bhs = df_bhs.dropna(axis=0, how='any')

# rename the columns
column_number = [5, 10]
column_name = ['Pre-Test Difficulty', 'Program Success']

df_bhs.rename(columns={df_bhs.columns[column_number[0]]: column_name[0], df_bhs.columns[column_number[1]]: column_name[1]}, inplace=True)


# now we deal with discharge summary
excel_discharge = pd.ExcelFile('data/Discharge Summary FY23 Report[7204].xlsx')
df_discharge = excel_discharge.parse(excel_discharge.sheet_names[0], header=1)
df_discharge = df_discharge.iloc[:, [1, 6]]

# merge with df_bhs
df_bhs_discharge = pd.merge(df_discharge, df, on='Case Number', how='left')
df_bhs_discharge = df_bhs_discharge.drop_duplicates(subset=['Case Number', 'Program Name'], keep='first')
df_bhs_discharge = df_bhs_discharge[df_bhs_discharge['Program Name'] != 'Clinical-Wait list']
df_bhs_discharge = pd.merge(df_bhs_discharge, df_bhs_pre, on='Case Number', how='left')
df_bhs_discharge = df_bhs_discharge.dropna(axis=0, how='any')

# the data size is too small (11 subjects), so we discard the data
# now we deal with the data from RJ
excel_rj = pd.ExcelFile('data/RJ Skills Survey Report (1)[6762].xlsx')
df_rj = excel_rj.parse(excel_rj.sheet_names[0], header=0)

# remove the rows with nan values
df_rj = df_rj.drop(df_rj.columns[[2, 6]], axis=1)
df_rj = df_rj.dropna(axis=0, how='any')

# divide the data into two parts
df_rj_pre = df_rj[df_rj['Type of services'] == 'Pretest']
df_rj_post = df_rj[df_rj['Type of services'] == 'Posttest']

# check the number of cases in df_rj_pre that are also in df_rj_post
df_rj_pre['Case Number'].isin(df_rj_post['Case Number']).value_counts()
print(f'The number of cases in df_rj_pre that are also in df_rj_post is {df_rj_pre.shape[0]-df_rj_pre["Case Number"].isin(df_rj_post["Case Number"]).value_counts()[0]}')

# there is no case in df_rj_pre that is also in df_rj_post, so we discard the data
# finally we deal with the data from community engagement
excel_community = pd.ExcelFile('data/Community Engagement Survey 2017 (1)[6763].xlsx')
df_community = excel_community.parse(excel_community.sheet_names[0], header=3)
df_community = df_community.dropna(axis=1, how='all').dropna(axis=0, how='any')


def code_community(x):
    if x == 'Strongly Agree':
        return 5
    elif x == 'Agree':
        return 4
    elif x == 'Neutral':
        return 3
    elif x == 'Disagree':
        return 2
    elif x == 'Strongly Disagree':
        return 1

df_community.iloc[:, 2:] = df_community.iloc[:, 2:].applymap(code_community)

# define a new column as the sum of all the questions
df_community['Community Engagement'] = df_community.iloc[:, 2:].sum(axis=1)
df_community = df_community.iloc[:, [0, 1, 17]]

# divide the data into two parts
df_community_pre = df_community[df_community.iloc[:, 0] == 'Pre-Test'].iloc[:, 1:]
df_community_post = df_community[df_community.iloc[:, 0] == 'Post-Test'].iloc[:, 1:]

# merge
df_community = pd.merge(df_community_pre, df_community_post, on='Case Number', how='right')
df_community = df_community.dropna(axis=0, how='any').drop_duplicates(subset=['Case Number'], keep='first')

# merge with df
df_community_all = pd.merge(df_community, df, on='Case Number', how='left').dropna(axis=0, how='any')

# rename the columns
column_number_community = [1, 2]
column_name_community = ['Pre-Test Community Engagement', 'Post-Test Community Engagement']

df_community_all.rename(columns={df_community_all.columns[column_number_community[0]]: column_name_community[0], df_community_all.columns[column_number_community[1]]: column_name_community[1]}, inplace=True)
