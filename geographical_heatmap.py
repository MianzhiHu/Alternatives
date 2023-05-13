import pandas as pd
import geopandas as gpd
import json
import folium
import matplotlib.pyplot as plt
from machine_learning_preparation import df_bhs, df_community_all


# Read data from xlsx file
df = pd.ExcelFile('data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx').parse('Agency - Summary')

# remove nan columns
df = df.dropna(axis=1, how='all')

# remove nan rows
df_all = df.dropna(axis=0, how='all')
df_zipcode = df_all['Zipcode'].dropna(axis=0, how='all')
df = df.dropna(axis=0, how='any')

# keep all the numbers as integers
df['Zipcode'] = df['Zipcode'].astype(int)
df['Case Number'] = df['Case Number'].astype(int)
df['Age'] = df['Age'].astype(int)

# find the maximum and minimum age
max_age = df['Age'].max()
min_age = df['Age'].min()

print(df['Age'].describe())

# visualize the distribution of age
plt.hist(df['Age'], bins=range(min_age, max_age + 1, 1))
plt.xlabel('Age')
plt.ylabel('Number of Cases')
plt.title('Distribution of Age')
plt.show()

df_age = df.groupby('Age')['Case Number'].count().reset_index()
print((df['Age'] >= 60).sum())

# select only participants with age between 10 and 18
df_age_selected = df[(df['Age'] >= 9) & (df['Age'] <= 25)]
print(df_age_selected['Age'].describe())

# check if there is any nan value in the case number column
print(df_all['Case Number'].isnull().values.any())
print(df_all['Program Name'].isnull().values.any())
print(df_all['Participant Role'].isnull().values.any())
print(df_all['Participant Role'].isnull().sum())
print(df_all['Age'].isnull().values.any())
print(df_all['Zipcode'].isnull().values.any())
print(df_all['Zipcode'].isnull().sum())

# check the rows with nan values in the participant role column
df_pr_null = df_all[df_all['Participant Role'].isnull()]

# remove the rows with zipcodes that are not 5 digits
df = df[df['Zipcode'].astype(str).str.len() == 5]

# check the categories in the program name column
print(df['Program Name'].unique())

# count the number of cases per zipcode
df_grouped = df.groupby('Zipcode')['Case Number'].count().reset_index()

# keep only the effective zipcodes
kept_zipcode = [60619, 60626, 60637, 60640]

# keep only the zipcodes in the list
df_bhs = df_bhs[df_bhs['Zipcode'].isin(kept_zipcode)]

# count the number of cases per zipcode
df_bhs_grouped = df_bhs.groupby('Zipcode')['Program Success'].sum().reset_index()


# read the zipcode shapefile
with open('data/cb_2018_us_zcta510_500k.geojson') as f:
    zipcode_US = json.load(f)

with open('data/Zip_Codes.geojson') as f:
    zipcode_Chicago = json.load(f)


# create a choropleth map
n = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
folium.Choropleth(
    geo_data=zipcode_US,
    name='choropleth',
    data=df_grouped,
    columns=['Zipcode', 'Case Number'],
    key_on='feature.properties.ZCTA5CE10',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Cases',
    nan_fill_color='transparent',
).add_to(n)
folium.LayerControl().add_to(n)
n.save('maps/US_map_choropleth.html')

n_Chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
folium.Choropleth(
    geo_data=zipcode_Chicago,
    name='choropleth',
    data=df_grouped,
    columns=['Zipcode', 'Case Number'],
    key_on='feature.properties.zip',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Cases',
    nan_fill_color='transparent',
).add_to(n_Chicago)
folium.LayerControl().add_to(n_Chicago)
n_Chicago.save('maps/chicago_map_choropleth.html')

# create a heatmap for program success
n_Chicago = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
folium.Choropleth(
    geo_data=zipcode_Chicago,
    name='choropleth',
    data=df_bhs_grouped,
    columns=['Zipcode', 'Program Success'],
    key_on='feature.properties.zip',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Program Succes',
    nan_fill_color='transparent',
).add_to(n_Chicago)
folium.LayerControl().add_to(n_Chicago)
n_Chicago.save('maps/chicago_program_success_choropleth.html')



