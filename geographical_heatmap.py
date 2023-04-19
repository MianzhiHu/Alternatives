import pandas as pd
import geopandas as gpd
import json
import folium


# Read data from xlsx file
df = pd.ExcelFile('data/KPI Backup Round 8 - 7.28 (1)[6572].xlsx').parse('Agency - Summary')

# remove nan columns
df = df.dropna(axis=1, how='all')

# remove the last column
df = df.iloc[:, :-1]

# remove nan rows
df_all = df.dropna(axis=0, how='all')
df_zipcode = df_all['Zipcode'].dropna(axis=0, how='all')
df = df.dropna(axis=0, how='any')

# keep all the numbers as integers
df['Zipcode'] = df['Zipcode'].astype(int)
df['Case Number'] = df['Case Number'].astype(int)
df['Age'] = df['Age'].astype(int)

# remove the rows with zipcodes that are not 5 digits
df = df[df['Zipcode'].astype(str).str.len() == 5]


# count the number of cases per zipcode
df_grouped = df.groupby('Zipcode')['Case Number'].count().reset_index()


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



