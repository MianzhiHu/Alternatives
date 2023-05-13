import pandas as pd
import json
import folium

def heatmap_function(path, zipcode_template_path, save_path, file_name, sheet_number=1, color='YlGn',
                     fill_opacity=0.7, line_opacity=0.2, program=None, max_age=None, min_age=None,
                     participant_role=None):
    """
    :param path: path to the Excel file
    :param zipcode_template: path to the zipcode shapefile
    :param save_path: path to save the html file
    :param file_name: name of the html file
    :param sheet_number: which sheet to read from the Excel file
    :param color: color of the heatmap
    :param fill_opacity: fill opacity of the heatmap
    :param line_opacity: line opacity of the heatmap
    :param program: program of the participants
    :param max_age: maximum age of the participants
    :param min_age: minimum age of the participants
    :param participant_role: role of the participants
    :return: a choropleth map
    """

    # Read data from xlsx file
    excel = pd.ExcelFile(path)
    df = excel.parse(excel.sheet_names[sheet_number - 1])

    # remove nan columns
    df = df.dropna(axis=1, how='all')

    # remove nan rows
    df = df.dropna(axis=0, how='any')

    # keep all the numbers as integers
    df['Zipcode'] = df['Zipcode'].astype(int)
    df['Case Number'] = df['Case Number'].astype(int)

    # remove the rows with zipcodes that are not 5 digits
    df = df[df['Zipcode'].astype(str).str.len() == 5]

    # create specific filters
    if program is not None:
        df = df[df['Program Name'] == program]
    if max_age is not None:
        df['Age'] = df['Age'].astype(int)
        df = df[df['Age'] <= max_age]
    if min_age is not None:
        df['Age'] = df['Age'].astype(int)
        df = df[df['Age'] >= min_age]
    if participant_role is not None:
        df = df[df['Participant Role'] == participant_role]

    # count the number of cases per zipcode
    df_grouped = df.groupby('Zipcode')['Case Number'].count().reset_index()

    # read the zipcode shapefile
    with open(zipcode_template_path) as f:
        zipcode = json.load(f)

    # create a choropleth map
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
    folium.Choropleth(
        geo_data=zipcode,
        name='choropleth',
        data=df_grouped,
        columns=['Zipcode', 'Case Number'],
        # Change the key_on to match the key in the zipcode shapefile
        # For the US zipcode shapefile, use the key 'feature.properties.ZCTA5CE10'
        key_on='feature.properties.zip',
        fill_color=color,
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        legend_name='Number of Cases',
        nan_fill_color='transparent',
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(save_path + file_name + '.html')

    print('The map is saved to ' + save_path + file_name + '.html')
    return m

