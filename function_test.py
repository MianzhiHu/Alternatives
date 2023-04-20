from heatmap_function import heatmap_function

heatmap_function('data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx', 'data/Zip_Codes.geojson', 'maps/','heatmap_function')
heatmap_function('data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx', 'data/Zip_Codes.geojson', 'maps/','heatmap_function_age', max_age=18, min_age=10)
heatmap_function('data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx', 'data/Zip_Codes.geojson', 'maps/','heatmap_function_program', program='Clinical-Behavioral Health Services')
heatmap_function('data/Copy of KPI Backup Round 8 - 7.28 (1)(6572).xlsx', 'data/Zip_Codes.geojson', 'maps/','heatmap_function_participant_role', participant_role='Youth')