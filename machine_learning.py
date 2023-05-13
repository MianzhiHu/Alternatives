from machine_learning_preparation import df_bhs, df_community_all
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz

# visualize the correlation
sns.heatmap(df_bhs.iloc[:, [5, 6, 7, 8, 9, 10]].corr(), annot=True, cmap='coolwarm')
# change the size of the heatmap
plt.rcParams['figure.figsize'] = (18, 18)
# plt.savefig('maps/correlation.png', dpi=600)
plt.show()

# test the significance of the correlation
corr_list = []
p_list = []
for i in range(5, 10):
    corr, p = pearsonr(df_bhs.iloc[:, i], df_bhs.iloc[:, 10])
    corr_list.append(corr)
    p_list.append(p)
    print('Correlation between {} and {}: {}'.format(df_bhs.columns[i], df_bhs.columns[10], corr))
    print('p-value: {}'.format(p))

# visualize the correlation
plt.bar(df_bhs.columns[5:10], corr_list)
plt.ylabel('Correlation')
plt.title('Correlation between Variables and Program Success')
plt.xticks(rotation=45)
plt.rcParams['figure.figsize'] = (12, 12)
# plt.savefig('maps/correlation_bar.png', dpi=300)
plt.show()

# machine learning starts here
# data preparation
# remove the Case Number column
df_bhs_ml = df_bhs.drop(['Case Number', 'Participant Role'], axis=1)
df_community_all_ml = df_community_all.drop(['Case Number', 'Participant Role'], axis=1)

# convert numeric values to categorical values
df_bhs_ml['Zipcode'] = df_bhs['Zipcode'].astype('category').cat.codes
df_bhs_ml['Program Name'] = df_bhs['Program Name'].astype('category').cat.codes
df_bhs_ml['Program Success'] = df_bhs['Program Success'].astype('category')

# keep a copy of matches between the numeric values and the categorical values
# merge the two columns into a dataframe
df_zipcode = pd.DataFrame({'Zipcode': df_bhs['Zipcode'], 'Zipcode Code': df_bhs_ml['Zipcode']})
df_program_name = pd.DataFrame({'Program Name': df_bhs['Program Name'], 'Program Name Code': df_bhs_ml['Program Name']})

# drop the duplicated rows
df_zipcode = df_zipcode.drop_duplicates()
df_program_name = df_program_name.drop_duplicates()

# sort the dataframe by the numeric values
df_zipcode = df_zipcode.sort_values(by=['Zipcode Code'])
df_program_name = df_program_name.sort_values(by=['Program Name Code'])

# keep zipcode as integer
df_zipcode['Zipcode'] = df_zipcode['Zipcode'].astype('int64')

# export the dataframes to csv files
df_zipcode.to_csv('data/zipcode.csv', index=False)
df_program_name.to_csv('data/program_name.csv', index=False)

# split the data into training and testing sets
X = df_bhs_ml.iloc[:, :-1]
y = df_bhs_ml.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression does not work well
# multiple linear regression does not work well
rf = RandomForestClassifier(random_state=42, criterion='entropy')
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5],
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(cross_val_score(grid_search.best_estimator_, X, y, cv=5))

rf_best = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=8, min_samples_leaf=1, n_estimators=100)
rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(rf_best.score(X_test, y_test))
print(f'R^2: {r2_score(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')


# visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='coolwarm')
# recode the labels
plt.xticks([0.5, 1.5, 2.5, 3.5], ['Not at all', 'Only a little', 'Quite a lot', 'A great deal'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['Not at all', 'Only a little', 'Quite a lot', 'A great deal'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (8, 8)
plt.savefig('maps/confusion_matrix.png', dpi=300)
plt.show()

# visualize the feature importance
feature_importance = pd.Series(rf_best.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.rcParams['figure.figsize'] = (15, 12)
# plt.savefig('maps/feature_importance.png', dpi=300)
plt.show()

# visualize the decision tree
dot_data = tree.export_graphviz(rf_best.estimators_[0], out_file=None, feature_names=X.columns,
                                class_names=['A great deal', 'Quite a lot', 'Only a little', 'Not at all'], filled=True,
                                rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('maps/decision_tree')


# now use the model to predict the community data
# convert numeric values to categorical values
df_community_all_ml['Zipcode'] = df_community_all['Zipcode'].astype('category').cat.codes
df_community_all_ml['Program Name'] = df_community_all['Program Name'].astype('category').cat.codes

# keep a copy of matches between the numeric values and the categorical values
# merge the two columns into a dataframe
df_zipcode_community = pd.DataFrame({'Zipcode': df_community_all['Zipcode'], 'Zipcode Code': df_community_all_ml['Zipcode']})
df_program_name_community = pd.DataFrame({'Program Name': df_community_all['Program Name'], 'Program Name Code': df_community_all_ml['Program Name']})

# drop the duplicated rows
df_zipcode_community = df_zipcode_community.drop_duplicates()
df_program_name_community = df_program_name_community.drop_duplicates()

# sort the dataframe by the numeric values
df_zipcode_community = df_zipcode_community.sort_values(by=['Zipcode Code'])
df_program_name_community = df_program_name_community.sort_values(by=['Program Name Code'])

# keep zipcode as integer
df_zipcode_community['Zipcode'] = df_zipcode_community['Zipcode'].astype('int64')

# export the dataframes to csv files
df_zipcode_community.to_csv('data/zipcode_community.csv', index=False)
df_program_name_community.to_csv('data/program_name_community.csv', index=False)

# split the data into training and testing sets
X = df_community_all_ml.iloc[:, [0, 2, 3, 4]]
y = df_community_all_ml.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(random_state=42)

rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# visualize the model by plotting the predicted values against the actual values
y_test = np.array(y_test, dtype=np.float64)
y_pred = np.array(y_pred, dtype=np.float64)
plt.scatter(y_test, y_pred)
plt.plot([25, 75], [25, 75], color='red', linestyle='--')
plt.xlim(25, 75)
plt.ylim(25, 75)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='green')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
# add legend
plt.legend(['Value Points', 'Perfect Prediction', 'Actual Prediction'])
plt.title('Actual Values vs Predicted Values')
plt.rcParams['figure.figsize'] = (8, 8)
plt.savefig('maps/actual_vs_predicted_community.png')
plt.show()

# visualize the feature importance
feature_importance = pd.Series(rf_regressor.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.yticks(rotation=45)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.rcParams['figure.figsize'] = (15, 12)
plt.savefig('maps/feature_importance_community.png', dpi=300)
plt.show()

# visualize the decision tree
dot_data = tree.export_graphviz(rf_regressor.estimators_[0], out_file=None, feature_names=X.columns,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('maps/decision_tree_community')
















