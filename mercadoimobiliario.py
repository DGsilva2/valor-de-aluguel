import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import seaborn as sns

px.set_mapbox_access_token(open('token.txt').read())

df_data = pd.read_csv('sao-paulo-properties-april-2019.csv')

df_rent = df_data[df_data['Negotiation Type']== 'rent']
fig = px.scatter_mapbox(df_rent, lat='Latitude', lon='Longitude', color='Price', size='Size',
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15,zoom=10,opacity=0.4)

fig.update_coloraxes(colorscale = [[0, 'rgb(166,206,277,0.5)'],
                                   [0.02, 'rgb(31,120,180,0.5)'],
                                   [0.05, 'rgb(178,223,138,0.5)'],
                                   [0.10, 'rgb(51,160,44,0.5)'],
                                   [0.15, 'rgb(251,154,153,0.5)'],
                                   [1, 'rgb(227,26,28,0.52)']],)

fig.update_layout(height=500, width=850, mapbox=dict (center=go.layout.mapbox.Center(lat=-23.543138, lon=-46.69486)))
fig.show()

df_rent.info()
df_rent.hist(bins=30, figsize=(30,15))

df_rent.corr()
df_rent.corr()['Price'].sort_values(ascending=False)#correlacionando os dados para justificar o preço 


#PREPARAÇÃO DE DADOS PARA MODELOS DE ML
import sklearn

df_cleaned = df_rent.drop(['New', 'Property Type', 'Negotiation Type'], axis=1)
df_cleaned

#tratamento de dados categoricos
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

district_encoded = ordinal_encoder.fit_transform(df_rent[['District']])

district_encoded

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(df_cleaned[['District']])
housing_cat_1hot.toarray()


one_hot = pd.get_dummies(df_cleaned['District'])
df = df_cleaned.drop('District', axis=1)
df = df.join(one_hot)
df

#TREINANDO O MODELO 
from sklearn.model_selection import train_test_split

Y = df['Price']
X = df.loc[:,df.columns != 'Price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#LinearRegression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)#aqui ele vai utilizar a função custo para fazer a otimização do modelo

alguns_dados = x_train.iloc[:5]
alguns_label = y_train.iloc[:5]

print('Prediçoes:', lin_reg.predict(alguns_dados))
print('Labels:', alguns_label.values)
alguns_dados

#erro de predição
from sklearn.metrics import mean_squared_error

preds = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, preds)

lin_rmse = np.sqrt(lin_mse)
lin_rmse

## DecesionTree Regressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)

preds = tree_reg.predict(x_train)
tree_mse = mean_squared_error(y_train, preds)

tree_rmse = np.sqrt(tree_mse)
tree_rmse

#Comparando o modelos 
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:', scores)
    print('Mean score:', scores.mean())
    print('Standard deviation:', scores.std())

display_scores(tree_rmse_scores)

#modelo linear
scores = cross_val_score(lin_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:', scores)
    print('Mean score:', scores.mean())
    print('Standard deviation:', scores.std())

display_scores(lin_rmse_scores)

#MODELO DE FLORESTA DE REGRESSION
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

preds = rf_reg.predict(x_train)
rf_mse = mean_squared_error(y_train, preds)

rf_rmse = np.sqrt(rf_mse)
rf_rmse

#CROOS VALIDATION DO MODELO DE FLORESTA
scores = cross_val_score(rf_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10)
rf_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:', scores)
    print('Mean score:', scores.mean())
    print('Standard deviation:', scores.std())

display_scores(rf_rmse_scores)


##AVALIAR E OTIMIZAR O MODELO 

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features' : [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

grid_search.best_params_

final_model = grid_search.best_estimator_
final_model_predictions = final_model.predict(x_test)

final_mse = mean_squared_error(y_test, final_model_predictions)
print(np.sqrt(final_mse))


fig = go.Figure(data=[go.Scatter(y=y_test.values),
                      go.Scatter(y=final_model_predictions)])

fig.show()