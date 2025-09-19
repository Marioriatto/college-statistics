import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
df = pd.read_csv('./archive/data.csv')
df = df.dropna()
X = df[['Min','MP']]
y = df['Gls']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regresor = DecisionTreeRegressor()
regresor.fit(X_train, y_train)
y_hat = regresor.predict(X_test)

dfpredicho = X_test.copy()
dfpredicho['Gls_Real'] = y_test
dfpredicho['Gls_Predicho'] = y_hat

plt.figure(figsize=(10,6))
plt.scatter(dfpredicho['Gls_Real'], dfpredicho['Gls_Predicho'], color='blue', alpha=0.6)
plt.plot([dfpredicho['Gls_Real'].min(), dfpredicho['Gls_Real'].max()], [dfpredicho['Gls_Real'].min(), dfpredicho['Gls_Real'].max()], color='red', linestyle='--')
plt.xlabel('Gls Real')
plt.ylabel('Gls Predicho')
plt.title('Gls Real vs Gls Predicho')
plt.axline((0,0), slope=1,color='black',linewidth=0.5, ls='--')
plt.grid()
plt.show()

min_input = widgets.FloatSlider(value=50, min=0, max=100, step=1, description='Min:')
mp_input = widgets.FloatSlider(value=2000, min=0, max=4000, step=10, description='MP:')

mae = mean_absolute_error(y_test, y_hat)
print(f'Mean Absolute Error (MAE): {mae:.2f}')
mse = mean_squared_error(y_test, y_hat)
print(f'Mean Squared Error (MSE): {mse:.2f}')
rmse = root_mean_squared_error(y_test, y_hat)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
r2 = r2_score(y_test, y_hat)
print(f'R² Score: {r2:.2f}')