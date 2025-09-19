import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
df = pd.read_csv('./archive/data.csv')
df = df.dropna()
X = df[['90s','Min','MP']]
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