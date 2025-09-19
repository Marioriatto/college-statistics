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

scaler = MinMaxScaler()
X['MP'] = scaler.fit_transform(X[['MP']])
X['Min'] = scaler.fit_transform(X[['Min']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_pred)
plt.xlabel('Features')
plt.ylabel('Predicted Goals')
plt.title('Decision Tree Regression Predictions')
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test)
plt.xlabel('Features')
plt.ylabel('Actual Goals')
plt.title('Actual Goals')
plt.tight_layout()
plt.show()