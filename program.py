import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#from sklearn.linear_model import linear_regression
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
df = pd.read_csv('./archive/data.csv')
df = df.dropna()
X = df[['Age','Min','MP']]
y = df['Gls']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

selector = SelectKBest(score_func=f_regression, k=2)
X_train = selector.fit_transform(X_train, y_train)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
plt.show()
