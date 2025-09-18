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
X = df[['90s','Min','MP','Rk']]
y = df['Gls']

scaler = MinMaxScaler()
X['MP'] = scaler.fit_transform(X[['MP']])
X['Min'] = scaler.fit_transform(X[['Min']])

selector = SelectKBest(score_func=f_regression, k=2)
selector.fit_transform(X, y)
scores = selector.scores_
variables = X.columns

selectorResults = pd.DataFrame({'Variable': variables, 'Score': scores}).sort_values(by='Score', ascending=False)
plt.figure(figsize=(10,6))
plt.bar(selectorResults['Variable'], selectorResults['Score'])
plt.title('Feature Selection using ANOVA F-test')
plt.xlabel('Variables')
plt.ylabel('Influence on Goals')
plt.show()

