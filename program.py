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

