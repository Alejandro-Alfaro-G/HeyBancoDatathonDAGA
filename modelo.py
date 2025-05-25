import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

print("¡Todas las librerías están instaladas!")

# usen el siguiente comando para instalar las dependencias necesarias:
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn jupyterlab tqdm
