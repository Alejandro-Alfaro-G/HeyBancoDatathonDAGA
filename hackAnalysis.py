#initial commit

import pandas as pd

print("DATOS DE LA BASE DE CLIENTES")
# Load the dataset
df = pd.read_csv('HeyBancoDatathonDAGA/data/base_clientes_final.csv')

# Display the shape of the dataset
print(df.shape)

# Display the columns of the dataset
print(df.columns)  

# Display the data types of the columns
print(df.dtypes)
# Display the summary statistics of the dataset
print(df.describe())    

print("DATOS DE LA BASE DE TRANSACCIONES")

# Load the dataset
df_t = pd.read_csv('HeyBancoDatathonDAGA/data/base_transacciones_final.csv')

# Display the shape of the dataset
print(df_t.shape)
# Display the columns of the dataset
print(df_t.columns)
# Display the data types of the columns
print(df_t.dtypes)
# Display the summary statistics of the dataset
print(df_t.describe())


print("vamos a por toda en este hack")