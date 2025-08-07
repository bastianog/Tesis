#%%

##################################################################################################
#PREPARACIÓN DE DATOS Y MODELO

# Importar las librerías necesarias
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats



#%%
df = pd.read_excel("df_ordenado_para_analisis.xlsx")
#%%

# Reemplazar posibles textos como 'N/A' o vacíos por NaN
df['Superficie Afectada'] = pd.to_numeric(df['Superficie Afectada'], errors='coerce')
df['N Incendios'] = pd.to_numeric(df['N Incendios'], errors = 'coerce')

#%%
# Eliminar filas donde Superficie Afectada sea NaN ** No lo eliminé porque podría ser útil para el modelo
df = df.dropna(subset=['Superficie Afectada'])
df = df.dropna(subset=['N Incendios'])
#%%

#Renombramos las columnas para facilitar el trabajo del modelo
df = df.rename(columns={
    'Ventas anuales en UF': 'Ventas_uf',
    'Superficie Afectada': 'Superficie',
    'N Incendios': 'Incendios',
    'Comuna del domicilio o casa matriz': 'Comuna',
    'Año Comercial': 'Año',
    'Renta neta informada en UF': 'Renta_uf'
})
#%%
df['Superficie'] = df['Superficie'].fillna(0)
#%%

# Convertir las columnas a los tipos de datos adecuados
df['Número de empresas'] = pd.to_numeric(df['Número de empresas'], errors='coerce')
df['Año'] = df['Año'].astype(int)
df['Comuna'] = df['Comuna'].astype(str)
df['Superficie'] = pd.to_numeric(df['Superficie'], errors='coerce')
df['Renta_uf'] = pd.to_numeric(df['Renta_uf'], errors='coerce')
df['Ventas_uf'] = pd.to_numeric(df['Ventas_uf'], errors='coerce')
df['Incendios'] = pd.to_numeric(df['Incendios'], errors='coerce')
#%%

#Aplicamos logaritmo a las variables de superficie, ventas y renta para evitar problemas de escala 
# y normalizar la distribución

df['log_superficie'] = np.log(df['Superficie']+1)
df['log_ventas'] = np.log(df['Ventas_uf']+1)
df['log_renta'] = np.log(df['Renta_uf']+1)
#%%

# Calcular superficie promedio afectada por incendio
df['sup_promedio'] = np.where(
    df['Incendios'] == 0,
    np.nan,  # Si no hay incendios, superficie promedio es 0
    df['Superficie'] / df['Incendios']
)

#%%

# Aplicar logaritmo a la superficie promedio, evitando log(0) sumando 1
df['log_sup_prom']  = np.log(df['sup_promedio'] + 1) 
#%%
# Convertir las columnas a tipo numérico, manejando errores

df['sup_promedio'] = pd.to_numeric(df['sup_promedio'], errors='coerce')
df['log_superficie'] = pd.to_numeric(df['log_superficie'], errors='coerce')
df['log_sup_prom'] = pd.to_numeric(df['log_sup_prom'], errors='coerce')
df['log_ventas'] = pd.to_numeric(df['log_ventas'], errors='coerce')
df['log_renta'] = pd.to_numeric(df['log_renta'], errors='coerce')
df['log_sup_prom'] = pd.to_numeric(df['log_sup_prom'], errors='coerce')

#%%
# Eliminar columnas no utilizadas *** No lo eliminé porque podría ser útil para el modelo
df['Renta_uf'] = df['Renta_uf'].fillna(0)
df['Ventas_uf'] = df['Ventas_uf'].fillna(0) 
df['Número de empresas'] = df['Número de empresas'].fillna(0)
df['sup_promedio'] = df['sup_promedio'].fillna(0)
#%%
# Crear variable dummy: 1 si hubo incendio, 0 si no hubo
df['hubo_incendio'] = (df['Incendios'] > 0).astype(int)

#%%

# Abrir archivo de rubros económicos
dfrubros = pd.read_excel("PUB_COMU_RUBR.xlsb",header=4)
#%%

# Convertir las columnas a los tipos de datos adecuados
dfrubros = dfrubros.rename(columns={
    'Rubro': 'Rubro economico',
    'Ventas anuales en UF': 'Ventas_uf',
    'Superficie Afectada': 'Superficie',
    'N Incendios': 'Incendios',
    'Comuna del domicilio o casa matriz': 'Comuna',
    'Año Comercial': 'Año',
    'Renta neta informada en UF': 'Renta_uf'
})

#%%
dfrubros.head()
#%%
print(dfrubros['Rubro economico'].unique())
#%%
# Filtrar los datos por rubro económico de Industria Forestal
dfrubro1 = dfrubros[dfrubros['Rubro economico'] == 'A - Agricultura, ganadería, silvicultura y pesca']
#%%
# Filtrar los datos por rubro económico de Turismo
dfrubro2 = dfrubros[dfrubros['Rubro economico'] == 'I - Actividades de alojamiento y de servicio de comidas']

#%%
# Filtrar los datos por rubro económico de Inmobiliarias
dfrubro3 = dfrubros[dfrubros['Rubro economico'] == 'L - Actividades inmobiliarias']

#%%
# Filtrar los datos por rubro económico de Logística
dfrubro4 = dfrubros[dfrubros['Rubro economico'] == 'H - Transporte y almacenamiento']
#%%
# Agrupar por Comuna y Año, y calcular número de empresas, suma de ventas y renta
agg_rubro1 = dfrubro1.groupby(['Comuna', 'Año']).agg(
    empresas_rubro1=('Número de empresas', 'first'),
    ventas_rubro1=('Ventas_uf', 'sum'),
    renta_rubro1=('Renta_uf', 'sum')
).reset_index()

agg_rubro2 = dfrubro2.groupby(['Comuna', 'Año']).agg(
    empresas_rubro2=('Número de empresas', 'first'),
    ventas_rubro2=('Ventas_uf', 'sum'),
    renta_rubro2=('Renta_uf', 'sum')
).reset_index()

agg_rubro3 = dfrubro3.groupby(['Comuna', 'Año']).agg(
    empresas_rubro3=('Número de empresas', 'first'),
    ventas_rubro3=('Ventas_uf', 'sum'),
    renta_rubro3=('Renta_uf', 'sum')
).reset_index()

agg_rubro4 = dfrubro4.groupby(['Comuna', 'Año']).agg(
    empresas_rubro4=('Número de empresas', 'first'),
    ventas_rubro4=('Ventas_uf', 'sum'),
    renta_rubro4=('Renta_uf', 'sum')
).reset_index()
#%%

# Unir los datos agregados de los rubros con el DataFrame original
df = df.merge(agg_rubro1, on=['Comuna', 'Año'], how='left')
df = df.merge(agg_rubro2, on=['Comuna', 'Año'], how='left')
df = df.merge(agg_rubro3, on=['Comuna', 'Año'], how='left') 
df = df.merge(agg_rubro4, on=['Comuna', 'Año'], how='left')
#%%
df.head()

#%%
# Obtener los umbrales de percentiles
p50 = df['sup_promedio'].quantile(0.50)
p75 = df['sup_promedio'].quantile(0.75)
p90 = df['sup_promedio'].quantile(0.90)
p95 = df['sup_promedio'].quantile(0.95)
p99 = df['sup_promedio'].quantile(0.99)
p999 = df['sup_promedio'].quantile(0.999)
p9999 = df['sup_promedio'].quantile(0.9999)

# Crear columnas dummy
df['sup_mayor_50'] = (df['sup_promedio'] > p50).astype(int)
df['sup_mayor_75'] = (df['sup_promedio'] > p75).astype(int)
df['sup_mayor_90'] = (df['sup_promedio'] > p90).astype(int)
df['sup_mayor_95'] = (df['sup_promedio'] > p95).astype(int)
df['sup_mayor_99'] = (df['sup_promedio'] > p99).astype(int)
df['sup_mayor_999'] = (df['sup_promedio'] > p999).astype(int)
df['sup_mayor_9999'] = (df['sup_promedio'] > p9999).astype(int)

#%%

# Supongamos que tus columnas se llaman 'comuna' y 'año'
dfordenado = df.sort_values(['Comuna', 'Año'])


#%%
# Convierte las columnas a numérico, forzando errores a NaN
dfordenado['Renta_uf'] = pd.to_numeric(dfordenado['Renta_uf'], errors='coerce')
dfordenado['Ventas_uf'] = pd.to_numeric(dfordenado['Ventas_uf'], errors='coerce')
dfordenado['ventas_rubro1'] = pd.to_numeric(dfordenado['ventas_rubro1'], errors='coerce')
dfordenado['renta_rubro1'] = pd.to_numeric(dfordenado['renta_rubro1'], errors='coerce')
dfordenado['ventas_rubro2'] = pd.to_numeric(dfordenado['ventas_rubro2'], errors='coerce')
dfordenado['renta_rubro2'] = pd.to_numeric(dfordenado['renta_rubro2'], errors='coerce')
dfordenado['ventas_rubro3'] = pd.to_numeric(dfordenado['ventas_rubro3'], errors='coerce')
dfordenado['renta_rubro3'] = pd.to_numeric(dfordenado['renta_rubro3'], errors='coerce')
dfordenado['ventas_rubro4'] = pd.to_numeric(dfordenado['ventas_rubro4'], errors='coerce')
dfordenado['renta_rubro4'] = pd.to_numeric(dfordenado['renta_rubro4'], errors='coerce')

#%%
# Ahora puedes calcular la variación porcentual por comuna
dfordenado['var_pct_renta_uf'] = dfordenado.groupby('Comuna')['Renta_uf'].pct_change() * 100
dfordenado['var_pct_ventas_uf'] = dfordenado.groupby('Comuna')['Ventas_uf'].pct_change() * 100
#%%
dfordenado['var_pct_renta_r1'] = dfordenado.groupby('Comuna')['renta_rubro1'].pct_change() * 100
dfordenado['var_pct_ventas_r1'] = dfordenado.groupby('Comuna')['ventas_rubro1'].pct_change() * 100
dfordenado['var_pct_renta_r2'] = dfordenado.groupby('Comuna')['renta_rubro2'].pct_change() * 100
dfordenado['var_pct_ventas_r2'] = dfordenado.groupby('Comuna')['ventas_rubro2'].pct_change() * 100
dfordenado['var_pct_renta_r3'] = dfordenado.groupby('Comuna')['renta_rubro3'].pct_change() * 100
dfordenado['var_pct_ventas_r3'] = dfordenado.groupby('Comuna')['ventas_rubro3'].pct_change() * 100
dfordenado['var_pct_renta_r4'] = dfordenado.groupby('Comuna')['renta_rubro4'].pct_change() * 100
dfordenado['var_pct_ventas_r4'] = dfordenado.groupby('Comuna')['ventas_rubro4'].pct_change() * 100

#%%
dfordenado.to_excel("df_ordenado_rubros.xlsx", index=False)
#%%
###############################################################################################################################

#ANALISIS EXPLORATORIO DE DATOS Y DISTRIBUCIÓN
#Visualización de la distribución de las variables independientes sin aplicar logaritmo ni transformaciones lineales
import matplotlib.pyplot as plt
import scipy.stats as stats

for var in ['Superficie', 'sup_promedio', 'Incendios']:
    datos = pd.to_numeric(df[var], errors='coerce').dropna()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(datos, bins=30, color='orange', edgecolor='black')
    plt.title(f'Histograma de {var}')
    plt.subplot(1,2,2)
    stats.probplot(datos, dist="norm", plot=plt)
    plt.title(f'QQ-plot de {var}')
    plt.tight_layout()
    plt.show()
    shapiro_test = stats.shapiro(datos)
    print(f'{var} - Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}')
#%%
# Visualización de la distribución de las variables independientes aplicando logaritmo y transformaciones lineales
for var in ['log_superficie', 'log_sup_prom']:
    datos = pd.to_numeric(df[var], errors='coerce').dropna()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(datos, bins=30, color='orange', edgecolor='black')
    plt.title(f'Histograma de {var}')
    plt.subplot(1,2,2)
    stats.probplot(datos, dist="norm", plot=plt)
    plt.title(f'QQ-plot de {var}')
    plt.tight_layout()
    plt.show()
    shapiro_test = stats.shapiro(datos)
    print(f'{var} - Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}')
#%%
# Visualización de la distribución de las variables dependientes sin aplicar logaritmo ni transformaciones lineales

# Lista de variables dependientes a analizar
vars_dependientes = [
    'Renta_uf', 'Ventas_uf',
    'renta_rubro1', 'ventas_rubro1',
    'renta_rubro2', 'ventas_rubro2',
    'renta_rubro3', 'ventas_rubro3',
    'renta_rubro4', 'ventas_rubro4'
]

for var in vars_dependientes:
    # Convierte a numérico y elimina NaN
    datos = pd.to_numeric(df[var], errors='coerce').dropna()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(datos, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de {var}')
    plt.subplot(1,2,2)
    stats.probplot(datos, dist="norm", plot=plt)
    plt.title(f'QQ-plot de {var}')
    plt.tight_layout()
    plt.show()
    shapiro_test = stats.shapiro(datos)
    print(f'{var} - Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}')
#%%
df['ventas_rubro1'] = pd.to_numeric(df['ventas_rubro1'], errors='coerce')
df['renta_rubro1'] = pd.to_numeric(df['renta_rubro1'], errors='coerce')
df['ventas_rubro2'] = pd.to_numeric(df['ventas_rubro2'], errors='coerce')
df['renta_rubro2'] = pd.to_numeric(df['renta_rubro2'], errors='coerce') 
df['ventas_rubro3'] = pd.to_numeric(df['ventas_rubro3'], errors='coerce')
df['renta_rubro3'] = pd.to_numeric(df['renta_rubro3'], errors='coerce')
df['ventas_rubro4'] = pd.to_numeric(df['ventas_rubro4'], errors='coerce')
df['renta_rubro4'] = pd.to_numeric(df['renta_rubro4'], errors='coerce')

#%%
df['log_ventas_rubro1'] = np.log1p(df['ventas_rubro1'])
df['log_renta_rubro1'] = np.log1p(df['renta_rubro1'])    
df['log_ventas_rubro2'] = np.log1p(df['ventas_rubro2'])
df['log_renta_rubro2'] = np.log1p(df['renta_rubro2'])   
df['log_ventas_rubro3'] = np.log1p(df['ventas_rubro3'])
df['log_renta_rubro3'] = np.log1p(df['renta_rubro3'])
df['log_ventas_rubro4'] = np.log1p(df['ventas_rubro4'])
df['log_renta_rubro4'] = np.log1p(df['renta_rubro4'])

#%%

#Variables Dependientes luego de aplicar logaritmo
vars_dependientes = [
    'log_renta', 'log_ventas',
    'log_renta_rubro1', 'log_ventas_rubro1',
    'log_renta_rubro2', 'log_ventas_rubro2',
    'log_renta_rubro3', 'log_ventas_rubro3',
    'log_renta_rubro4', 'log_ventas_rubro4'
]

for var in vars_dependientes:
    # Convierte a numérico y elimina NaN
    datos = pd.to_numeric(df[var], errors='coerce').dropna()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(datos, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histograma de {var}')
    plt.subplot(1,2,2)
    stats.probplot(datos, dist="norm", plot=plt)
    plt.title(f'QQ-plot de {var}')
    plt.tight_layout()
    plt.show()
    shapiro_test = stats.shapiro(datos)
    print(f'{var} - Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}')

#Aunque el test de Shapiro-Wilk rechaza la normalidad (p < 0.05), la inspección visual sugiere que la distribución es aproximadamente normal, lo cual es suficiente para los supuestos del modelo OLS en muestras grandes.”

#%%
cols_inf = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        cols_inf.append(col)

print("Columnas numéricas con valores inf o -inf:", cols_inf)
#%%
####################################################################################################################

#Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de las comunas
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime

#%%
# Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de la comuna de Cauquenes

# Filtrar datos para la comuna de Cauquenes
df_cauquenes = df[df['Comuna'].str.lower() == 'cauquenes']

# Asegurarse de que 'Año' es numérico y ordenar
df_cauquenes['Año'] = pd.to_numeric(df_cauquenes['Año'], errors='coerce')
df_cauquenes = df_cauquenes.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_cauquenes['Año'], df_cauquenes['Renta_uf'], marker='o', label='Renta UF')
plt.plot(df_cauquenes['Año'], df_cauquenes['Ventas_uf'], marker='s', label='Ventas UF')



plt.title('Renta y Ventas en UF a través del tiempo - Comuna de Cauquenes')
plt.xlabel('Año')
plt.ylabel('UF')
plt.xticks(df_cauquenes['Año'].unique())  # Mostrar todos los años disponibles
plt.legend()  # <-- Agrega la leyenda al gráfico
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la superficie afectada (hectáreas) en la comuna de Cauquenes

# Filtrar datos para la comuna de Cauquenes
df_cauquenes = df[df['Comuna'].str.lower() == 'cauquenes']

# Asegurarse de que 'Año' es numérico y ordenar
df_cauquenes['Año'] = pd.to_numeric(df_cauquenes['Año'], errors='coerce')
df_cauquenes = df_cauquenes.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_cauquenes['Año'], df_cauquenes['Superficie'], marker='^', color='firebrick', label='Superficie afectada (ha)')

plt.title('Superficie afectada (hectáreas) a través del tiempo - Comuna de Cauquenes')
plt.xlabel('Año')
plt.ylabel('Superficie afectada (ha)')
plt.xticks(df_cauquenes['Año'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de la comuna de Nacimiento

# Filtrar datos para la comuna de Nacimiento
df_nacimiento = df[df['Comuna'].str.lower() == 'nacimiento']

# Asegurarse de que 'Año' es numérico y ordenar
df_nacimiento['Año'] = pd.to_numeric(df_nacimiento['Año'], errors='coerce')
df_nacimiento = df_nacimiento.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_nacimiento['Año'], df_nacimiento['Renta_uf'], marker='o', label='Renta UF')
plt.plot(df_nacimiento['Año'], df_nacimiento['Ventas_uf'], marker='s', label='Ventas UF')



plt.title('Renta y Ventas en UF a través del tiempo - Comuna de Nacimiento')
plt.xlabel('Año')
plt.ylabel('UF')
plt.xticks(df_nacimiento['Año'].unique())  # Mostrar todos los años disponibles
plt.legend()  # <-- Agrega la leyenda al gráfico
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la superficie afectada (hectáreas) en la comuna de Nacimiento

# Filtrar datos para la comuna de Nacimiento
df_nacimiento = df[df['Comuna'].str.lower() == 'nacimiento']

# Asegurarse de que 'Año' es numérico y ordenar
df_nacimiento['Año'] = pd.to_numeric(df_nacimiento['Año'], errors='coerce')
df_nacimiento = df_nacimiento.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_nacimiento['Año'], df_nacimiento['Superficie'], marker='^', color='firebrick', label='Superficie afectada (ha)')

plt.title('Superficie afectada (hectáreas) a través del tiempo - Comuna de Nacimiento')
plt.xlabel('Año')
plt.ylabel('Superficie afectada (ha)')
plt.xticks(df_nacimiento['Año'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de la comuna de Nacimiento

# Filtrar datos para la comuna de Pumanque
df_pumanque = df[df['Comuna'].str.lower() == 'pumanque']

# Asegurarse de que 'Año' es numérico y ordenar
df_pumanque['Año'] = pd.to_numeric(df_pumanque['Año'], errors='coerce')
df_pumanque = df_pumanque.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_pumanque['Año'], df_pumanque['Renta_uf'], marker='o', label='Renta UF')
plt.plot(df_pumanque['Año'], df_pumanque['Ventas_uf'], marker='s', label='Ventas UF')



plt.title('Renta y Ventas en UF a través del tiempo - Comuna de Pumanque')
plt.xlabel('Año')
plt.ylabel('UF')
plt.xticks(df_pumanque['Año'].unique())  # Mostrar todos los años disponibles
plt.legend()  # <-- Agrega la leyenda al gráfico
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la superficie afectada (hectáreas) en la comuna de Pumanque

# Filtrar datos para la comuna de Pumanque
df_pumanque = df[df['Comuna'].str.lower() == 'pumanque']

# Asegurarse de que 'Año' es numérico y ordenar
df_pumanque['Año'] = pd.to_numeric(df_pumanque['Año'], errors='coerce')
df_pumanque = df_pumanque.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_pumanque['Año'], df_pumanque['Superficie'], marker='^', color='firebrick', label='Superficie afectada (ha)')

plt.title('Superficie afectada (hectáreas) a través del tiempo - Comuna de Pumanque')
plt.xlabel('Año')
plt.ylabel('Superficie afectada (ha)')
plt.xticks(df_pumanque['Año'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de la comuna de Nacimiento

# Filtrar datos para la comuna de Molina
df_molina = df[df['Comuna'].str.lower() == 'molina']

# Asegurarse de que 'Año' es numérico y ordenar
df_molina['Año'] = pd.to_numeric(df_molina['Año'], errors='coerce')
df_molina = df_molina.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_molina['Año'], df_molina['Renta_uf'], marker='o', label='Renta UF')
plt.plot(df_molina['Año'], df_molina['Ventas_uf'], marker='s', label='Ventas UF')



plt.title('Renta y Ventas en UF a través del tiempo - Comuna de Molina')
plt.xlabel('Año')
plt.ylabel('UF')
plt.xticks(df_molina['Año'].unique())  # Mostrar todos los años disponibles
plt.legend()  # <-- Agrega la leyenda al gráfico
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la superficie afectada (hectáreas) en la comuna de Molina

# Filtrar datos para la comuna de Molina
df_molina = df[df['Comuna'].str.lower() == 'molina']

# Asegurarse de que 'Año' es numérico y ordenar
df_molina['Año'] = pd.to_numeric(df_molina['Año'], errors='coerce')
df_molina = df_molina.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_molina['Año'], df_molina['Superficie'], marker='^', color='firebrick', label='Superficie afectada (ha)')

plt.title('Superficie afectada (hectáreas) a través del tiempo - Comuna de Molina')
plt.xlabel('Año')
plt.ylabel('Superficie afectada (ha)')
plt.xticks(df_molina['Año'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Visualicemos un gráfico a través del tiempo donde observemos la superficie afectada (hectáreas) en la comuna de Florida

# Filtrar datos para la comuna de Florida
df_florida = df[df['Comuna'].str.lower() == 'florida']

# Asegurarse de que 'Año' es numérico y ordenar
df_florida['Año'] = pd.to_numeric(df_florida['Año'], errors='coerce')
df_florida = df_florida.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_florida['Año'], df_florida['Superficie'], marker='^', color='firebrick', label='Superficie afectada (ha)')

plt.title('Superficie afectada (hectáreas) a través del tiempo - Comuna de Florida')
plt.xlabel('Año')
plt.ylabel('Superficie afectada (ha)')
plt.xticks(df_florida['Año'].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Visualicemos un gráfico a través del tiempo donde observemos la renta y las ventas en uf de la comuna de Florida

# Filtrar datos para la comuna de Florida
df_florida = df[df['Comuna'].str.lower() == 'florida']

# Asegurarse de que 'Año' es numérico y ordenar
df_florida['Año'] = pd.to_numeric(df_florida['Año'], errors='coerce')
df_florida = df_florida.sort_values('Año')

plt.figure(figsize=(12, 6))
plt.plot(df_florida['Año'], df_florida['Renta_uf'], marker='o', label='Renta UF')
plt.plot(df_florida['Año'], df_florida['Ventas_uf'], marker='s', label='Ventas UF')

plt.title('Renta y Ventas en UF a través del tiempo - Comuna de Florida')
plt.xlabel('Año')
plt.ylabel('UF')
plt.xticks(df_florida['Año'].unique())  # Mostrar todos los años disponibles
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%


##############################################################################################################################
# MODELO DE REGRESIÓN LINEAL SIMPLE Y MÚLTIPLE INICIALES

#Modelo de regresión lineal (Variable dependiente: Ventas en UF)
X = df[['Incendios', 'Superficie']]
Y = df['Ventas_uf']

print(X.columns)

# Agregar constante
X = sm.add_constant(X)

# Ajustar modelo
modelo = sm.OLS(Y, X, missing='drop').fit()

# Ver resumen
print(modelo.summary())
#%%

#Modelo de regresión lineal (Variable dependiente: Renta en UF) donde los incendios y la superficie son mayores a 0
df1 = df[(df['Incendios'] > 0) | (df['Superficie'] > 0)]

X1 = df1[['Incendios', 'Superficie']]
Y1 = df1['Ventas_uf']

X1 = sm.add_constant(X1)

# Ajustar modelo
modelo = sm.OLS(Y1, X1, missing='drop').fit()

# Ver resumen
print(modelo.summary())

#%%
#Modelo filtrado tomando en consideración sólo gran número de incendios sobre el 98% y superficie afectada sobre el 98%

p_incendios = df['Incendios'].quantile(0.98)
p_superficie = df['Superficie'].quantile(0.98)

df2 = df[(df['Incendios'] > p_incendios) | (df['Superficie'] > p_superficie)]

X2 = df2[['Incendios', 'Superficie']]
Y2 = df2['Ventas_uf']

X2 = sm.add_constant(X2)

# Ajustar modelo
modelo = sm.OLS(Y2, X2, missing='drop').fit()

# Ver resumen
print(modelo.summary())
#%%
#Modelado a través de variable dependiente renta de todos los datos

df['Renta_uf'] = pd.to_numeric(df['Renta_uf'], errors='coerce')

X3 = df[['Incendios', 'Superficie']]
Y3 = df['Renta_uf']

# Agregar constante
X3 = sm.add_constant(X3)

# Ajustar modelo
modelo = sm.OLS(Y3, X3, missing='drop').fit()

# Ver resumen
print(modelo.summary())
#%%

#################################################################################################################################################
#MODELO DE PANEL INICIALES

# Crear índice de panel: (comuna, año)
df3 = df.set_index(['Comuna', 'Año'])

# Filtrar columnas necesarias
panel_data = df3[['Ventas_uf', 'Incendios', 'Superficie']].dropna()

# Definir modelo
modelo_panel = PanelOLS.from_formula(
    formula='Q("Ventas_uf") ~ Q("Incendios") + Q("Superficie") + EntityEffects',
    data=panel_data
)

# Ajustar modelo
resultados = modelo_panel.fit()
print(resultados.summary)
#%%
#Modelo de Panel ajustado a ventas en logaritmo
panel_data['log_ventas'] = np.log1p(panel_data['Ventas_uf'])

modelo = PanelOLS.from_formula(
    'log_ventas ~ Incendios + Superficie + EntityEffects',
    data=panel_data
)

resultados = modelo_panel.fit()
print(resultados.summary)
#%%
df['sup_promedio'].describe()

#%%
###########################################################################################################
#MODELO DE REGRESIÓN APLICANDO TRANSFORMACIONES LOGARÍTMICAS, VARIABLES DUMMY Y RUBROS
# Regresión: log_renta ~ log_superficie + Incendios para df

X = df[['log_superficie', 'Incendios']]
Y = df['log_renta']

X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()

print(modelo.summary())

#%%
# Regresión: log_renta ~ sup_promedio + Incendios para df

X = df[['sup_promedio', 'Incendios']]
Y = df['log_renta']

X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()

print(modelo.summary())

#%%
# Regresión: log_renta ~ sup_promedio + Incendios para df

X = df[['log_sup_prom', 'Incendios', 'hubo_incendio']]
Y = df['log_renta']

X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()

print(modelo.summary())

#%%
print("Percentiles de sup_promedio:")
print("50%:", df['sup_promedio'].quantile(0.50))
print("60%:", df['sup_promedio'].quantile(0.60))
print("70%:", df['sup_promedio'].quantile(0.70))
print("75%:", df['sup_promedio'].quantile(0.75))
print("80%:", df['sup_promedio'].quantile(0.80))
print("85%:", df['sup_promedio'].quantile(0.85))
print("90%:", df['sup_promedio'].quantile(0.90))
print("95%:", df['sup_promedio'].quantile(0.95))
print("96%:", df['sup_promedio'].quantile(0.96))
print("97%:", df['sup_promedio'].quantile(0.97))
print("97,5%:", df['sup_promedio'].quantile(0.975))
print("98%:", df['sup_promedio'].quantile(0.98))
print("99%:", df['sup_promedio'].quantile(0.99))
print("99,9%:", df['sup_promedio'].quantile(0.999))
print("99,99%:", df['sup_promedio'].quantile(0.9999))


#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]
Y = df['log_renta']

X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()

print(modelo.summary())

# Filtrar X y residuos solo para las filas usadas en el modelo
rows_used = modelo.model.data.row_labels
X_used = X.loc[rows_used]
residuos = modelo.resid

# Diagnóstico de residuos y multicolinealidad

# VIF (elimina filas con NaN o inf)
X_vif = X_used.drop('const', axis=1, errors='ignore')
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()
vif_data = pd.DataFrame()
vif_data["variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nVIF:")
print(vif_data)

# Normalidad de residuos
print("\nShapiro-Wilk p-value:", stats.shapiro(residuos).pvalue)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(residuos, bins=30)
plt.title('Histograma de residuos')
plt.subplot(1,2,2)
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('QQ-plot de residuos')
plt.tight_layout()
plt.show()

# Autocorrelación
print("Durbin-Watson:", durbin_watson(residuos))

# Homoscedasticidad
print("Breusch-Pagan p-value:", het_breuschpagan(residuos, X_used)[1])


#%%
#VIF:
#log_sup_prom: VIF ≈ 1.92
#Incendios: VIF ≈ 1.00
#hubo_incendio: VIF ≈ 2.90
#sup_mayor_50: VIF ≈ 1.92

#los VIF son menores a 5, lo que indica que no hay multicolinealidad 
#las variables independientes.

#el p-valor de Shapiro muy bajo (por ejemplo, 8.6e-24):
#Los residuos no siguen una distribución normal.

#Durbin Watson de 1.860960768331559, indica que no hay autocorrelación significativa en los residuos.

#p-valor de Breusch-Pagan de 4.5e-19 indica
#que existe heterocedasticidad, la varianza de los residuos no es constante.
#%%
# Calcular el porcentaje de empresas del rubro en la comuna
df['p_rubro1'] = df['empresas_rubro1'] / df['Número de empresas'] * 100

# Calcular percentil 75 del porcentaje
p75_pct = df['p_rubro1'].quantile(0.01250)

# Crear variable dummy
df['mayor_p_rubro1'] = (df['p_rubro1'] > p75_pct).astype(int)

# Filtrar comunas con mayor porcentaje de empresas del rubro
df_mayor_pct = df[df['mayor_p_rubro1'] == 1]

# Regresión ejemplo

X = df_mayor_pct[['sup_promedio', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]
Y = df_mayor_pct['log_ventas']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())
#%%
X = df_mayor_pct[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]
Y = df_mayor_pct['log_renta']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df_mayor_pct[['sup_promedio', 'sup_mayor_50']]
Y = df_mayor_pct['log_renta']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio']]
Y = df['log_ventas_rubro1']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())
# %%

X = df[['log_sup_prom', 'Incendios']]
Y = df['log_renta_rubro1']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]
Y = df['log_ventas_rubro2']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]  
Y = df['log_renta_rubro2']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]  
Y = df['log_ventas_rubro3']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio' , 'sup_mayor_50']]  
Y = df['log_renta_rubro3']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['sup_promedio', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]  
Y = df['log_ventas_rubro4']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())

#%%
X = df[['log_sup_prom', 'Incendios', 'hubo_incendio', 'sup_mayor_50']]  
Y = df['log_renta_rubro4']
X = sm.add_constant(X)
modelo = sm.OLS(Y, X, missing='drop').fit()
print(modelo.summary())


#%%
#############################################################################################################################
#MODELO DE PANEL 

# Crear índice de panel: (comuna, año)
df3 = df.set_index(['Comuna', 'Año'])
#Modelo de Panel ajustado a Renta en UF y sólo a incendios

# Filtrar columnas necesarias
panel_data = df3[['Renta_uf', 'Incendios']].dropna()

# Definir modelo
modelo_panel = PanelOLS.from_formula(
    formula='Q("Renta_uf") ~ Q("Incendios") + EntityEffects',
    data=panel_data
)

# Ajustar modelo
resultados = modelo_panel.fit()
print(resultados.summary)
#%%
#Modelo de Panel logarítmico ajustado a Renta en UF y sólo a incendios
panel_data['log_renta'] = np.log1p(panel_data['Renta_uf'])

modelo = PanelOLS.from_formula(
    'log_renta ~ Incendios + EntityEffects',
    data=panel_data
)

resultados = modelo_panel.fit()
print(resultados.summary)



cols = ['log_renta_rubro1', 'Incendios', 'Superficie', 'log_sup_prom', 'hubo_incendio', 'sup_mayor_50']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

panel_data = df.set_index(['Comuna', 'Año'])[cols].dropna()

Y = panel_data['log_renta_rubro1']
X = panel_data[[ 'log_sup_prom', 'sup_mayor_50']]
X = sm.add_constant(X)

from linearmodels.panel import PanelOLS
modelo_panel = PanelOLS(Y, X, entity_effects=True)
resultados = modelo_panel.fit()
print(resultados.summary)
#%%
# Supón que ya calculaste el porcentaje y la dummy
df_mayor_pct = df[df['mayor_p_rubro1'] == 1]

panel_data = df_mayor_pct.set_index(['Comuna', 'Año'])[
    ['renta_rubro1', 'Incendios', 'Superficie', 'log_sup_prom', 'hubo_incendio', 'sup_mayor_50']
].dropna()

Y = panel_data['renta_rubro1']
X = panel_data[['Incendios','log_sup_prom', 'sup_mayor_50']]
X = sm.add_constant(X)

from linearmodels.panel import PanelOLS
modelo_panel = PanelOLS(Y, X, entity_effects=True)
resultados = modelo_panel.fit()
print(resultados.summary)
# %%
# Rellenar con 0 las columnas relevantes
cols_a_rellenar = [
    'Incendios', 'Superficie', 'log_superficie',
    'sup_promedio', 'log_sup_prom'
]
dfordenado[cols_a_rellenar] = dfordenado[cols_a_rellenar].fillna(0)


#%%
# Muestra todas las filas que tienen al menos un NaN en cualquier columna
dfordenado = dfordenado[dfordenado['Año'] != 2005]

#%%
dfordenado.to_excel("df_ordenado_rubros.xlsx", index=False)

#%%
## SE REVISARON LOS DATOS DONDE NO HAY VENTAS NI RENTA y corresponda a 0 Y SE ELIMINARON
## PORQUE NO TIENEN SENTIDO EN EL ANÁLISIS Y NO CONTIENEN EMPRESAS.




columnas_modelo = [
    'var_pct_ventas_uf', 'Incendios', 'log_superficie', 'log_sup_prom', 'var_pct_renta_uf'
]

dfordenado[columnas_modelo] = dfordenado[columnas_modelo].replace([np.inf, -np.inf], np.nan)

dfordenado = dfordenado.dropna(subset=columnas_modelo)
# %%

X = dfordenado[['Incendios', 'log_superficie', 'log_sup_prom']]
X = sm.add_constant(X)  # Agrega el intercepto
y = dfordenado['var_pct_renta_uf']

modelo = sm.OLS(y, X).fit()

# 7. Mostrar el resumen del modelo
print(modelo.summary())
#%%

# Asegúrate de que 'comuna' y 'año' sean índices
dfordenado = dfordenado.set_index(['Comuna', 'Año'])

#%%
# Define la variable dependiente y las independientes
y = dfordenado['var_pct_ventas_uf']
X = dfordenado[['log_sup_prom', 'hubo_incendio']]
X = sm.add_constant(X)

# Ajusta el modelo de efectos fijos (sin constante, porque ya se controla por comuna)
modelo_panel = PanelOLS(y, X, entity_effects=True)
resultado = modelo_panel.fit()

print(resultado.summary)

#%%
#%%
# Define la variable dependiente y las independientes
y = dfordenado['var_pct_renta_rubro1']
X = dfordenado[['log_sup_prom', 'hubo_incendio']]


# Ajusta el modelo de efectos fijos (sin constante, porque ya se controla por comuna)
modelo_panel = PanelOLS(y, X, entity_effects=True)
resultado = modelo_panel.fit()

print(resultado.summary)

# %%
from sklearn.cluster import KMeans

#%%
X = df[['Superficie']].values
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

df['Severidad_Incendio'] = kmeans.labels_
# %%
plt.figure(figsize=(8,6))
for cluster in sorted(df['Severidad_Incendio'].unique()):
    subset = df[df['Severidad_Incendio'] == cluster]
    plt.scatter(subset.index, subset['Superficie'], label=f'Clúster {cluster}', alpha=0.7)

plt.xlabel('Índice')
plt.ylabel('Superficie afectada')
plt.title('Clústeres de Severidad de Incendios')
plt.legend()
plt.show()
# %%
df['severidad'] = df['Severidad_Incendio'].isin([1, 2, 3]).astype(int)
# %%

# Agrupar por Comuna y Año, y calcular la severidad promedio
dfordenado = dfordenado.merge(
    df[['Comuna', 'Año', 'severidad']],
    on=['Comuna', 'Año'],
    how='left'
)
# %%

# Define las variables
X = dfordenado[['log_superficie', 'severidad']]
X = sm.add_constant(X)  # Agrega el intercepto
y = dfordenado['var_pct_renta_uf']



# Ajusta el modelo
modelo = sm.OLS(y, X).fit()

# Muestra el resumen
print(modelo.summary())
# %%
# Elimina filas con NaN o inf en las columnas relevantes
X = X.replace([np.inf, -np.inf], np.nan)
dfordenado_clean = dfordenado.dropna(subset=['var_pct_renta_uf', 'log_superficie', 'severidad'])
X = dfordenado_clean[['log_superficie', 'severidad']]
X = sm.add_constant(X)
y = dfordenado_clean['var_pct_renta_uf']
# %%
dfordenado_panel = dfordenado.replace([np.inf, -np.inf], np.nan)
dfordenado_panel = dfordenado_panel.dropna(subset=['var_pct_renta_uf', 'log_superficie', 'severidad', 'Comuna', 'Año'])

# Establece el índice de panel
dfordenado_panel = dfordenado_panel.set_index(['Comuna', 'Año'])

# Define variables
y = dfordenado_panel['var_pct_renta_uf']
X = dfordenado_panel[['log_superficie', 'severidad']]
X = sm.add_constant(X)

# Modelo de efectos fijos por comuna
modelo_panel = PanelOLS(y, X, entity_effects=True)
resultado = modelo_panel.fit()

print(resultado.summary)


