# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#%%
import pandas as pd
#%%
#cargamos el archivo excel
df_empresas = pd.read_excel("PUB_COMU.xlsb")
xls = pd.ExcelFile("Incendios por temporada.xlsx")

#%%
#Permite generar una database con cada hoja junta y asignandole la categoría
#de la hoja que llevaba
df_incendios = pd.concat([xls.parse(sheet_name).assign(Temporada=sheet_name) 
                          for sheet_name in xls.sheet_names])
#%%
# Reasignar los nombres de columnas desde la fila 0 (puede ser 1 u otra si es necesario)

#Reasigna nombre de filas con las de la fila 4 y elimina las filas que no me sirven
df_empresas.columns = df_empresas.iloc[3]
df_empresas = df_empresas.iloc[3:].reset_index(drop=True)

df_incendios.columns = df_incendios.iloc[2]
df_incendios = df_incendios.iloc[2:].reset_index(drop=True)

#%%
# Estandarizar nombres de comunas para tener los mismos valores en una columna llamada de igual manera
df_empresas["Comuna"] = df_empresas["Comuna del domicilio o casa matriz"].str.strip().str.upper()
df_incendios["Comuna"] = df_incendios["COMUNA"].str.strip().str.upper()

#%%
#eliminamos filas que no me sirven para las variables independientes
df_incendios = df_incendios.drop(df_incendios.columns[4:18], axis=1)

#%%
#eliminar nan de las comunas
df_incendios = df_incendios.dropna(subset=['COMUNA'])

#%%
#eliminar temporadas que no se utilizarán
df_incendios = df_incendios[(df_incendios['2023-2024'] >= '2004-2005') & (df_incendios['2023-2024'] <= '2022-23')]

#%%
#Eliminar filas de totales
df_incendios = df_incendios[~df_incendios['COMUNA'].str.contains('TOTAL', case=False, na=False)]

#%%
# Convertir temporada (por ejemplo: '2004-05') en año comercial (como 2005)
df_incendios["Año"] = df_incendios["2023-2024"].str[-2:].astype(int) + 2000

#%%
#Asegurar formato de columna
df_incendios['temporada'] = df_incendios['2023-2024'].astype(str)

df_incendios['Año_Comercial'] = df_incendios['temporada'].str[:4].astype(int) + 1

#%%
#Armamos un nuevo dataframe que crucen ambas columnas respecto a año comercial
#y comuna
df_merged = pd.merge(
    df_empresas,
    df_incendios,
    left_on=['Comuna', 'Año Comercial'],
    right_on=['Comuna', 'Año'],
    how='left'  # Mantiene todas las filas de empresas, con info de incendios si existe
)

#%%
#Eliminar columnas que no sirven

df_merged = df_merged.iloc[1:].reset_index(drop=True)

indices_a_eliminar = [24, 25, 26, 27, 29, 31, 32, 33, 34]

#%%
# Crear lista de índices de columnas a conservar
indices_a_conservar = [i for i in range(df_merged.shape[1]) if i not in indices_a_eliminar]

#%%
# Filtrar columnas por posición
df_merged = df_merged.iloc[:, indices_a_conservar]

#%%
#Renombrar columnas
df_merged = df_merged.rename(columns={
    'NUMERO INCENDIOS ': 'N Incendios',
    'TOTAL SUPERFICIE AFECTADA': 'Superficie Afectada',
})

#%%
#Reordenar las variables independientes
cols = df_merged.columns.tolist()

#Extraer las últimas dos columnas
Variables_X = cols[-2:]

# 3. Eliminar esas columnas de su posición original
resto = cols[:-2]  # todas menos las dos últimas

# 4. Insertar las dos columnas al índice 4 y 5
# Insertamos en orden inverso para que no se altere el orden original
resto.insert(4, Variables_X[0])  # primera de las últimas dos va a posición 5 (índice 4)
resto.insert(5, Variables_X[1])  # segunda va a posición 6 (índice 5)

# 5. Reordenar el DataFrame
df_merged = df_merged[resto]

#%%
df_merged.to_excel('df_ordenado_para_analisis.xlsx', index=False)

#%%
dfrubros = pd.read_excel("PUB_COMU_RUBR.xlsb",header=4)
#%%
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
dfrubro = dfrubros[dfrubros['Rubro economico'] == 'A - Agricultura, ganadería, silvicultura y pesca']
#%%
# Agrupar por Comuna y Año, y calcular número de empresas, suma de ventas y renta
agg_rubro = dfrubro.groupby(['Comuna', 'Año']).agg(
    n_empresas=('Número de empresas', 'first'),
    ventas_rubro=('Ventas_uf', 'sum'),
    renta_rubro=('Renta_uf', 'sum')
).reset_index()
#%%
df_merged = df_merged.merge(agg_rubro, on=['Comuna', 'Año'], how='left')
#%%
df_merged.head()
#%%

print(df_merged.columns)
print(agg_rubro.columns)
# %%
