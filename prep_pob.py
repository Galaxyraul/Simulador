import pandas as pd

# Configuración de archivos
archivo_entrada = 'pobmun25.xlsx' # O .xlsx
archivo_salida = 'poblacion_procesada.csv'

# 1. Cargar solo las columnas que necesitas
# Es mejor usar los nombres 'NOMBRE' y 'POB25' que los índices numéricos,
# es más seguro por si cambias el orden en el Excel.
try:
    df = pd.read_excel(
        archivo_entrada, 
        usecols=['NOMBRE', 'POB25'] # Carga SOLO estas dos columnas
    )
    
    # Si prefieres hacerlo estrictamente por posición (4ª y 5ª columna):
    # Recuerda que en Python se empieza en 0. 
    # Columna 4 -> índice 3, Columna 5 -> índice 4
    # df = pd.read_excel(archivo_entrada, usecols=[3, 4]) 

    # 2. Renombrar columnas (Opcional, para estandarizar en tu simulación)
    # Esto asegura que en tu CSV final las cabeceras sean limpias
    df.columns = ['id_municipio', 'poblacion'] 

    # 3. Exportar a CSV
    # index=False evita que se guarde una columna extra con el número de fila (0,1,2...)
    df.to_csv(archivo_salida, index=False)
    
    print(f"✅ Éxito: Se han exportado {len(df)} filas a {archivo_salida}")
    print(df.head()) # Muestra las primeras 5 filas para verificar

except ValueError as e:
    print("❌ Error: No se encontraron las columnas 'NOMBRE' o 'POB25'. Revisa el Excel.")
    print(e)
except FileNotFoundError:
    print(f"❌ Error: No encuentro el archivo {archivo_entrada}")