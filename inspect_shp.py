import geopandas as gpd
import pandas as pd
import os

# --- CONFIGURACIÓN ---
#ARCHIVO_SHAPEFILE = "terminos_municipales/canarias/recintos_municipales_inspire_canarias_regcan95.shp"
#ARCHIVO_SHAPEFILE ='terminos_municipales/peninsula/recintos_municipales_inspire_peninbal_etrs89.shp'
ARCHIVO_SHAPEFILE = 'municipios_procesados/mapa_con_datos.shp'
ARCHIVO_SALIDA = "coordenadas_municipios.csv"

def exportar_a_csv():
    print(f"🔍 Leyendo: {ARCHIVO_SHAPEFILE}...")
    
    if not os.path.exists(ARCHIVO_SHAPEFILE):
        print("❌ Error: No encuentro el archivo .shp")
        return

    try:
        # 1. Cargar el archivo
        gdf = gpd.read_file(ARCHIVO_SHAPEFILE)
        print(f"   -> Columnas de atributos: {gdf.columns}")
        print(f"   -> Municipios cargados: {len(gdf)}")
        print(f"   -> Sistema de Coordenadas: {gdf.crs}")
        print(gdf.head(5))


    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")

if __name__ == "__main__":
    exportar_a_csv()