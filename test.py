import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from model import EpidemicModel

# --- CONFIGURACIÓN ---
# Usamos el archivo que acabamos de crear con el script de preparación
SHAPEFILE_DATOS = 'municipios_procesados/mapa.shp' 
CONFIG_FILE = 'params.yaml'
DIAS_SIMULACION = 365

def main():
    print("🚀 INICIANDO SIMULADOR (Modo Geográfico)")
    
    # 1. CARGA DEL "TABLERO" (Shapefile procesado)
    print(f"🌍 Leyendo archivo maestro: {SHAPEFILE_DATOS}...")
    try:
        # Geopandas carga el archivo con geometría y datos
        gdf = gpd.read_file(SHAPEFILE_DATOS)
    except Exception as e:
        print(f"❌ Error: No se encuentra el archivo. Ejecuta primero 'preparar_mapa_excel.py'.\n{e}")
        return

    # 2. PREPARACIÓN MÍNIMA
    # Nos aseguramos de tener el índice interno 0..N para los tensores
    # El modelo necesita saber que el pueblo 0 es la fila 0, el 1 la fila 1, etc.
    gdf = gdf.reset_index(drop=True)
    gdf['id_municipio'] = gdf.index
    
    print(f"   -> {len(gdf)} Municipios cargados.")
    print(f"   -> Población Total: {gdf['poblacion'].sum():,}")

    # 3. INICIALIZAR MODELO
    # Le pasamos el GeoDataFrame entero.
    # El modelo buscará dentro las columnas 'coord_x', 'coord_y' y 'poblacion'.
    modelo = EpidemicModel(df_data=gdf, config_path=CONFIG_FILE)

    # 4. BUCLE DE SIMULACIÓN
    pbar = tqdm(range(DIAS_SIMULACION), desc="Simulando", unit="día")
    for dia in pbar:
        stats = modelo.step()
        
        # Info en la barra de progreso
        pbar.set_postfix(
            Infectados=f"{stats['I']:,}", 
            Viajes=f"{stats['Moves']:,}"
        )
        
    # 5. EXPORTAR RESULTADOS
    modelo.export_results("resultados_finales.csv")
    print("\n✅ Simulación terminada.")

if __name__ == "__main__":
    main()