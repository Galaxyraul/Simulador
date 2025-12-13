import pandas as pd
import time
from tqdm import tqdm
from model import EpidemicModel

# CONFIGURACIÓN
CSV_FILE = 'poblacion_procesada.csv' # Tu archivo original
CONFIG_FILE = 'params.yaml'
DIAS_SIMULACION = 100

def main():
    print("🚀 INICIANDO SIMULADOR (Gestión de Datos en Main)")
    print("================================================")
    
    # 1. ETL: EXTRACCIÓN Y TRANSFORMACIÓN
    print(f"📂 Cargando {CSV_FILE}...")
    
    # Leemos todo como string para no romper IDs como "001"
    df = pd.read_csv(CSV_FILE, dtype={'id_municipio': str})
    
    print(f"   Municipios encontrados: {len(df)}")
    
    # --- LA MAGIA: REEMPLAZO DE ID ---
    # Guardamos el ID original (string) para referencias futuras o logs
    df['id_original'] = df['id_municipio']
    
    # Sobrescribimos 'id_municipio' con el índice numérico (0, 1, 2...)
    # Esto es lo que la GPU necesita: enteros secuenciales.
    df['id_municipio'] = df.index
    
    print("✅ IDs transformados a enteros secuenciales (0..N) para la GPU.")

    # 2. INICIALIZAR MODELO
    # Le pasamos el DataFrame ya modificado, NO la ruta del archivo
    try:
        start_init = time.time()
        modelo = EpidemicModel(df_data=df, config_path=CONFIG_FILE)
        end_init = time.time()
        print(f"⏱️  Modelo inicializado en {end_init - start_init:.2f} s")
    except Exception as e:
        print(f"❌ Error al iniciar modelo: {e}")
        return

    # 3. BUCLE DE SIMULACIÓN
    print(f"\n▶️  Ejecutando {DIAS_SIMULACION} días...")
    pbar = tqdm(range(DIAS_SIMULACION), desc="Simulación", unit="día")
    
    for dia in pbar:
        stats = modelo.step()
        
        # Actualizamos la barra con info en tiempo real
        pbar.set_postfix(
            Inf=f"{stats['I']:,}", 
            Mue=f"{stats['D']:,}", 
            Viajes=f"{stats['Moves']:,}"
        )

    # 4. EXPORTACIÓN
    print("\n💾 Guardando resultados...")
    # Si quieres recuperar los nombres originales en el CSV final:
    # Podrías hacer un merge con el df original si guardas estadísticas por pueblo.
    # Para estadísticas globales, basta con esto:
    modelo.export_results("resultados_finales.csv")
    
    print("✅ Proceso terminado.")

if __name__ == "__main__":
    main()