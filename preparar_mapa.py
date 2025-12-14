import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

# --- CONFIGURACIÓN ---
# 1. Archivo Península (ETRS89)
ARCHIVO_PENINSULA = "terminos_municipales/peninsula/recintos_municipales_inspire_peninbal_etrs89.shp"
# 2. Archivo Canarias (REGCAN95) - ¡PON AQUÍ TU RUTA!
ARCHIVO_CANARIAS  = "terminos_municipales/canarias/recintos_municipales_inspire_canarias_regcan95.shp"

ARCHIVO_EXCEL  = "pobmun25.xlsx"
ARCHIVO_SALIDA = "municipios_procesados/mapa.shp"

# Columnas
COL_NATIONALCO = "NATCODE" # Ojo: Asegúrate que en Canarias se llame igual. Si no, avísame.
XLS_PROV, XLS_MUN, XLS_POB = "CPRO", "CMUN", "POB25"

def extraer_poligonos_seguros(geom):
    """ Filtro para evitar errores de guardado (GeometryCollection) """
    if geom is None or geom.is_empty: return None
    if geom.geom_type in ['Polygon', 'MultiPolygon']: return geom
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        return unary_union(polys) if polys else None
    return None

def unificar_espana():
    print("🚀 INICIANDO FUSIÓN DE ESPAÑA (PENÍNSULA + CANARIAS)...")

    # 1. CARGA Y ESTANDARIZACIÓN A WGS84
    # -----------------------------------------------------------
    print("🌍 Cargando y traduciendo coordenadas...")
    
    if not os.path.exists(ARCHIVO_PENINSULA) or not os.path.exists(ARCHIVO_CANARIAS):
        print("❌ ERROR: Revisa las rutas de los archivos .shp")
        return

    # Cargamos y convertimos a EPSG:4326 (Lat/Lon Mundial) al vuelo
    gdf_pen = gpd.read_file(ARCHIVO_PENINSULA).to_crs(epsg=4326)
    gdf_can = gpd.read_file(ARCHIVO_CANARIAS).to_crs(epsg=4326)
    
    print(f"   -> Península: {len(gdf_pen)} filas")
    print(f"   -> Canarias:  {len(gdf_can)} filas")

    # 2. UNIÓN (CONCATENAR)
    # -----------------------------------------------------------
    print("🔗 Pegando los dos mapas...")
    # ignore_index=True es vital para que no haya índices duplicados
    gdf_total = pd.concat([gdf_pen, gdf_can], ignore_index=True)
    
    print(f"   -> Total filas brutas: {len(gdf_total)}")

    # 3. EXTRAER CÓDIGO INE (Lógica Últimos 5 dígitos)
    # -----------------------------------------------------------
    print("✂️  Extrayendo códigos INE (últimos 5)...")
    # .strip() quita espacios en blanco por si acaso
    gdf_total['INE_MERGE'] = gdf_total[COL_NATIONALCO].astype(str).str.strip().str[-5:]

    # 4. LIMPIEZA Y DISSOLVE
    # -----------------------------------------------------------
    print("🧩 Unificando municipios fragmentados...")
    gdf_total['geometry'] = gdf_total.geometry.buffer(0)
    gdf_dissolved = gdf_total.dissolve(by='INE_MERGE', as_index=False)
    
    total_municipios = len(gdf_dissolved)
    print(f"   📊 Municipios Reales Únicos: {total_municipios}")
    
    if total_municipios > 8100:
        print("   ✅ ¡PINTA BIEN! Tienes prácticamente toda España.")
    else:
        print("   ⚠️  OJO: Faltan municipios. ¿Seguro que es el archivo de Canarias completo?")

    # 5. FIX GEOMETRÍA (Evitar crash al guardar)
    # -----------------------------------------------------------
    print("🚑 Limpiando geometrías corruptas...")
    gdf_dissolved['geometry'] = gdf_dissolved.geometry.apply(extraer_poligonos_seguros)
    gdf_dissolved = gdf_dissolved.dropna(subset=['geometry'])
    # Todo a MultiPolygon
    gdf_dissolved['geometry'] = gdf_dissolved.geometry.apply(
        lambda x: MultiPolygon([x]) if x.geom_type == 'Polygon' else x
    )

    # 6. CONVERTIR A METROS (PARA EL SIMULADOR)
    # -----------------------------------------------------------
    print("🌐 Pasando mapa final a METROS (UTM 30N)...")
    # Ahora que están unidos, los proyectamos juntos
    gdf_final = gdf_dissolved.to_crs(epsg=25830)

    # 7. CRUCE CON CENSO
    # -----------------------------------------------------------
    print("📊 Cruzando con la Población...")
    df = pd.read_excel(ARCHIVO_EXCEL)
    df['INE_MERGE'] = (
        df[XLS_PROV].fillna(0).astype(int).astype(str).str.zfill(2) + 
        df[XLS_MUN].fillna(0).astype(int).astype(str).str.zfill(3)
    )
    df = df.groupby('INE_MERGE', as_index=False).agg({XLS_POB: 'sum'})

    # Merge
    gdf_final = gdf_final.merge(df[['INE_MERGE', XLS_POB]], on='INE_MERGE', how='right')
    
    # Chequeo de población perdida
    sin_datos = gdf_final[gdf_final[XLS_POB].isna()]
    if len(sin_datos) > 0:
        print(f"   ⚠️  Hay {len(sin_datos)} municipios en el mapa sin datos en el Excel.")
    
    gdf_final['poblacion'] = gdf_final[XLS_POB].fillna(0).astype(int)

    # 8. COORDENADAS Y GUARDADO
    # -----------------------------------------------------------
    print("📍 Calculando coordenadas X/Y...")
    gdf_final['coord_x'] = gdf_final.geometry.centroid.x
    gdf_final['coord_y'] = gdf_final.geometry.centroid.y

    # Limpieza de columnas
    cols = ['INE_MERGE', 'poblacion', 'coord_x', 'coord_y', 'geometry']
    for c in ['NAMEUNIT', 'NOMBRE', 'nombre', 'municipio']:
        if c in gdf_final.columns: cols.append(c); break
    
    gdf_final = gdf_final[[c for c in cols if c in gdf_final.columns]]
    gdf_final['id_mun'] = range(len(gdf_final))

    print(f"💾 Guardando: {ARCHIVO_SALIDA}")
    gdf_final.to_file(ARCHIVO_SALIDA)
    print("✅ ¡OPERACIÓN COMPLETADA! Ya tienes España entera unificada.")

if __name__ == "__main__":
    unificar_espana()