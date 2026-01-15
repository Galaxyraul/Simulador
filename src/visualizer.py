import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

class VisualizadorMapa:
    def __init__(self, ruta_shp, dot_scale=5000):
        self.dot_scale = dot_scale
        archivo_existe = os.path.exists(ruta_shp)
        
        # 1. MODO DUMMY
        if not HAS_GEOPANDAS or not archivo_existe:
            self.dummy = True
            print(f"⚠️ MODO DUMMY (Geopandas: {HAS_GEOPANDAS}, Archivo: {archivo_existe})")
            self.gdf = pd.DataFrame()
            self.all_x = np.random.rand(500) * 100
            self.all_y = np.random.rand(500) * 100
            self.point_belongs_to = np.random.randint(0, 10, 500)
            self.estado_visual = np.zeros(500)
            
        # 2. MODO REAL
        else:
            print(f"🌍 Cargando mapa real: {ruta_shp}")
            self.dummy = False
            try:
                self.gdf = gpd.read_file(ruta_shp)
                self.gdf['coord_x'] = self.gdf.geometry.centroid.x
                self.gdf['coord_y'] = self.gdf.geometry.centroid.y
                self._generar_puntos()
                self.estado_visual = np.zeros(len(self.all_x))
            except Exception as e:
                print(f"❌ Error Shapefile: {e}. Usando dummy.")
                self.dummy = True
                self.all_x = np.random.rand(100)
                self.all_y = np.random.rand(100)
                self.estado_visual = np.zeros(100)
                self.point_belongs_to = np.zeros(100)

    def _generar_puntos(self):
        print("🔨 Generando puntos...")
        xs, ys, owners = [], [], []
        self.gdf = self.gdf.reset_index(drop=True)
        
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            poblacion = row.get("poblacion", 1000)
            if poblacion <= 0 or geom is None: continue
            
            n = max(1, int(poblacion / self.dot_scale))
            n = min(n, 2000) 

            minx, miny, maxx, maxy = geom.bounds
            puntos = 0
            intentos = 0
            while puntos < n and intentos < n*10:
                rx = np.random.uniform(minx, maxx)
                ry = np.random.uniform(miny, maxy)
                if geom.contains(Point(rx, ry)):
                    xs.append(rx); ys.append(ry); owners.append(idx)
                    puntos += 1
                intentos += 1
        
        if len(xs) == 0:
            xs = self.gdf.geometry.centroid.x.values
            ys = self.gdf.geometry.centroid.y.values
            owners = self.gdf.index.values

        self.all_x = np.array(xs)
        self.all_y = np.array(ys)
        self.point_belongs_to = np.array(owners)

    def dibujar_curvas(self, historia):
        """
        Pinta la evolución temporal de S, I, R, D.
        historia: Lista de diccionarios [{'day':0, 'S':100...}, ...]
        """
        plt.close('all') # Limpieza de memoria
        
        if not historia: return None

        # Extraer datos
        dias = [h['day'] for h in historia]
        s = [h['S'] for h in historia]
        i = [h['I'] for h in historia]
        r = [h['R'] for h in historia]
        d = [h['D'] for h in historia]

        # Configurar Gráfico (Estilo Oscuro)
        fig, ax = plt.subplots(figsize=(10, 4)) # Más ancho que alto
        fondo = "#050505"
        fig.patch.set_facecolor(fondo)
        ax.set_facecolor(fondo)
        
        # Pintar Líneas
        ax.plot(dias, s, color='#00FF00', label='Sanos', linewidth=2, alpha=0.8)       # Verde
        ax.plot(dias, i, color='#FF0000', label='Infectados', linewidth=2, alpha=0.9)  # Rojo
        ax.plot(dias, r, color='#0088FF', label='Recuperados', linewidth=2, alpha=0.8) # Azul
        ax.plot(dias, d, color='#DDDDDD', label='Fallecidos', linewidth=2, alpha=0.8)  # Blanco/Gris

        # Estética
        ax.set_title("Evolución de la Pandemia", color='white', fontsize=12)
        ax.set_xlabel("Días", color='gray')
        ax.set_ylabel("Población", color='gray')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        ax.grid(color='#333333', linestyle='--', linewidth=0.5)
        
        # Leyenda
        leg = ax.legend(facecolor='#111111', edgecolor='#333333', labelcolor='white')
        
        plt.tight_layout()
        return fig

    def actualizar_colores(self, mapa_stats):
        self.estado_visual[:] = 0 # 0=Sano (Verde)
        
        for muni_id, stats in mapa_stats.items():
            ratio_inf = stats.get('I', 0.0)
            
            if ratio_inf <= 0 : continue
            
            # Obtener puntos de este pueblo
            indices = np.where(self.point_belongs_to == muni_id)[0]
            total_puntos = len(indices)
            if total_puntos == 0: continue
            
            # --- CORRECCIÓN: REDONDEO GENEROSO ---
            # Si hay infección matemática pero es poca (ej: 0.4 puntos), 
            # forzamos a que sea 1 punto visual el 50% de las veces o usamos ceil.
            # Aquí usamos una lógica simple: si ratio > 1% y n_inf daría 0, ponemos 1.
            
            n_inf = int(total_puntos * ratio_inf)
            if n_inf == 0 and ratio_inf > 0.01: 
                n_inf = 1 # Forzar visualización si hay >1% infectado
            
            # Protección para no superar el total de puntos
            if n_inf  > total_puntos:
                factor = total_puntos / (n_inf + n_dead)
                n_inf = int(n_inf * factor)
                n_dead = total_puntos - n_inf # El resto a muertos

            if n_inf == 0 : continue

            # Asignación aleatoria
            indices_barajados = np.random.permutation(indices)
            
            # 1. Pintar Rojos (Infectados)
            if n_inf > 0:
                idxs_rojos = indices_barajados[:n_inf]
                self.estado_visual[idxs_rojos] = 1

    def dibujar(self):
        plt.close('all') 
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fondo = "#050505"
        fig.patch.set_facecolor(fondo)
        ax.set_facecolor(fondo)

        if not self.dummy and hasattr(self, 'gdf') and HAS_GEOPANDAS:
            try: self.gdf.boundary.plot(ax=ax, color="#444444", linewidth=0.5, alpha=0.5)
            except: pass
        
        n = len(self.estado_visual)
        if n > 0:
            colors = np.zeros((n, 4)) 
            
            # 0: Verde (Sano)
            colors[self.estado_visual == 0] = [0.0, 1.0, 0.0, 0.4] # Más transparente
            
            # 1: Rojo (Infectado)
            colors[self.estado_visual == 1] = [1.0, 0.0, 0.0, 0.9] # Muy visible
            
            # 2: Azul (Muerto)
            colors[self.estado_visual == 2] = [0.9, 0.9, 0.9, 0.85] # 
            
            ax.scatter(self.all_x, self.all_y, c=colors, s=3, marker='.')
        else:
            ax.text(0.5, 0.5, "SIN DATOS", color="white", ha="center")

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        
        return fig