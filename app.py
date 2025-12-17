import gradio as gr
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
ARCHIVO_MAPA = "municipios_procesados/mapa.shp"
DOT_SCALE = 5000  # 1 punto = X habitantes

PARAMETROS = {
    "velocidad": {"min": 0.1, "max": 5.0, "default": 1.0},
    "pasos": {"min": 10, "max": 1000, "default": 100},
    "ruido": {"min": 0.0, "max": 1.0, "default": 0.1},
}

# ─────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────
def iniciar_simulacion(estado):
    txt = "🧪 Simulación iniciada con:\n\n"
    for k, v in estado.items():
        txt += f"- {k}: {v}\n"
    return txt, "Simulación en curso…", f"{len(estado)} parámetros"

def pausar_simulacion():
    return "⏸ Simulación pausada"

def visualizar_resultados():
    return "👁 Visualizando resultados"

def exportar_resultados():
    return "💾 Resultados exportados"
    
def inicializar_estado():
    return {k: v["default"] for k, v in PARAMETROS.items()}

def actualizar_editor(param):
    meta = PARAMETROS[param]
    return (
        gr.update(minimum=meta["min"], maximum=meta["max"], value=meta["default"]),
        gr.update(value=meta["default"])
    )

def guardar_valor(param, valor, estado):
    estado[param] = valor
    return estado

# ─────────────────────────────────────────────
# VISUALIZADOR
# ─────────────────────────────────────────────
class VisualizadorMapa:
    def __init__(self, ruta_shp):
        if not os.path.exists(ruta_shp):
            raise FileNotFoundError(ruta_shp)

        print("🌍 Cargando mapa...")
        self.gdf = gpd.read_file(ruta_shp)

        self.all_x = []
        self.all_y = []
        self.estado = None  # 0 sano, 1 infectado

        self._generar_puntos()
        self._infectar_inicial(300)

    def _generar_puntos(self):
        xs, ys = [], []

        for _, row in self.gdf.iterrows():
            geom = row.geometry
            poblacion = row["poblacion"]

            if poblacion <= 0 or geom is None:
                continue

            n = max(1, int(poblacion / DOT_SCALE))
            minx, miny, maxx, maxy = geom.bounds

            puntos = 0
            while puntos < n:
                rx = np.random.uniform(minx, maxx)
                ry = np.random.uniform(miny, maxy)
                if geom.contains(Point(rx, ry)):
                    xs.append(rx)
                    ys.append(ry)
                    puntos += 1

        self.all_x = np.array(xs)
        self.all_y = np.array(ys)
        self.estado = np.zeros(len(xs), dtype=int)

        print(f"✅ Puntos generados: {len(xs)}")

    def _infectar_inicial(self, n):
        idx = np.random.choice(len(self.estado), n, replace=False)
        self.estado[idx] = 1

    def paso_simulacion(self, prob=0.02):
        sanos = np.where(self.estado == 0)[0]
        infectados = np.where(self.estado == 1)[0]

        if len(sanos) == 0:
            return

        nuevos = np.random.choice(
            sanos,
            size=min(len(sanos), int(len(infectados) * prob)),
            replace=False
        )
        self.estado[nuevos] = 1

    def dibujar(self):
        fig, ax = plt.subplots(figsize=(18, 14))

        fondo = "#050505"
        fig.patch.set_facecolor(fondo)
        ax.set_facecolor(fondo)

        self.gdf.boundary.plot(
            ax=ax, color="#333333", linewidth=0.5, alpha=0.5
        )

        colores = np.zeros((len(self.estado), 3))
        colores[self.estado == 0] = [0, 1, 0]  # sano
        colores[self.estado == 1] = [1, 0, 0]  # infectado

        ax.scatter(
            self.all_x,
            self.all_y,
            c=colores,
            s=1.5,
            alpha=0.6,
            marker="."
        )

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Densidad de Población (1 punto = {DOT_SCALE} habs)",
            color="white",
            fontsize=14
        )

        plt.tight_layout(pad=0)
        plt.close(fig)
        return fig

# ─────────────────────────────────────────────
# INSTANCIA GLOBAL
# ─────────────────────────────────────────────
try:
    VISUALIZADOR = VisualizadorMapa(ARCHIVO_MAPA)
except Exception as e:
    print("❌ Error:", e)
    VISUALIZADOR = None

def cargar_mapa():
    if VISUALIZADOR:
        return VISUALIZADOR.dibujar()
    return None

def avanzar():
    VISUALIZADOR.paso_simulacion()
    return VISUALIZADOR.dibujar()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks(
    title="Simulador de Epidemias",
    css="""
    footer {display: none;}

    #mapa_plot {
        height: calc(100vh - 220px);
    }

    #mapa_plot > div {
        height: 100% !important;
    }

    #mapa_plot canvas {
        width: 100% !important;
        height: 100% !important;
    }

    
    """
) as demo:

    estado = gr.State(inicializar_estado())

    gr.Markdown("""
    # 🦠 Simulador de Epidemias
    **Entornos virtuales y simulación**
    ---
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.Markdown("### 📂 Configuración")
            gr.File(label="Archivo de configuración")

            gr.Markdown("### ⚙️ Parámetros")
            parametro = gr.Dropdown(
                choices=list(PARAMETROS.keys()),
                value="velocidad"
            )
            slider = gr.Slider(0, 1, value=0)
            valor = gr.Number()

            gr.Markdown("### ▶ Control")
            iniciar = gr.Button("🚀 Iniciar", variant="primary")
            paso = gr.Button("⏭ Avanzar paso")
            pausar = gr.Button("⏸ Pausar")
            visualizar = gr.Button("👁 Visualizar resultados")
            exportar = gr.Button("💾 Exportar resultados")

        with gr.Column(scale=4):
            mapa_plot = gr.Plot(
                label="Distribución Real de la Población",
                elem_id="mapa_plot"
            )

            with gr.Row():
                progreso = gr.Textbox(label="⏳ Progreso", lines=3)
                resumen = gr.Textbox(label="📊 Resumen", lines=3)

    demo.load(cargar_mapa, None, mapa_plot)

    parametro.change(actualizar_editor, parametro, [slider, valor])
    slider.change(lambda v: v, slider, valor)
    valor.change(lambda v: v, valor, slider)
    slider.change(guardar_valor, [parametro, slider, estado], estado)

    paso.click(avanzar, None, mapa_plot,show_progress=False)
    iniciar.click(iniciar_simulacion, estado, [progreso, resumen])
    pausar.click(pausar_simulacion, None, progreso)
    exportar.click(exportar_resultados, None, progreso)

demo.launch()
