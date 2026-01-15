import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import config

from src.visualizer import VisualizadorMapa
from src.model import EpidemicModel
from src.utils import (
    cargar_yaml_config, actualizar_parametros, actualizar_subparametros,
    actualizar_editor, guardar_valor, 
    agregar_parametro_nivel_2, eliminar_parametro_nivel_2
)

print("⏳ [APP] Inicializando...")
try:
    VIS = VisualizadorMapa(config.MUNICIPIOS_PATH, config.DOT_SCALE)
except:
    VIS = None

def dibujar_mapa():
    return VIS.dibujar() if VIS else None

# ─────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────
def bucle_simulacion(estado_config):
    if VIS is None: 
        yield None, None, "❌ Error visual", "Error"
        return
    if not estado_config: 
        yield VIS.dibujar(), None, "⚠️ Carga YAML", "Esperando..."
        return

    try:
        n_steps = int(estado_config.get('simulation', {}).get('steps', 200))
    except: n_steps = 200

    print("🚀 [APP] Iniciando...")
    
    # Estado inicial visual
    fig_mapa = VIS.dibujar()
    yield fig_mapa, None, "⚙️ Calculando...", "Cargando..."

    try:
        modelo = EpidemicModel(VIS.gdf, estado_config)
    except Exception as e:
        yield fig_mapa, None, f"Error: {e}", "Error"
        return

    # Historial para las gráficas
    historia = []
    
    for i in tqdm(range(n_steps), desc="🦠 Simulando", unit="step"):
        stats = modelo.step()
        historia.append(stats) # Guardamos datos
        
        # Cálculos para el Panel de Texto (Stats Globales)
        total_pop = stats['S'] + stats['I'] + stats['R'] + stats['D']
        if total_pop == 0: total_pop = 1
        
        pct_s = (stats['S'] / total_pop) * 100
        pct_i = (stats['I'] / total_pop) * 100
        pct_r = (stats['R'] / total_pop) * 100
        pct_d = (stats['D'] / total_pop) * 100
        
        texto_stats = (
            f"📅 DÍA {stats['day']}/{n_steps}\n"
            f"────────────────\n"
            f"🟢 Sanos:      {stats['S']:,.0f} ({pct_s:.1f}%)\n"
            f"🔴 Infectados: {stats['I']:,.0f} ({pct_i:.1f}%)\n"
            f"🔵 Recuperados:{stats['R']:,.0f} ({pct_r:.1f}%)\n"
            f"⚪ Fallecidos:  {stats['D']:,.0f} ({pct_d:.1f}%)\n"
            f"────────────────\n"
            f"🚗 Viajes hoy: {stats['Moves']:,.0f}"
        )

        # Renderizado (Mapa + Curvas)
        if i % config.PLOT_FREQUENCY == 0 or i == n_steps - 1:
            # 1. Mapa
            ratios = modelo.obtener_estado_visual()
            VIS.actualizar_colores(ratios)
            fig_mapa = VIS.dibujar()
            
            # 2. Curvas SIR
            fig_curvas = VIS.dibujar_curvas(historia)
            
            yield fig_mapa, fig_curvas, "🟢 Simulando...", texto_stats
        else:
            # Solo actualizamos texto en pasos intermedios (más rápido)
            yield gr.update(), gr.update(), "🟢 Simulando (Turbo)...", texto_stats
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    ratios = modelo.obtener_estado_visual()
    VIS.actualizar_colores(ratios)
    fig_mapa = VIS.dibujar()
    
    # 2. Curvas SIR
    yield fig_mapa, VIS.dibujar_curvas(historia), "✅ Finalizado", texto_stats

def placeholder(): return "..."

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks(title="Simulador Epidemias") as demo:
    estado = gr.State({})

    gr.Markdown("# 🦠 Simulador: Dashboard Global")
    
    with gr.Row():
        # COLUMNA IZQUIERDA (Configuración)
        with gr.Column(scale=1):
            archivo = gr.File(label="📂 Configuración YAML",value=config.YAML_PATH)
            
            seccion = gr.Dropdown(label="Sección")
            parametro = gr.Dropdown(label="Parámetro", allow_custom_value=True)
            subparametro = gr.Dropdown(label="Propiedad", visible=False, allow_custom_value=True)
            valor = gr.Number(label="Valor")
            
            gr.HTML("<hr>")
            
            with gr.Group(visible=False) as panel_variantes:
                gr.Markdown("### 🧬 Variantes")
                with gr.Row():
                    nuevo_nombre = gr.Textbox(placeholder="Nombre", show_label=False, container=False)
                with gr.Row():
                    btn_add = gr.Button("➕", size="sm")
                    btn_del = gr.Button("🗑️", variant="stop", size="sm")
            
            gr.HTML("<hr>")
            
            # PANEL DE ESTADÍSTICAS EN VIVO (Movido aquí para visibilidad)
            stats_box = gr.Textbox(label="📊 Estadísticas Globales", lines=8, value="Esperando datos...")

            gr.HTML("<hr>")
            btn_run = gr.Button("▶ EJECUTAR", variant="primary")
            btn_stop = gr.Button("⏹ DETENER", variant="stop")

        # COLUMNA DERECHA (Visualización Doble)
        with gr.Column(scale=3):
            # 1. Mapa Geográfico
            plot_mapa = gr.Plot(label="Mapa de Propagación")
            
            # 2. Gráfico de Curvas
            plot_curvas = gr.Plot(label="Curvas SIR (Evolución Temporal)")
            
            info = gr.Textbox(label="Estado del Sistema", value="Listo.")

    # ─────────────────────────────────────────────
    # EVENTOS
    # ─────────────────────────────────────────────
    
    demo.load(dibujar_mapa, None, plot_mapa)
    
    archivo.change(cargar_yaml_config, archivo, [estado, seccion, parametro, valor, info])
    seccion.change(actualizar_parametros, [seccion, estado], [parametro, valor])
    
    parametro.change(
        actualizar_subparametros, 
        [seccion, parametro, estado], 
        [subparametro, valor, panel_variantes]
    )
    
    subparametro.change(actualizar_editor, [seccion, parametro, subparametro, estado], valor)
    valor.change(guardar_valor, [seccion, parametro, subparametro, valor, estado], estado)

    btn_add.click(agregar_parametro_nivel_2, [seccion, parametro, nuevo_nombre, estado], [estado, parametro, info])
    btn_del.click(eliminar_parametro_nivel_2, [seccion, parametro, estado], [estado, parametro, info])

    # Ejecución conecta con DOS plots y UN textbox de stats
    evento_run = btn_run.click(
        bucle_simulacion, 
        inputs=[estado], 
        outputs=[plot_mapa, plot_curvas, info, stats_box]
    )
    
    btn_stop.click(fn=None, inputs=None, outputs=None, cancels=[evento_run])

if __name__ == "__main__":
    demo.queue().launch()