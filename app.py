import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from visualizer import VisualizadorMapa
from model import EpidemicModel
from utils import (
    cargar_yaml_config, actualizar_parametros, actualizar_subparametros,
    actualizar_editor, guardar_valor, 
    agregar_parametro_nivel_2, eliminar_parametro_nivel_2 # <--- Nuevas funciones
)

# Configuración
ARCHIVO_MAPA = "municipios_procesados/mapa.shp"
DOT_SCALE = 5000

print("⏳ [APP] Inicializando...")
try:
    VIS = VisualizadorMapa(ARCHIVO_MAPA, DOT_SCALE)
except:
    VIS = None

def dibujar_mapa():
    return VIS.dibujar() if VIS else None

def bucle_simulacion(estado_config):
    # (El código del bucle se mantiene idéntico al que te funcionaba antes)
    if VIS is None: yield None, "Error visual", "Error"; return
    if not estado_config: yield None, "Falta YAML", "Error"; return

    try:
        n_steps = int(estado_config.get('simulation', {}).get('n_steps', 50))
    except: n_steps = 50

    yield None, "⚙️ Calculando...", "Cargando..."

    try:
        modelo = EpidemicModel(VIS.gdf, estado_config)
    except Exception as e:
        yield None, f"Error: {e}", "Error"; return

    for i in range(n_steps):
        stats = modelo.step()
        ratios = modelo.obtener_estado_visual()
        VIS.actualizar_colores(ratios)
        fig = VIS.dibujar()
        yield fig, "🟢 Simulando...", f"Día {stats['day']}/{n_steps} | I: {stats['I']}"
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    yield fig, "✅ Fin", f"Día {stats['day']} completado"

def placeholder(): return "..."

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks(title="Simulador Epidemias") as demo:
    estado = gr.State({})

    gr.Markdown("# 🦠 Simulador: Configuración & Ejecución")
    
    with gr.Row():
        with gr.Column(scale=1):
            archivo = gr.File(label="📂 Configuración YAML")
            
            seccion = gr.Dropdown(label="1. Sección")
            # Este es el dropdown que se actualizará al añadir/borrar
            parametro = gr.Dropdown(label="2. Parámetro / Variante", allow_custom_value=True)
            subparametro = gr.Dropdown(label="3. Propiedad", visible=False, allow_custom_value=True)
            
            valor = gr.Number(label="Valor")
            
            gr.HTML("<hr>")

            # Panel de Variantes
            with gr.Group(visible=False) as panel_variantes:
                gr.Markdown("### 🧬 Gestión de Variantes (Nivel Parámetro)")
                with gr.Row():
                    nuevo_nombre = gr.Textbox(placeholder="Nombre (ej: Omicron)", show_label=False, container=False)
                with gr.Row():
                    btn_add = gr.Button("➕ Nueva Variante", size="sm")
                    btn_del = gr.Button("🗑️ Borrar Actual", variant="stop", size="sm")
            
            gr.HTML("<hr>")
            
            btn_run = gr.Button("▶ EJECUTAR", variant="primary")
            with gr.Row():
                btn_stop = gr.Button("⏹ DETENER", variant="stop")
                btn_export = gr.Button("💾 EXPORTAR")

        with gr.Column(scale=3):
            plot = gr.Plot(label="Mapa")
            info = gr.Textbox(label="Info", value="Listo.")
            stats = gr.Textbox(label="Métricas")

    # ─────────────────────────────────────────────
    # EVENTOS
    # ─────────────────────────────────────────────
    
    demo.load(dibujar_mapa, None, plot)
    
    archivo.change(cargar_yaml_config, archivo, [estado, seccion, parametro, valor, info])
    seccion.change(actualizar_parametros, [seccion, estado], [parametro, valor])
    
    parametro.change(
        actualizar_subparametros, 
        [seccion, parametro, estado], 
        [subparametro, valor, panel_variantes]
    )
    
    subparametro.change(actualizar_editor, [seccion, parametro, subparametro, estado], valor)
    valor.change(guardar_valor, [seccion, parametro, subparametro, valor, estado], estado)

    # 🔥 EVENTOS CORREGIDOS 🔥
    # Al añadir/borrar, actualizamos 'parametro' (el dropdown de nivel 2), NO 'subparametro'
    btn_add.click(
        agregar_parametro_nivel_2, 
        inputs=[seccion, parametro, nuevo_nombre, estado], 
        outputs=[estado, parametro, info] # <-- Actualiza el dropdown principal
    )
    
    btn_del.click(
        eliminar_parametro_nivel_2, 
        inputs=[seccion, parametro, estado], 
        outputs=[estado, parametro, info] # <-- Actualiza el dropdown principal
    )

    evento_run = btn_run.click(bucle_simulacion, inputs=[estado], outputs=[plot, info, stats])
    btn_stop.click(fn=None, inputs=None, outputs=None, cancels=[evento_run])
    btn_export.click(placeholder, None, info)

if __name__ == "__main__":
    demo.queue().launch()