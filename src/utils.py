import gradio as gr
import yaml
import copy

# ─────────────────────────────────────────────
# FILTROS DE CONVERSIÓN
# ─────────────────────────────────────────────
def aplicar_filtro_lectura(seccion, parametro, valor):
    if seccion == "simulation" and parametro == "device":
        if valor == "cuda": return 1
        if valor == "cpu": return 0
    return valor

def aplicar_filtro_escritura(seccion, parametro, valor_numerico):
    if seccion == "simulation" and parametro == "device":
        if valor_numerico == 1: return "cuda"
        if valor_numerico == 0: return "cpu"
    return valor_numerico

# ─────────────────────────────────────────────
# CARGA Y NAVEGACIÓN
# ─────────────────────────────────────────────
def cargar_yaml_config(archivo):
    if archivo is None: return {}, [], [], None, "Vacío"
    with open(archivo, "r") as f: data = yaml.safe_load(f)
    if not data: return {}, [], [], None, "Archivo vacío"

    sec_init = list(data.keys())[0]
    params = list(data[sec_init].keys())
    param_init = params[0] if params else None
    
    valor_num = None
    if param_init:
        val_raw = data[sec_init][param_init]
        if isinstance(val_raw, dict) and val_raw:
            val_hijo = list(val_raw.values())[0]
            if isinstance(val_hijo, (int, float)): valor_num = val_hijo
        elif isinstance(val_raw, (int, float, str)):
            valor_num = aplicar_filtro_lectura(sec_init, param_init, val_raw)
            if not isinstance(valor_num, (int, float)): valor_num = None

    return (
        data,
        gr.update(choices=list(data.keys()), value=sec_init),
        gr.update(choices=params, value=param_init),
        valor_num,
        "✅ YAML cargado"
    )

def actualizar_parametros(seccion, estado):
    if not seccion or seccion not in estado:
        return gr.update(choices=[], value=None), None

    params = list(estado.get(seccion, {}).keys())
    if not params: return gr.update(choices=[], value=None), None

    primer = params[0]
    valor = estado[seccion][primer]
    
    valor_filtrado = aplicar_filtro_lectura(seccion, primer, valor)
    es_numero = isinstance(valor_filtrado, (int, float))

    return (
        gr.update(choices=params, value=primer),
        valor_filtrado if es_numero else None
    )

def actualizar_subparametros(seccion, parametro, estado):
    if not seccion or not parametro or seccion not in estado:
        return gr.update(visible=False), None, gr.update(visible=False)

    if parametro not in estado[seccion]:
        return gr.update(visible=False), None, gr.update(visible=False)

    valor = estado[seccion][parametro]

    # SI EL PARÁMETRO ES UN DICCIONARIO (EJ: variante alpha), MOSTRAMOS EL PANEL
    if isinstance(valor, dict):
        subparams = list(valor.keys())
        primero = subparams[0] if subparams else None
        
        val_primero = valor[primero] if primero else None
        es_num = isinstance(val_primero, (int, float))

        return (
            gr.update(choices=subparams, value=primero, visible=True),
            val_primero if es_num else None,
            gr.update(visible=True) # Mostrar botones de gestión
        )
    
    else:
        val_filtrado = aplicar_filtro_lectura(seccion, parametro, valor)
        es_num = isinstance(val_filtrado, (int, float))
        return (
            gr.update(visible=False), 
            val_filtrado if es_num else None,
            gr.update(visible=False) # Ocultar botones
        )

def actualizar_editor(seccion, parametro, subparametro, estado):
    if not seccion or not parametro or seccion not in estado: return None
    if parametro not in estado[seccion]: return None

    valor = estado[seccion][parametro]
    
    if isinstance(valor, dict):
        if subparametro and subparametro in valor:
            res = valor[subparametro]
            return res if isinstance(res, (int, float)) else None
    else:
        val = aplicar_filtro_lectura(seccion, parametro, valor)
        return val if isinstance(val, (int, float)) else None
    return None

def guardar_valor(seccion, parametro, subparametro, num, estado):
    if not seccion or not parametro or seccion not in estado: return estado
    if parametro not in estado[seccion]: return estado

    container = estado[seccion][parametro]

    if isinstance(container, dict):
        if subparametro and subparametro in container:
            if not isinstance(container[subparametro], dict):
                container[subparametro] = num
    else:
        estado[seccion][parametro] = aplicar_filtro_escritura(seccion, parametro, num)
    return estado

# ─────────────────────────────────────────────
# 🔥 GESTIÓN DE VARIANTES (CORREGIDA - NIVEL 2) 🔥
# ─────────────────────────────────────────────

def agregar_parametro_nivel_2(seccion, parametro_actual, nombre_nuevo, estado):
    """
    Crea un NUEVO PARÁMETRO (hermano del actual) en la sección.
    Usa 'parametro_actual' como plantilla para clonar la estructura.
    """
    if not nombre_nuevo:
        return estado, gr.update(), "⚠️ Escribe nombre"
    
    # Objetivo: La sección entera (ej: epidemiology)
    contenedor_seccion = estado[seccion]

    if nombre_nuevo in contenedor_seccion:
        return estado, gr.update(), "⚠️ Ya existe"

    # CLONACIÓN: Usamos el parámetro actual como molde
    if parametro_actual and parametro_actual in contenedor_seccion:
        plantilla = contenedor_seccion[parametro_actual]
        contenedor_seccion[nombre_nuevo] = copy.deepcopy(plantilla)
    else:
        # Si no hay nada seleccionado, creamos estructura vacía
        contenedor_seccion[nombre_nuevo] = {"default": 0.0}

    nuevas_opciones = list(contenedor_seccion.keys())
    
    # Devolvemos actualización para el dropdown de PARÁMETROS (Nivel 2)
    return (
        estado,
        gr.update(choices=nuevas_opciones, value=nombre_nuevo), # Seleccionamos el nuevo
        f"✅ Variante '{nombre_nuevo}' creada"
    )

def eliminar_parametro_nivel_2(seccion, parametro_actual, estado):
    """
    Elimina el PARÁMETRO seleccionado de la sección.
    """
    if not parametro_actual:
        return estado, gr.update(), "⚠️ Nada seleccionado"

    contenedor_seccion = estado[seccion]
    
    if parametro_actual in contenedor_seccion:
        del contenedor_seccion[parametro_actual]
    
    nuevas_opciones = list(contenedor_seccion.keys())
    nuevo_val = nuevas_opciones[0] if nuevas_opciones else None

    # Devolvemos actualización para el dropdown de PARÁMETROS (Nivel 2)
    return (
        estado,
        gr.update(choices=nuevas_opciones, value=nuevo_val),
        f"🗑️ '{parametro_actual}' eliminada"
    )