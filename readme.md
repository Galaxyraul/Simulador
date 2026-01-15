
# 🦠 Simulador Epidémico Geoespacial (SIRD + Movilidad)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)
![Status](https://img.shields.io/badge/Status-Beta-green)

Simulador estocástico de propagación de epidemias de alto rendimiento. Combina el modelo epidemiológico **SIRD** (Susceptible, Infectado, Recuperado, Fallecido) con un **Modelo de Gravedad** para la movilidad entre municipios, todo acelerado por GPU.

El proyecto cuenta con una interfaz web interactiva (**Gradio**) que permite visualizar la evolución en tiempo real sobre mapas geográficos y ajustar parámetros dinámicamente sin detener el servidor.

---

## 🚀 Características Principales

* **⚡ Motor de Alto Rendimiento:** Simulación basada en tensores (**PyTorch**) optimizada para **CUDA**. Capaz de manejar millones de agentes divididos en "Shards" para eficiencia de memoria.
* **🌍 Movilidad y Geografía:** Los agentes se mueven entre municipios basándose en la atracción gravitacional (población/distancia). Soporta archivos Shapefile (`.shp`) reales o genera mapas sintéticos ("Dummy Mode").
* **🧬 Sistema Multi-Variante:** Gestión dinámica de variantes virales. Puedes introducir nuevas cepas en días específicos, cada una con su propia tasa de contagio (`P_base`), letalidad y resistencia.
* **📊 Dashboard Interactivo:**
    * **Mapa de Calor:** Visualización de infectados/fallecidos por municipio.
    * **Curvas SIRD:** Gráficas de evolución temporal.
    * **Edición en Vivo:** Modifica el `yaml` de configuración directamente desde la UI.
* **📉 Factores Sociales:** Modelado granular de uso de mascarillas, confinamientos (`lockdown`), cumplimiento de normas (`noncompliance`) y pérdida de inmunidad.

---

## 🛠️ Instalación

### Requisitos Previos
* Python 3.8 o superior.
* NVIDIA GPU (Recomendado para simulaciones masivas).

### Dependencias
Instala las librerías necesarias:

```bash
pip install torch pandas matplotlib tqdm pyyaml gradio shapely geopandas

```

> **Nota:** `geopandas` es opcional. Si no se instala, el visualizador funcionará en modo abstracto (puntos aleatorios).

---

## ▶️ Uso

1. **Clonar el repositorio:**
```bash
git clone https://github.com/Galaxyraul/Simulador.git
cd simulador-epidemias

```


2. **Iniciar la aplicación:**
```bash
python app.py
```


3. **Acceder al Dashboard:**
Abre tu navegador en la dirección local mostrada (usualmente `http://127.0.0.1:7860`).
4. **Ejecutar:**
* Verifica los parámetros en el panel izquierdo.
* Pulsa **▶ EJECUTAR**.



---

## ⚙️ Configuración (`params.yaml`)

El corazón de la simulación es el archivo `assets/params.yaml`. Controla desde la física de la infección hasta la demografía.

```yaml
simulation:
  steps: 200            # Duración en días
  device: "cuda"        # "cuda" para GPU, "cpu" para procesador
  initial infection rate: 0.005

population:
  contacts_per_day: 30  # Media de contactos diarios
  mask_factor: 0.5      # Eficacia de mascarillas (0.5 = 50%)
  lockdown_factor: 0.4  # Reducción de movilidad en cuarentena

variants:
  original_strain:      # Cepa base
    P_base: 0.15        # Probabilidad de infección por contacto
    recovery_day: 14    # Días para recuperación
    death_prob: 0.02    # Tasa de letalidad

```

---

## 📂 Estructura del Proyecto

```text
.
├── app.py              # Entry point. Interfaz UI (Gradio).
├── config.py           # Constantes y rutas globales.
├── src/
│   ├── agent.py        # Lógica de agentes (Shard) y motor SIRD.
│   ├── model.py        # Orquestador: Gestión de GPU, Shards y Viajes.
│   ├── visualizer.py   # Renderizado de mapas y gráficas.
│   └── utils.py        # Helpers para gestión de YAML y UI callbacks.
└── assets/
    ├── params.yaml     # Configuración por defecto.
    └── media/
        └── municipios/ # Carpeta para Shapefiles (.shp, .shx, .dbf)

```

---

## 🧠 Detalles Técnicos

### Arquitectura de Shards

Para escalar a poblaciones grandes (ej. una comunidad autónoma o país entero), el modelo divide la población en **Shards**. Cada Shard es un contenedor de datos independiente que puede moverse entre CPU y GPU según sea necesario, permitiendo simular poblaciones que exceden la memoria VRAM de una sola tarjeta gráfica.

### Modelo de Infección

La probabilidad de infección se calcula vectorizadamente:
$$P(inf) = 1 - e^{-\lambda}$$

Donde  es función de:

1. **Carga Viral Local:** Suma ponderada de infectados en el municipio.
2. **Movilidad:** Visitantes infectados de otros municipios (Gravedad).
3. **Susceptibilidad Individual:** Atributo único de cada agente.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](https://www.google.com/search?q=LICENSE) para más detalles.
