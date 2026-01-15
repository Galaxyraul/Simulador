import os

# HIPERPARAMETROS
DOT_SCALE = 5000
PLOT_FREQUENCY = 5 

# RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MUNICIPIOS_PATH = os.path.join(ASSETS_DIR,"media/municipios")
YAML_PATH = os.path.join(ASSETS_DIR, "params.yaml")