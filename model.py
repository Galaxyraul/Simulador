from mesa import Model
import yaml
from agent import Population
from torch import Generator
import pandas as pd
class EpidemicModel(Model):
    def __init__(self, csv_path,config_path):
        super().__init__()
        with open(config_path,'r') as f:
            self.params = yaml.safe_load(f)

        self.shared_rng = Generator(device=self.params['simulation']['device'])

        self.variants_schedule = self.params['virus']['variantes']
        self.variants_schedule.sort(key=lambda x: x['start_step'])
        
        self.current_variant_factor = 1.0
        self.current_variant_name = "Original"

        print(f"Cargando censo desde {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No encuentro el archivo: {csv_path}")

        self.poblaciones = []
        col_nombre = 'id_municipio' # O 'NOMBRE'
        col_pob = 'poblacion'       # O 'POB25'
        for idx, row in df.iterrows():
            nombre = row[col_nombre]
            poblacion = int(row[col_pob])

            # Saltamos pueblos vacíos o con errores para no romper PyTorch
            if poblacion <= 0:
                continue

            # Instanciamos el agente
            # idx: Es el número de fila (0, 1, 2...) -> Sirve para la semilla
            # nombre: Es el texto ("Madrid") -> Sirve para logs
            nuevo_pueblo = Population(
                model=self,
                N=poblacion, 
                rng=self.shared_rng,
                variant_factor=self.current_variant_factor,
                name=nombre, 
                params=self.params
            )
            
            self.poblaciones.append(nuevo_pueblo)

        print(f"✅ Modelo inicializado con {len(self.poblaciones)} pueblos.")
        print(f"   Población total simulada: {sum(p.N for p in self.poblaciones):,}")

if __name__ == '__main__':
    archivo_csv = "poblacion_procesada.csv" 
    archivo_yaml = "params.yaml"

    # Iniciar modelo
    modelo = EpidemicModel(csv_path=archivo_csv, config_path=archivo_yaml)