from mesa import Model
import yaml
from agent import Population
import torch
import pandas as pd
from tqdm import tqdm
class EpidemicModel(Model):
    def __init__(self, csv_path,config_path):
        super().__init__()
        with open(config_path,'r') as f:
            self.params = yaml.safe_load(f)

        self.shared_rng = torch.Generator(device=self.params['simulation']['device'])
        self.seed = self.params['simulation']['seed']
        self.shared_rng.manual_seed(self.seed)

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
        self.num_pueblos = len(self.poblaciones)
        self.mailbox = [[] for _ in range(self.num_pueblos)]
        
        print(f"✅ Modelo iniciado: {self.num_pueblos} nodos, Aleatoriedad pura.")

        print(f"   Población total simulada: {sum(p.N for p in self.poblaciones):,}")

    def _choose_destinations(self, num_leavers, origin_node_id):
        """
        CEREBRO: Decide los IDs de destino para un grupo de viajeros.
        Actualmente implementa: MOVILIDAD ALEATORIA UNIFORME.
        """
        # Generamos enteros aleatorios entre 0 y num_pueblos
        # low=0, high=self.num_pueblos
        destinations = torch.randint(
            low=0, 
            high=self.num_pueblos, 
            size=(num_leavers,), 
            dtype=torch.long
        )
        return destinations

    def _route_travelers(self, leavers, origin_node_id, target_mailbox):
        """
        MÚSCULO: Orquestra el movimiento de salida.
        1. Pregunta al cerebro a dónde van.
        2. Usa el helper para trocear los datos y meterlos en buzones.
        """
        if leavers is None:
            return 0

        # 1. Calculamos cuántos se van
        # Usamos la primera key disponible para ver el tamaño (ej: 'susceptibility')
        first_key = list(leavers.keys())[0]
        num_leavers = leavers[first_key].shape[0]

        if num_leavers == 0:
            return 0

        # 2. Elegimos destinos (Aquí es donde cambiarás la lógica en el futuro)
        destinations = self._choose_destinations(num_leavers, origin_node_id)

        # 3. Distribuimos físicamente los datos a los buzones
        self._distribute_to_mailboxes(leavers, destinations, target_mailbox)
        
        return num_leavers
    
    def _merge_travelers(self, list_of_dicts):
        if not list_of_dicts: return None
        merged = {}
        keys = list_of_dicts[0].keys()
        for key in keys:
            merged[key] = torch.cat([d[key] for d in list_of_dicts], dim=0)
        return merged

    def _distribute_to_mailboxes(self, leavers_dict, destinations, target_mailbox):
        unique_dests = torch.unique(destinations)
        for dest_id in unique_dests:
            dest_idx = int(dest_id.item())
            mask = (destinations == dest_id)
            
            package = {}
            for key, tensor in leavers_dict.items():
                package[key] = tensor[mask]
            
            target_mailbox[dest_idx].append(package)

    def step(self):
        next_day_mailbox = [[] for _ in range(self.num_pueblos)]
        total_travelers = 0

        for i, p in enumerate(self.poblaciones):
            
            # --- FASE 1: ENTRADA (Check-in) ---
            # Si hay gente en el buzón, la metemos. Si no, reseteamos contadores.
            incomers = self._merge_travelers(self.mailbox[i]) if self.mailbox[i] else None
            p.add_new_individuals(incomers)

            # --- FASE 2: SIMULACIÓN (GPU Work) ---
            p.to_gpu()
            
            # Semilla determinista
            self.shared_rng.manual_seed(self.seed + i)
            
            # Ejecutar lógica
            leavers = p.step()
            
            p.to_cpu()
            
            # --- FASE 3: SALIDA (Routing) ---
            # Delegamos toda la complejidad al nuevo método
            count = self._route_travelers(
                leavers=leavers, 
                origin_node_id=i,  # Pasamos 'i' por si el enrutamiento necesita saber el origen
                target_mailbox=next_day_mailbox
            )
            total_travelers += count

        # Rotación de buzones y avance del tiempo
        self.mailbox = next_day_mailbox
if __name__ == '__main__':
    archivo_csv = "poblacion_procesada.csv" 
    archivo_yaml = "params.yaml"

    # Iniciar modelo
    modelo = EpidemicModel(csv_path=archivo_csv, config_path=archivo_yaml)
    for dia in tqdm(range(100)):
        modelo.step()