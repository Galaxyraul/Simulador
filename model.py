import torch
import yaml
import pandas as pd
from tqdm import tqdm
from agent import Shard

class EpidemicModel:
    def __init__(self, df_data, config_path='config.yaml'):
        # 1. Cargar Configuración
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
            
        self.device = torch.device(self.params['simulation']['device'])
        self.step_count = 0
        self.history = []
        
        # Datos básicos del "Tablero"
        self.num_pueblos = len(df_data)
        
        # Cargamos la población a la GPU (float32 para cálculos)
        self.population_sizes = torch.tensor(
            df_data['poblacion'].values, 
            device=self.device, 
            dtype=torch.float32
        )
        
        self.valid_ids = torch.arange(self.num_pueblos, device=self.device)

        # -----------------------------------------------------------
        # 🚀 MOTOR DE GRAVEDAD: CÁLCULO DE VIAJES (BLINDADO)
        # -----------------------------------------------------------
        print("🧲 Calculando Matriz de Gravedad (Modo Seguro)...")
        
        # A. Extraer Coordenadas a GPU
        coords = torch.tensor(
            df_data[['coord_x', 'coord_y']].values, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # B. Matriz de Distancias
        dists = torch.cdist(coords, coords)
        
        # 1. Evitar división por cero en distancias (puntos superpuestos)
        dists = torch.clamp(dists, min=100.0) 
        
        # C. Fórmula de Gravedad
        alpha = 2.0 
        attraction = self.population_sizes.unsqueeze(0) # [1, N]
        
        # Cálculo de pesos
        weights = attraction / (dists.pow(alpha))
        
        # 2. Limpieza agresiva de basura matemática
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Anular viajes a uno mismo (Diagonal = 0)
        weights.fill_diagonal_(0.0)
        
        # 4. GESTIÓN DE FILAS MUERTAS (CRÍTICO PARA EVITAR CRASH)
        # Calculamos la suma de cada fila
        row_sums = weights.sum(dim=1, keepdim=True)
        
        # Buscamos pueblos que no atraen a nadie o no tienen a dónde ir (Suma ~ 0)
        # Esto pasa si el pueblo tiene población 0 o está aislado matemáticamente
        dead_rows = (row_sums < 1e-9).squeeze()
        
        if dead_rows.any():
            print(f"   ⚠️  Detectadas {dead_rows.sum().item()} filas con probabilidad 0. Corrigiendo...")
            # A estos pueblos les damos una probabilidad uniforme pequeña para evitar el error
            # Asignamos 1.0 a todo y luego la normalización lo dejará en 1/N
            weights[dead_rows] = 1.0
            # Volvemos a anular su diagonal para que no viajen a sí mismos
            weights.fill_diagonal_(0.0)
            # Recalculamos sumas
            row_sums = weights.sum(dim=1, keepdim=True)
        
        # 5. Normalización Final
        # Añadimos épsilon minúsculo por seguridad absoluta
        self.travel_probs = weights / (row_sums + 1e-9)
        
        # 6. VERIFICACIÓN FINAL (Para dormir tranquilos)
        # Validamos que ninguna fila tenga NaNs o sume 0
        if torch.isnan(self.travel_probs).any() or (self.travel_probs.sum(dim=1) < 1e-9).any():
            print("❌ ERROR CRÍTICO: La matriz sigue corrupta. Revisar datos de entrada.")
            # Fallback de emergencia: Matriz uniforme
            self.travel_probs = torch.ones_like(weights) / self.num_pueblos
        
        # Limpieza VRAM
        del dists, coords, weights, attraction, row_sums
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        print(f"   ✅ Matriz calculada. Shape: {self.travel_probs.shape}")

        # -----------------------------------------------------------
        # CREACIÓN DE SHARDS (Troceado de la población)
        # -----------------------------------------------------------
        self.shards = []
        self.routing_table = {}
        self.mailbox = [[] for _ in range(self.num_pueblos)]
        
        # Tamaño objetivo por bloque (ajustar según tu VRAM)
        target_shard_size = 10_000_000 
        
        current_shard_df = []
        current_pop = 0
        shard_id_counter = 0
        
        print("🏗️  Creando Shards de agentes...")
        
        # Iteramos sobre el DataFrame para agrupar pueblos en Shards
        # df_data es el GeoDataFrame, pero iterrows funciona igual
        for _, row in df_data.iterrows():
            current_shard_df.append(row)
            current_pop += row['poblacion']
            
            if current_pop >= target_shard_size:
                df_chunk = pd.DataFrame(current_shard_df)
                new_shard = Shard(df_chunk, shard_id_counter, self.params)
                self.shards.append(new_shard)
                
                # Mapeamos IDs globales a este Shard
                for gid in new_shard.global_ids.tolist():
                    self.routing_table[gid] = shard_id_counter
                
                current_shard_df = []
                current_pop = 0
                shard_id_counter += 1
        
        # Último shard (remanente)
        if current_shard_df:
            df_chunk = pd.DataFrame(current_shard_df)
            new_shard = Shard(df_chunk, shard_id_counter, self.params)
            self.shards.append(new_shard)
            for gid in new_shard.global_ids.tolist():
                self.routing_table[gid] = shard_id_counter

    def step(self):
        # 1. Gestión de Políticas (Lockdown, Vacunas, Mascarillas)
        self._manage_interventions()
        
        next_mailbox = [[] for _ in range(self.num_pueblos)]
        daily_stats = {'day': self.step_count, 'S':0, 'I':0, 'R':0, 'D':0, 'Moves': 0}
        
        # Iteramos sobre los trozos de población
        pbar = tqdm(self.shards, desc=f"Día {self.step_count}", leave=False, unit="shard")
        for shard in pbar:
            # --- RECEPCIÓN DE VIAJEROS ---
            shard_incomers = []
            shard_dest_ids = []
            
            local_pueblos = shard.global_ids.tolist()
            
            for local_idx, global_id in enumerate(local_pueblos):
                mail = self.mailbox[global_id]
                if mail:
                    merged = self._merge_travelers(mail)
                    if merged:
                        shard_incomers.append(merged)
                        n = merged['state'].shape[0]
                        shard_dest_ids.append(torch.full((n,), local_idx, dtype=torch.long))
            
            if shard_incomers:
                final_incomers = self._merge_travelers(shard_incomers)
                final_dest_ids = torch.cat(shard_dest_ids)
                shard.add_incomers(final_incomers, final_dest_ids)

            # --- SIMULACIÓN FÍSICA ---
            shard._to_gpu()
            
            # Semilla determinista variable por día
            seed = shard.base_seed + (self.step_count * 777)
            rng = torch.Generator(device=self.device).manual_seed(seed)
            
            leavers = shard.step(rng)
            
            stats = shard.get_summary()
            for k in ['S','I','R','D']: daily_stats[k] += stats[k]
            
            shard._to_cpu()
            
            # --- ENRUTAMIENTO POR GRAVEDAD ---
            if leavers:
                # Los que se van están en CPU. Necesitamos sus IDs en GPU para consultar la matriz.
                origins_cpu = leavers['origin_global_id']
                origins_gpu = origins_cpu.to(self.device)
                
                num_travelers = origins_gpu.shape[0]
                daily_stats['Moves'] += num_travelers
                
                # Consultamos la matriz de viajes para estos orígenes
                # batch_probs tendrá tamaño [Num_Viajeros, Total_Pueblos]
                batch_probs = self.travel_probs[origins_gpu]
                
                # Sorteo Ponderado: ¿A dónde va cada uno?
                dest_indices = torch.multinomial(batch_probs, num_samples=1).squeeze()
                
                # Volvemos a CPU para repartir el correo
                destinations_cpu = dest_indices.cpu()
                
                self._distribute_to_mailboxes(leavers, destinations_cpu, next_mailbox)

        self.mailbox = next_mailbox
        self.history.append(daily_stats)
        self.step_count += 1
        
        return daily_stats

    # --- MÉTODOS AUXILIARES ---
    def _manage_interventions(self):
        day = self.step_count
        # Ejemplo: Lockdown el día 20
        if day == 20:
             print("\n🚨 ALERTA: Confinamiento decretado.")
             self.params['population']['lockdown_factor'] = 0.7

    def _distribute_to_mailboxes(self, leavers, destinations, mailbox):
        for i in range(len(destinations)):
            dest_id = destinations[i].item()
            # Empaquetamos al viajero individualmente
            traveler = {k: v[i].unsqueeze(0) for k, v in leavers.items() if k != 'origin_global_id'}
            mailbox[dest_id].append(traveler)

    def _merge_travelers(self, traveler_list):
        if not traveler_list: return None
        keys = traveler_list[0].keys()
        merged = {}
        for k in keys:
            merged[k] = torch.cat([t[k] for t in traveler_list])
        return merged
        
    def export_results(self, filename):
        pd.DataFrame(self.history).to_csv(filename, index=False)