import pandas as pd
import torch
import yaml
from tqdm import tqdm
from agent import Shard

class EpidemicModel:
    def __init__(self, df_data, config_path='config.yaml'):
        """
        df_data: DataFrame LIMPIO donde 'id_municipio' YA ES un entero (0..N).
        """
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
            
        self.device = torch.device(self.params['simulation']['device'])
        self.step_count = 0
        self.history = []
        
        # 1. PREPARACIÓN ESTRUCTURAL
        self.num_pueblos = len(df_data)
        
        # El buzón tiene el tamaño exacto del número de filas
        self.mailbox = [[] for _ in range(self.num_pueblos)]
        
        self.shards = []
        self.routing_table = {} 
        
        TARGET_SHARD_SIZE = 500_000 
        
        batch = []
        current_pop = 0
        shard_idx = 0
        
        # Lista temporal para saber qué IDs (índices) son válidos para viajar
        valid_ids_list = []
        
        # 2. CREACIÓN DE SHARDS
        print(f"⚙️  Distribuyendo {self.num_pueblos} municipios en Shards GPU...")
        
        # Iteramos sobre el DataFrame que nos pasó el main
        for _, row in tqdm(df_data.iterrows(), total=self.num_pueblos, desc="Creando Shards", unit="pueblo"):
            pop = int(row['poblacion'])
            if pop <= 0: continue
            
            # Como id_municipio es el índice, esto es un entero seguro
            valid_ids_list.append(int(row['id_municipio']))
            
            batch.append(row)
            current_pop += pop
            
            if current_pop >= TARGET_SHARD_SIZE:
                self._create_shard(batch, shard_idx)
                batch = []
                current_pop = 0
                shard_idx += 1
                
        if batch: self._create_shard(batch, shard_idx)
        
        # Tensor de IDs válidos para el enrutador (viajes rápidos)
        self.valid_ids = torch.tensor(valid_ids_list, dtype=torch.long, device='cpu')
        
        print(f"✅ Sharding completado: {len(self.shards)} bloques en {self.device}.")

    def _create_shard(self, batch, idx):
        df_subset = pd.DataFrame(batch)
        # agent.py recibirá este subset con la columna 'id_municipio' corregida
        new_shard = Shard(df_subset, idx, self.params)
        self.shards.append(new_shard)
        
        # Rellenar tabla de enrutamiento
        for local_idx, global_id in enumerate(new_shard.global_ids.tolist()):
            self.routing_table[global_id] = (idx, local_idx)

    def step(self):
        # Reiniciamos buzón
        next_mailbox = [[] for _ in range(self.num_pueblos)]
        daily_stats = {'day': self.step_count, 'S':0, 'I':0, 'R':0, 'D':0, 'Moves': 0}
        
        # Barra de progreso interna (oculta al terminar para no ensuciar)
        pbar = tqdm(self.shards, desc=f"Día {self.step_count}", leave=False, unit="shard")
        
        for shard in pbar:
            # --- FASE 1: CHECK-IN ---
            shard_incomers = []
            shard_dest_ids = []
            
            for local_idx, global_id in enumerate(shard.global_ids.tolist()):
                # global_id es un entero (índice), acceso directo ultra-rápido
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

            # --- FASE 2: SIMULACIÓN ---
            shard.to_gpu()
            seed = shard.base_seed + (self.step_count * 1000)
            rng = torch.Generator(device=self.device).manual_seed(seed)
            leavers = shard.step(rng)
            
            stats = shard.get_summary()
            for k in ['S','I','R','D']: daily_stats[k] += stats[k]
            shard.to_cpu()
            
            # --- FASE 3: ENRUTAMIENTO ---
            if leavers:
                num_leavers = leavers['state'].shape[0]
                daily_stats['Moves'] += num_leavers
                
                # Sorteo de destinos usando solo índices válidos
                idx_choices = torch.randint(0, len(self.valid_ids), (num_leavers,))
                destinations = self.valid_ids[idx_choices]
                
                self._distribute_to_mailboxes(leavers, destinations, next_mailbox)
            
            pbar.set_postfix(I=f"{daily_stats['I']:,}")

        self.mailbox = next_mailbox
        self.history.append(daily_stats)
        self.step_count += 1
        
        return daily_stats

    # Helpers
    def _merge_travelers(self, list_of_dicts):
        if not list_of_dicts: return None
        merged = {}
        for key in list_of_dicts[0].keys():
            merged[key] = torch.cat([d[key] for d in list_of_dicts], dim=0)
        return merged

    def _distribute_to_mailboxes(self, leavers, destinations, target_mailbox):
        unique_dests = torch.unique(destinations)
        for dest_id in unique_dests:
            d_id = int(dest_id.item())
            if d_id >= len(target_mailbox): continue
            
            mask = (destinations == dest_id)
            package = {k: v[mask] for k, v in leavers.items()}
            target_mailbox[d_id].append(package)

    def export_results(self, filename="resultados.csv"):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        return df