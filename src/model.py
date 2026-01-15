import torch
import pandas as pd
from src.agent import Shard

class EpidemicModel:
    def __init__(self, df_data, config):
        self.params = config   
        
        # 1. Device
        dev_conf = self.params.get('simulation', {}).get('device', 'cpu')
        dev_str = 'cuda' if dev_conf == 1 else 'cpu' if dev_conf == 0 else str(dev_conf)
        if dev_str == 'cuda' and not torch.cuda.is_available(): dev_str = 'cpu'
        self.device = torch.device(dev_str)
        
        # 2. Parsear Variantes
        self.variant_info = self._parse_variants(config)
        print(f"⚙️ Iniciando modelo en: {self.device} | Variantes: {self.variant_info['names']}")

        self.step_count = 0
        self.num_pueblos = len(df_data)
        
        # 3. Datos Geográficos
        df_data = df_data.reset_index(drop=True)
        if 'id_municipio' not in df_data.columns: df_data['id_municipio'] = df_data.index
        
        # Gravedad
        self.population_sizes = torch.tensor(df_data['poblacion'].values, device=self.device, dtype=torch.float32)
        
        if 'coord_x' not in df_data.columns: df_data['coord_x'] = 0.0; df_data['coord_y'] = 0.0
        coords = torch.tensor(df_data[['coord_x', 'coord_y']].values, dtype=torch.float32, device=self.device)
        
        dists = torch.cdist(coords, coords)
        dists = torch.clamp(dists, min=100.0) 
        alpha = 2.0 
        attraction = self.population_sizes.unsqueeze(0)
        weights = attraction / (dists.pow(alpha))
        weights.fill_diagonal_(0.0)
        row_sums = weights.sum(dim=1, keepdim=True)
        weights[row_sums.squeeze() < 1e-9] = 1.0
        weights.fill_diagonal_(0.0)
        row_sums = weights.sum(dim=1, keepdim=True)
        self.travel_probs = weights / (row_sums + 1e-9)
        
        del dists, coords, weights, attraction, row_sums
        if self.device.type == 'cuda': torch.cuda.empty_cache()

        # 4. Crear Shards
        self.shards = []
        self.mailbox = [[] for _ in range(self.num_pueblos)]
        
        target_shard_size = 1_000_000 
        current_shard_df = []
        current_pop = 0
        shard_id_counter = 0
        
        for idx, row in df_data.iterrows():
            row_dict = row.to_dict()
            current_shard_df.append(pd.Series(row_dict))
            current_pop += row['poblacion']
            
            if current_pop >= target_shard_size:
                df_chunk = pd.DataFrame(current_shard_df)
                self.shards.append(Shard(df_chunk, shard_id_counter, self.params, self.variant_info))
                current_shard_df = []
                current_pop = 0
                shard_id_counter += 1
        
        if current_shard_df:
            df_chunk = pd.DataFrame(current_shard_df)
            self.shards.append(Shard(df_chunk, shard_id_counter, self.params, self.variant_info))

    def _parse_variants(self, config):
        variants_dict = config.get('variants', {})
        if not variants_dict:
             variants_dict = config.get('epidemiology', {}).get('variants', {})

        names = list(variants_dict.keys())
        p_bases, recoveries, deaths, wanings, starts = [], [], [], [], []

        for name in names:
            data = variants_dict[name]
            p_bases.append(float(data.get('P_base', 0.15)))
            recoveries.append(int(data.get('recovery_day', 14)))
            deaths.append(float(data.get('death_prob', 0.02)))
            wanings.append(float(data.get('immunity_waning_rate', 0.005)))
            starts.append(int(data.get('step_start', 0)))

        return {
            'names': names,
            'count': len(names),
            'p_bases': torch.tensor(p_bases, dtype=torch.float32),
            'recoveries': torch.tensor(recoveries, dtype=torch.int16),
            'deaths': torch.tensor(deaths, dtype=torch.float32),
            'wanings': torch.tensor(wanings, dtype=torch.float32),
            'starts': torch.tensor(starts, dtype=torch.int16)
        }

    def step(self):
        next_mailbox = [[] for _ in range(self.num_pueblos)]
        daily_stats = {'day': self.step_count, 'S':0, 'I':0, 'R':0, 'D':0, 'Moves': 0}
        
        for shard in self.shards:
            # Recepción
            shard_incomers = []
            shard_dest_local_ids = []
            my_global_ids = shard.global_ids.cpu().numpy()
            
            for local_idx, global_id in enumerate(my_global_ids):
                mail = self.mailbox[global_id]
                if mail:
                    merged = self._merge_travelers(mail)
                    if merged:
                        shard_incomers.append(merged)
                        n = merged['state'].shape[0]
                        shard_dest_local_ids.append(torch.full((n,), local_idx, dtype=torch.long))
            
            if shard_incomers:
                final_incomers = self._merge_travelers(shard_incomers)
                final_dest = torch.cat(shard_dest_local_ids)
                shard.add_incomers(final_incomers, final_dest)

            # Simulación
            shard._to_gpu()
            seed = shard.base_seed + (self.step_count * 777)
            rng = torch.Generator(device=shard.gpu_device).manual_seed(seed)
            leavers = shard.step(rng, current_day=self.step_count)
            
            stats = shard.get_summary()
            for k in ['S','I','R','D']: daily_stats[k] += stats[k]
            shard._to_cpu()
            
            # Gravedad (Enrutamiento)
            if leavers:
                origins_gpu = leavers['origin_global_id'].to(self.device)
                daily_stats['Moves'] += origins_gpu.shape[0]
                
                batch_probs = self.travel_probs[origins_gpu]
                dest_indices = torch.multinomial(batch_probs, num_samples=1).squeeze()
                
                # 🔥 CORRECCIÓN CRÍTICA: Proteger contra 0-d tensor si solo hay 1 viajero 🔥
                if dest_indices.ndim == 0:
                    dest_indices = dest_indices.unsqueeze(0)
                
                self._distribute_to_mailboxes(leavers, dest_indices.cpu(), next_mailbox)

        self.mailbox = next_mailbox
        self.step_count += 1
        return daily_stats
    
    def _merge_travelers(self, traveler_list):
        if not traveler_list: return None
        keys = traveler_list[0].keys()
        merged = {}
        for k in keys: merged[k] = torch.cat([t[k] for t in traveler_list])
        return merged

    def _distribute_to_mailboxes(self, leavers, destinations, mailbox):
        """
        Versión Vectorizada: Agrupa viajeros por destino en lugar de iterar uno a uno.
        """
        # 1. Encontrar destinos únicos y sus índices inversos
        # unique_dests: Lista de IDs de pueblos destino únicos
        # inverse_indices: A qué destino único va cada viajero original
        unique_dests, inverse_indices = torch.unique(destinations, return_inverse=True)
        
        # 2. Iterar solo sobre los destinos (ej: 50 pueblos) en lugar de los viajeros (ej: 5000 personas)
        for i, dest_id in enumerate(unique_dests):
            # Máscara booleana para los viajeros que van a ESTE destino
            mask = (inverse_indices == i)
            
            # Extraemos el bloque entero de datos para estos viajeros
            # Esto mantiene los tensores unidos y es mucho más rápido
            traveler_chunk = {k: v[mask] for k, v in leavers.items()}
            
            # Enviamos el paquete al buzón
            mailbox[dest_id.item()].append(traveler_chunk)

    def obtener_estado_visual(self):
        mapa_infeccion = {}
        for shard in self.shards:
            mapa_infeccion.update(shard.get_municipality_stats())
        return mapa_infeccion