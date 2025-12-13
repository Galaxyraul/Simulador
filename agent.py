import torch
class Shard():
    def __init__(self, df_subset, shard_id, params):
        """
        Un Shard agrupa varios pueblos para procesarlos juntos en la GPU.
        """
        self.device = 'cpu'
        self.gpu_device = params['simulation']['device']
        self.params = params
        self.shard_id = shard_id
        
        # 1. METADATOS (Identificación de pueblos)
        # ID Global (Real del INE) y Población inicial
        self.global_ids = torch.tensor(df_subset['id_municipio'].values, dtype=torch.long)
        self.population_sizes = torch.tensor(df_subset['poblacion'].values, dtype=torch.float32)
        
        self.num_towns = len(df_subset)
        self.total_N = int(self.population_sizes.sum().item())

        # 2. CONSTRUCCIÓN DE MÁSCARAS DE IDENTIDAD
        # town_ids: Dice a qué índice LOCAL (0..num_towns-1) pertenece cada agente
        poblaciones_lista = df_subset['poblacion'].values.tolist()
        local_indices = torch.arange(self.num_towns, dtype=torch.long)
        
        self.town_ids = torch.repeat_interleave(
            local_indices, 
            torch.tensor(poblaciones_lista, dtype=torch.long)
        )

        # 3. ESTADOS DE LOS AGENTES
        # 0: Susceptible, 1: Infectado, 2: Recuperado, 3: Muerto
        self.state = torch.zeros(self.total_N, dtype=torch.uint8)
        self.days_in_state = torch.zeros(self.total_N, dtype=torch.int16)
        self.susceptibility = torch.ones(self.total_N, dtype=torch.float16) 
        # Aquí puedes añadir más (mobility, mask_usage, age...)

        # Atributos que deben viajar a la GPU
        self.tensor_attributes = ['state', 'days_in_state', 'susceptibility', 'town_ids']

        # 4. PACIENTE CERO (Vectorizado)
        # Usamos una semilla única por Shard para reproducibilidad
        self.base_seed = params['simulation']['seed'] + (shard_id * 9999)
        self._initialize_infections(params)

    def _initialize_infections(self, params):
        rate = params['simulation'].get('initial_infection_rate', 0.005)
        if rate <= 0: return
        
        # Infectamos un % de la población total del Shard
        num_infected = int(self.total_N * rate)
        if num_infected == 0 and self.total_N > 0: num_infected = 1
        
        rng = torch.Generator().manual_seed(self.base_seed)
        indices = torch.randperm(self.total_N, generator=rng)[:num_infected]
        
        self.state[indices] = 1 # Infectado
        self.days_in_state[indices] = 1 

    def to_gpu(self):
        self.population_sizes = self.population_sizes.to(self.gpu_device)
        for attr in self.tensor_attributes:
            t = getattr(self, attr)
            setattr(self, attr, t.to(self.gpu_device))
        self.device = self.gpu_device

    def to_cpu(self):
        self.population_sizes = self.population_sizes.to('cpu')
        for attr in self.tensor_attributes:
            t = getattr(self, attr)
            setattr(self, attr, t.to('cpu'))
        self.device = 'cpu'
        # torch.cuda.empty_cache() # Opcional: limpiar caché si hay poca VRAM

    # --- LÓGICA DE SIMULACIÓN ---

    def step_infection(self, generator):
        """Calcula nuevos contagios respetando las fronteras de los pueblos"""
        infected_mask = (self.state == 1)
        if not infected_mask.any(): return

        # 1. Contar infectados por pueblo (Local Index)
        infected_counts = torch.bincount(self.town_ids[infected_mask], minlength=self.num_towns).float()
        
        # 2. Densidad local (Infectados / Población del pueblo)
        # Sumamos epsilon para evitar división por cero si un pueblo se vacía
        densities = infected_counts / (self.population_sizes + 1e-6)
        
        # 3. Asignar riesgo individual
        my_density = densities[self.town_ids]
        
        # 4. Física
        p_base = self.params['virus']['P_base']
        contacts = self.params['population']['contacts_per_day']
        
        risk = 1.0 - torch.pow(1.0 - p_base * my_density, contacts)
        
        # 5. Tirar dados
        susceptibles = (self.state == 0)
        rand_vals = torch.rand(self.total_N, device=self.device, generator=generator)
        new_infections = susceptibles & (rand_vals < risk)
        
        if new_infections.any():
            self.state[new_infections] = 1
            self.days_in_state[new_infections] = 0

    def step_recover(self, generator):
        """Evolución: Muerte o Recuperación"""
        infected_mask = (self.state == 1)
        if not infected_mask.any(): return
        
        # Avanzar días
        self.days_in_state[infected_mask] += 1
        
        # Muerte
        death_prob = self.params['virus']['death_prob']
        rand_vals = torch.rand(self.total_N, device=self.device, generator=generator)
        dying_mask = infected_mask & (rand_vals < death_prob)
        
        if dying_mask.any():
            self.state[dying_mask] = 3 # Dead
            infected_mask = infected_mask & (~dying_mask) # Ya no cuenta como infectado
            
        # Recuperación
        recovery_days = self.params['virus']['recovery_day']
        recovering_mask = infected_mask & (self.days_in_state >= recovery_days)
        
        if recovering_mask.any():
            self.state[recovering_mask] = 2 # Recovered

    def step_leaving(self, generator):
        """Decide quién viaja y devuelve sus datos"""
        prob_leave = self.params['population']['leave_prob']
        rand_vals = torch.rand(self.total_N, device=self.device, generator=generator)
        
        # Condición: Querer viajar Y no estar muerto
        leaving_mask = (rand_vals < prob_leave) & (self.state != 3)
        
        num_leavers = leaving_mask.sum().item()
        if num_leavers == 0: return None
        
        # --- PASO 1: Capturar IDs para el conteo (ANTES DE BORRAR NADA) ---
        # Guardamos los town_ids de los que se van mientras el tensor sigue completo.
        # Lo mantenemos en el device actual (GPU/CPU) para hacer el bincount rápido.
        ids_leaving_device = self.town_ids[leaving_mask]

        # --- PASO 2: Empaquetar datos para el viaje (Mover a CPU) ---
        leavers = {}
        for attr in self.tensor_attributes:
            # Extraemos y mandamos a CPU para el enrutador
            leavers[attr] = getattr(self, attr)[leaving_mask].to('cpu')
            
        # Añadir ID Global de origen (ahora seguro porque leavers está en CPU)
        leavers['origin_global_id'] = self.global_ids[leavers['town_ids']]
        
        # --- PASO 3: Actualizar Población (Usando los IDs capturados en Paso 1) ---
        # Es crucial hacer esto antes o usando la variable auxiliar, no el tensor principal
        leavers_count = torch.bincount(ids_leaving_device, minlength=self.num_towns).float()
        self.population_sizes -= leavers_count

        # --- PASO 4: ELIMINAR AGENTES (Operación Destructiva) ---
        # Ahora sí, reducimos el tamaño de los tensores principales
        keep_mask = ~leaving_mask
        for attr in self.tensor_attributes:
            setattr(self, attr, getattr(self, attr)[keep_mask])
            
        self.total_N = self.state.shape[0]
        
        return leavers

    def add_incomers(self, incomers_dict, dest_local_ids):
        """Recibe viajeros y los añade a los tensores"""
        if incomers_dict is None or len(dest_local_ids) == 0: return
        
        # Actualizar town_ids de los que llegan para que coincidan con este Shard
        incomers_dict['town_ids'] = dest_local_ids.to(self.device)
        
        # Concatenar
        for attr in self.tensor_attributes:
            current = getattr(self, attr)
            incoming = incomers_dict[attr].to(self.device)
            setattr(self, attr, torch.cat([current, incoming], dim=0))
            
        self.total_N = self.state.shape[0]
        
        # Actualizar conteos
        new_counts = torch.bincount(dest_local_ids.to(self.device), minlength=self.num_towns).float()
        self.population_sizes += new_counts

    def get_summary(self):
        """Métricas rápidas"""
        counts = torch.bincount(self.state, minlength=4)
        return {
            'S': int(counts[0].item()),
            'I': int(counts[1].item()),
            'R': int(counts[2].item()),
            'D': int(counts[3].item())
        }

    def step(self, generator):
        self.step_infection(generator)
        self.step_recover(generator)
        return self.step_leaving(generator)