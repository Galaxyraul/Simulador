import torch
class Shard():
    def __init__(self, df_subset, shard_id, params):
            self.device = 'cpu' # Empezamos en CPU para no saturar VRAM al cargar
            self.gpu_device = params['simulation']['device']
            self.params = params
            self.shard_id = shard_id
            
            # 1. METADATOS
            # Como viene del main ya procesado, 'id_municipio' es el entero (índice)
            self.global_ids = torch.tensor(df_subset['id_municipio'].values, dtype=torch.long)
            self.population_sizes = torch.tensor(df_subset['poblacion'].values, dtype=torch.float32)
            
            self.num_towns = len(df_subset)
            self.total_N = int(self.population_sizes.sum().item())

            # 2. TOWN IDs (Mapeo agente -> pueblo)
            local_indices = torch.arange(self.num_towns, dtype=torch.long)
            counts = torch.tensor(df_subset['poblacion'].values, dtype=torch.long)
            self.town_ids = torch.repeat_interleave(local_indices, counts)

            # 3. SEMILLA Y GENERADOR (Para que tu aleatoriedad sea reproducible)
            self.base_seed = params['simulation']['seed'] + (shard_id * 9999)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(self.base_seed)

            # 4. INICIALIZACIÓN DE AGENTES (TU CÓDIGO RESTAURADO)
            # -----------------------------------------------------------------
            # Estado (0=S, 1=I, 2=R, 3=D)
            self.state = torch.zeros(self.total_N, dtype=torch.uint8)
            self.days_in_state = torch.zeros(self.total_N, dtype=torch.int16)
            
            self.susceptibility = (0.7 + 0.3 * torch.rand(self.total_N, generator=rng)).to(torch.float16)
            self.noncompliance = (0.2 + 0.8 * torch.rand(self.total_N, generator=rng)).to(torch.float16)
            self.mobility = (0.5 + 0.5 * torch.rand(self.total_N, generator=rng)).to(torch.float16)
            
            # Reinfecciones (opcional, pero útil)
            self.times_infected = torch.zeros(self.total_N, dtype=torch.int8)
            # -----------------------------------------------------------------

            # 5. DEFINE QUÉ VIAJA A LA GPU
            # ¡Es crucial añadir 'noncompliance' y 'mobility' aquí!
            self.tensor_attributes = [
                'state', 'days_in_state', 'susceptibility', 
                'noncompliance', 'mobility', 'town_ids', 'times_infected'
            ]

            # 6. PACIENTE CERO
            self._initialize_infections(params, rng) # Pasamos el rng para seguir la cadena aleatoria
            

    def _initialize_infections(self, params, rng): # Añadimos argumento rng
        rate = params['simulation'].get('initial_infection_rate', 0.005)
        if rate <= 0: return
        
        num_infected = int(self.total_N * rate)
        if num_infected == 0 and self.total_N > 0: num_infected = 1
        
        # Usamos el rng que viene del init para mantener consistencia
        indices = torch.randperm(self.total_N, generator=rng)[:num_infected]
        
        self.state[indices] = 1
        self.days_in_state[indices] = 1

    def _to_gpu(self):
        self.population_sizes = self.population_sizes.to(self.gpu_device)
        for attr in self.tensor_attributes:
            t = getattr(self, attr)
            setattr(self, attr, t.to(self.gpu_device))
        self.device = self.gpu_device

    def _to_cpu(self):
        self.population_sizes = self.population_sizes.to('cpu')
        for attr in self.tensor_attributes:
            t = getattr(self, attr)
            setattr(self, attr, t.to('cpu'))
        self.device = 'cpu'
        # torch.cuda.empty_cache() # Opcional: limpiar caché si hay poca VRAM

    # --- LÓGICA DE SIMULACIÓN ---

    def step_infection(self, generator):
        """
        Calcula contagios usando aproximación exponencial (Poisson process).
        Permite reinfección de Recuperados y Vacunados según su susceptibilidad.
        """
        # 1. ¿Quién contagia? (Solo los infectados activos)
        infected_mask = (self.state == 1)
        if not infected_mask.any(): return

        global_mask_factor = self.params['population'].get('mask_factor', 1.0)
        global_lockdown = self.params['population'].get('lockdown_factor', 0.0)
        nonc = getattr(self, 'noncompliance', torch.ones(self.total_N, device=self.device))
        mob = getattr(self, 'mobility', torch.ones(self.total_N, device=self.device))
        
        contacts_param = self.params['population']['contacts_per_day']
        
        # "Fuerza viral" que emite cada infectado hoy
        # (Contactos * Riesgo por comportamiento)
        effective_lockdown = (1.0 - global_lockdown) + (global_lockdown * nonc[infected_mask])
        
        viral_emission = contacts_param * effective_lockdown * mob[infected_mask]
        
        # Agrupamos toda esa carga viral por pueblo
        # Resultado: [Carga_Pueblo0, Carga_Pueblo1...]
        total_force_per_town = torch.bincount(
            self.town_ids[infected_mask], 
            weights=viral_emission, 
            minlength=self.num_towns
        )
        
        env_risk_per_town = total_force_per_town / (self.population_sizes + 1e-6)

        candidates_mask = (self.state != 1) & (self.state != 3)
        
        # Si no hay nadie sano, salimos
        if not candidates_mask.any(): return
        
        # 1. Asignamos el riesgo ambiental de su pueblo a cada candidato
        my_env_risk = env_risk_per_town[self.town_ids[candidates_mask]]

        my_susceptibility = self.susceptibility[candidates_mask]
        my_noncompliance = nonc[candidates_mask]
        my_effective_mask = global_mask_factor + (1.0 - global_mask_factor) * my_noncompliance
        p_base = self.params['virus']['P_base']
        
        lambda_infection = my_env_risk * p_base * my_susceptibility * my_effective_mask
        
        prob_infection = 1.0 - torch.exp(-lambda_infection)

        rand_vals = torch.rand(candidates_mask.sum().item(), device=self.device, generator=generator)

        newly_infected_local = rand_vals < prob_infection

        if newly_infected_local.any():
            # Necesitamos mapear los índices locales (del subset candidates) a los globales
            # Truco de PyTorch: indices de los True dentro de candidates_mask
            candidate_indices = torch.nonzero(candidates_mask).squeeze()
            
            # Seleccionamos los índices globales que se han infectado
            global_indices_infected = candidate_indices[newly_infected_local]
            
            # Aplicar infección
            self.state[global_indices_infected] = 1
            self.days_in_state[global_indices_infected] = 0

            if hasattr(self, 'times_infected'):
                self.times_infected[global_indices_infected] += 1

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

    def apply_vaccine(self, generator, daily_rate, vaccine_efficacy=0.1):
        """
        Vacuna a un porcentaje de la población viva y elegible.
        daily_rate: % de la población a vacunar hoy (ej: 0.005 es 0.5% diario)
        vaccine_efficacy: Nueva susceptibilidad (0.1 = 90% inmune)
        """
        # 1. Buscar candidatos: Vivos (no state 3) y que tengan susceptibilidad alta
        # (Para no gastar vacunas en gente que ya las tiene o que es inmune natural)
        candidates_mask = (self.state != 3) & (self.susceptibility > vaccine_efficacy)
        
        num_candidates = candidates_mask.sum().item()
        if num_candidates == 0: return

        # 2. Calcular cuántas vacunas tocan hoy en este Shard
        # Calculamos sobre el total de población para mantener ritmo constante
        n_to_vaccinate = int(self.total_N * daily_rate)
        
        # No podemos vacunar a más gente de la que hay disponible
        n_to_vaccinate = min(n_to_vaccinate, num_candidates)
        if n_to_vaccinate == 0: return

        # 3. Selección Aleatoria
        # Generamos índices aleatorios dentro de los candidatos
        rand_indices = torch.randperm(num_candidates, generator=generator, device=self.device)[:n_to_vaccinate]
        
        # Mapeamos a índices globales
        candidate_global_indices = torch.nonzero(candidates_mask).squeeze()
        
        # Si solo hay 1 candidato, squeeze puede devolver un escalar, aseguramos dimensión
        if candidate_global_indices.ndim == 0:
            candidate_global_indices = candidate_global_indices.unsqueeze(0)
            
        target_indices = candidate_global_indices[rand_indices]

        # 4. PINCHAZO
        # Bajamos su susceptibilidad al nivel de eficacia (ej: 0.1)
        self.susceptibility[target_indices] = vaccine_efficacy
    
    def step_immunity_loss(self):
        """
        Simula la pérdida progresiva de anticuerpos.
        La susceptibilidad de todos tiende a volver a 1.0 con el tiempo.
        """
        # Recuperamos la velocidad de pérdida del YAML (o valor por defecto)
        # 0.005 significa que recuperas aprox el 0.5% de tu susceptibilidad cada día
        # Una tasa de 0.005 implica que en ~6 meses (180 días) pierdes gran parte de la inmunidad.
        waning_rate = self.params['virus'].get('immunity_waning_rate', 0.005)
        
        if waning_rate <= 0: return

        # Aplicamos la fórmula a TODOS (vectorizado es instantáneo)
        # S_new = S_old + rate * (Distancia_hasta_1)
        
        # Nota: Clamp para asegurar que por errores de punto flotante no pasemos de 1.0
        self.susceptibility += waning_rate * (1.0 - self.susceptibility)
        self.susceptibility = torch.clamp(self.susceptibility, 0.0, 1.0)

    def step(self, generator):
        self.step_infection(generator)
        self.step_recover(generator)
        self.step_immunity_loss()
        return self.step_leaving(generator)