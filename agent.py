import torch

class Shard:
    def __init__(self, df_subset, shard_id, params, variant_info):
        dev_conf = params['simulation'].get('device', 'cpu')
        self.gpu_device = dev_conf if isinstance(dev_conf, str) else ('cuda' if dev_conf == 1 else 'cpu')
        self.device = 'cpu'
        
        self.params = params
        self.shard_id = shard_id
        
        # 1. INFO VARIANTES (Desde model.py)
        self.n_vars = variant_info['count']
        self.var_p_bases = variant_info['p_bases']      # P_base
        self.var_recoveries = variant_info['recoveries']
        self.var_deaths = variant_info['deaths']
        self.var_wanings = variant_info['wanings']
        self.var_starts = variant_info['starts']        # step_start

        # 2. METADATOS
        if 'id_municipio' not in df_subset.columns: df_subset['id_municipio'] = df_subset.index
        self.global_ids = torch.tensor(df_subset['id_municipio'].values, dtype=torch.long)
        self.population_sizes = torch.tensor(df_subset['poblacion'].values, dtype=torch.float32)
        self.num_towns = len(df_subset)
        self.total_N = int(self.population_sizes.sum().item())

        local_indices = torch.arange(self.num_towns, dtype=torch.long)
        counts = torch.tensor(df_subset['poblacion'].values, dtype=torch.long)
        self.town_ids = torch.repeat_interleave(local_indices, counts)

        # 3. SEMILLA
        seed_val = params['simulation'].get('seed', 42)
        self.base_seed = seed_val + (shard_id * 9999)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.base_seed)

        # 4. INICIALIZACIÓN CON 'DISTRIBUTIONS' DEL YAML
        # Leemos la sección distributions
        dists = params.get('distributions', {})
        
        self.state = torch.zeros(self.total_N, dtype=torch.uint8)
        self.variant_id = torch.full((self.total_N,), -1, dtype=torch.int8)
        self.days_in_state = torch.zeros(self.total_N, dtype=torch.int16)
        
        # Función auxiliar para generar dist normal truncada o base
        def gen_attribute(key, default_base, default_var):
            conf = dists.get(key, {'base': default_base, 'variance': default_var})
            base = conf.get('base', default_base)
            var = conf.get('variance', default_var)
            # Generamos distribución normal centrada en base
            vals = torch.normal(mean=base, std=var, size=(self.total_N,), generator=rng)
            return torch.clamp(vals, 0.0, 1.0).to(torch.float16)

        # Generamos atributos según YAML
        self.susceptibility = gen_attribute('susceptibility', 0.7, 0.3)
        self.noncompliance = gen_attribute('noncompliance', 0.2, 0.8)
        self.mobility = gen_attribute('mobility', 0.5, 0.5)

        # 5. ATRIBUTOS
        self.tensor_attributes = [
            'state', 'variant_id', 'days_in_state', 'susceptibility', 
            'noncompliance', 'mobility', 'town_ids'
        ]
        self.static_gpu_tensors = ['var_p_bases', 'var_recoveries', 'var_deaths', 'var_wanings', 'var_starts', 'population_sizes']

        # 6. INFECCIÓN INICIAL
        self._initialize_infections(params, rng)

    def _initialize_infections(self, params, rng):
        rate = params['simulation'].get('initial infection rate', 0.005) # Nota el espacio en tu YAML
        if rate <= 0: return
        num_infected = int(self.total_N * rate) or 1
        indices = torch.randperm(self.total_N, generator=rng)[:num_infected]
        self.state[indices] = 1
        self.days_in_state[indices] = 1
        self.variant_id[indices] = 0 # Empezamos con la primera variante

    def _to_gpu(self):
        if self.gpu_device == 'cpu': return
        for attr in self.tensor_attributes:
            setattr(self, attr, getattr(self, attr).to(self.gpu_device))
        for attr in self.static_gpu_tensors:
            setattr(self, attr, getattr(self, attr).to(self.gpu_device))
        self.device = self.gpu_device

    def _to_cpu(self):
        for attr in self.tensor_attributes:
            setattr(self, attr, getattr(self, attr).to('cpu'))
        # Estáticos también a CPU para liberar VRAM si es necesario
        for attr in self.static_gpu_tensors:
            setattr(self, attr, getattr(self, attr).to('cpu'))
        self.device = 'cpu'

    # --- LÓGICA DE INFECCIÓN CORREGIDA ---

    def step_infection(self, generator, current_day):
        """
        Calcula lambda basándose en P_base y contactos diarios.
        """
        # Parámetros Globales
        contacts_daily = self.params['population'].get('contacts_per_day', 30)
        mask_factor = self.params['population'].get('mask_factor', 0.5)
        lockdown = self.params['population'].get('lockdown_factor', 0.0)

        # Iteramos variantes
        for v_idx in range(self.n_vars):
            # 1. Comprobar si la variante ya empezó
            start_day = self.var_starts[v_idx]
            if current_day < start_day:
                continue 

            # 2. Infectadores de esta variante
            infectors_mask = (self.state == 1) & (self.variant_id == v_idx)
            if not infectors_mask.any(): continue 

            # 3. Calcular "Carga Viral" emitida
            # Fórmula: Contactos * Movilidad * (1 - Lockdown si cumple)
            eff_lockdown = (1.0 - lockdown) + (lockdown * self.noncompliance[infectors_mask])
            emission = contacts_daily * self.mobility[infectors_mask] * eff_lockdown
            
            # Sumar emisiones por pueblo
            force_per_town = torch.bincount(
                self.town_ids[infectors_mask], 
                weights=emission, 
                minlength=self.num_towns
            )
            
            # 4. Riesgo de encuentro (fuerza / población)
            # Esto representa cuántos contactos con infectados recibe una persona promedio
            encounter_rate_per_town = force_per_town / (self.population_sizes + 1e-6)

            # 5. Candidatos
            candidates_mask = (self.state == 0) | (self.state == 2)
            if not candidates_mask.any(): break
            
            # 6. Calcular Lambda para candidatos
            my_encounter_rate = encounter_rate_per_town[self.town_ids[candidates_mask]]
            if not (my_encounter_rate > 0).any(): continue

            # P_base es la probabilidad de infección DADO un contacto
            p_base = self.var_p_bases[v_idx]
            
            # Modificadores del candidato
            my_susc = self.susceptibility[candidates_mask]
            
            # Factor mascarilla: Si mask_factor es 0.5 (protección 50%)
            # Si noncompliance es 0 (usa siempre), factor = 0.5
            # Si noncompliance es 1 (nunca usa), factor = 1.0
            my_mask_effect = mask_factor + (1.0 - mask_factor) * self.noncompliance[candidates_mask]

            # Lambda = TasaEncuentros * Prob_Transmision * Susceptibilidad * Mascarilla
            lambda_val = my_encounter_rate * p_base * my_susc * my_mask_effect
            
            # Probabilidad Final (Poisson)
            prob_infection = 1.0 - torch.exp(-lambda_val)

            # 7. Tirada
            rand_vals = torch.rand(candidates_mask.sum().item(), device=self.device, generator=generator)
            new_infections = rand_vals < prob_infection

            if new_infections.any():
                cand_indices = torch.nonzero(candidates_mask).squeeze()
                if cand_indices.ndim == 0: cand_indices = cand_indices.unsqueeze(0)
                real_indices = cand_indices[new_infections]
                
                self.state[real_indices] = 1
                self.days_in_state[real_indices] = 0
                self.variant_id[real_indices] = v_idx

    def step_recover(self, generator):
        infected_mask = (self.state == 1)
        if not infected_mask.any(): return
        
        self.days_in_state[infected_mask] += 1
        active_vars = self.variant_id[infected_mask].long()

        # Muerte
        my_death_prob = self.var_deaths[active_vars]
        rand = torch.rand(infected_mask.sum().item(), device=self.device, generator=generator)
        dying = rand < my_death_prob
        
        if dying.any():
            inf_indices = torch.nonzero(infected_mask).squeeze()
            if inf_indices.ndim == 0: inf_indices = inf_indices.unsqueeze(0)
            self.state[inf_indices[dying]] = 3
            infected_mask[inf_indices[dying]] = False
            # Recalcular active_vars
            active_vars = self.variant_id[infected_mask].long()

        # Recuperación
        my_recovery = self.var_recoveries[active_vars]
        my_days = self.days_in_state[infected_mask]
        recovering = my_days >= my_recovery
        
        if recovering.any():
            inf_indices = torch.nonzero(infected_mask).squeeze()
            if inf_indices.ndim == 0: inf_indices = inf_indices.unsqueeze(0)
            
            real_recov = inf_indices[recovering]
            self.state[real_recov] = 2
        
            self.susceptibility[real_recov] = 0.0

    def step_immunity_loss(self):
        # Usamos el waning rate de la variante que te infectó (simplificación)
        # O el máximo si queremos ser pesimistas. Aquí usaremos un promedio simple o el waning del YAML si fuera global.
        # Dado que waning está en variante, lo aplicamos a los RECUPERADOS (State 2)
        recovered_mask = (self.state == 2)
        if not recovered_mask.any(): return
        
        recov_vars = self.variant_id[recovered_mask].long()
        my_waning = self.var_wanings[recov_vars]
        
        # S_new = S_old + waning * (1 - S_old)
        # Recuperan susceptibilidad
        self.susceptibility[recovered_mask] += my_waning * (1.0 - self.susceptibility[recovered_mask])
        self.susceptibility = torch.clamp(self.susceptibility, 0.0, 1.0)

    def step_leaving(self, generator):
        # (Sin cambios en lógica de movimiento, solo asegurar que variant_id viaja)
        prob_leave = self.params['population'].get('leave_prob', 0.0001)
        rand = torch.rand(self.total_N, device=self.device, generator=generator)
        leaving = (rand < prob_leave) & (self.state != 3)
        
        n_leavers = leaving.sum().item()
        if n_leavers == 0: return None
        
        ids_leaving_dev = self.town_ids[leaving]
        leavers = {}
        for attr in self.tensor_attributes:
            leavers[attr] = getattr(self, attr)[leaving].to('cpu')
            
        global_ids_cpu = self.global_ids.to('cpu')
        leavers['origin_global_id'] = global_ids_cpu[leavers['town_ids']]
        
        leavers_count = torch.bincount(ids_leaving_dev, minlength=self.num_towns).float()
        self.population_sizes -= leavers_count

        keep = ~leaving
        for attr in self.tensor_attributes:
            setattr(self, attr, getattr(self, attr)[keep])
        self.total_N = self.state.shape[0]
        return leavers

    def add_incomers(self, incomers, dest_ids):
        if not incomers: return
        incomers['town_ids'] = dest_ids.to(self.device)
        for attr in self.tensor_attributes:
            setattr(self, attr, torch.cat([getattr(self, attr), incomers[attr].to(self.device)], dim=0))
        self.total_N = self.state.shape[0]
        new_counts = torch.bincount(dest_ids.to(self.device), minlength=self.num_towns).float()
        self.population_sizes += new_counts

    def step(self, generator, current_day):
        self.step_infection(generator, current_day)
        self.step_recover(generator)
        self.step_immunity_loss()
        return self.step_leaving(generator)

    def get_summary(self):
        c = torch.bincount(self.state.long(), minlength=4)
        return {'S': int(c[0]), 'I': int(c[1]), 'R': int(c[2]), 'D': int(c[3])}

    def get_municipality_stats(self):
        stats = {}
        
        # 1. Máscaras para Infectados (1) y Muertos (3)
        inf_mask = (self.state == 1).float()
        dead_mask = (self.state == 3).float()
        
        # 2. Conteo ponderado por pueblo
        inf_town = torch.bincount(self.town_ids, weights=inf_mask, minlength=self.num_towns)
        dead_town = torch.bincount(self.town_ids, weights=dead_mask, minlength=self.num_towns)
        
        for i in range(self.num_towns):
            gid = self.global_ids[i].item()
            pop = self.population_sizes[i].item()
            
            if pop > 0:
                # Devolvemos una tupla o dict pequeño: (ratio_infectados, ratio_muertos)
                stats[gid] = {
                    'I': inf_town[i].item() / pop,
                    'D': dead_town[i].item() / pop
                }
            else:
                stats[gid] = {'I': 0.0, 'D': 0.0}
        return stats