

import torch
from mesa import Agent
class Population(Agent):
    def __init__(self, model,N,rng,variant_factor,name,params):
        super().__init__(model)
        self.params = params
        self.cpu = 'cpu'
        self.device = self.params['simulation']['device']
        self.N = N
        # Generador de aleatoriedad
        self.torch_rng = rng
        self.name = name
        #Parámetros q requieren aleatoriedad
        self.susceptibility = (params['distributions']['susceptibility']['base'] + 
                                params['distributions']['susceptibility']['variance'] * 
                                torch.rand(N, device=self.device, generator=self.torch_rng)
                            ).to(torch.float16)
        
        self.noncompliance = (params['distributions']['noncompliance']['base'] + 
                                params['distributions']['noncompliance']['variance'] * 
                                torch.rand(N, device=self.device, generator=self.torch_rng)
                            ).to(torch.float16)
        
        self.mobility = (params['distributions']['mobility']['base'] + 
                            params['distributions']['mobility']['variance'] *
                            torch.rand(N, device=self.device, generator=self.torch_rng)
                        ).to(torch.float16)
        
        self.age_factor = (params['distributions']['age_factor']['base'] + 
                            params['distributions']['age_factor']['multiplier'] * 
                            torch.rand(N, dtype=torch.float16,device=self.device)
                        ).to(torch.float16)

        # Datos informativos
        self.state = torch.zeros(N, dtype=torch.uint8,device=self.device)  # 0=S, 1=I, 2=R, 3=M, 4=V
        self.days_in_state = torch.zeros(N, dtype=torch.uint8,device=self.device)
        self.times_infected = torch.zeros(N, dtype=torch.uint8,device=self.device)
    
        # Parámetros población
        self.contacts_per_day = params['population']['contacts_per_day']
        self.mask_factor = params['population']['mask_factor']
        self.lockdown_factor = params['population']['lockdown_factor']
        self.leave_prob = params['population']['leave_prob']

        #Parámetros virus
        self.recovery_day = params['virus']['recovery_day']
        self.death_prob = params['virus']['death_prob']
        self.P_base = params['virus']['P_base']
        self.variant_factor = variant_factor

        #Variables contadoras por step
        self.deaths_step = 0
        self.infections_step = 0
        self.leaves_step = 0
        self.incomes_step = 0

        self.tensor_attributes = [
            'susceptibility', 'noncompliance', 'mobility', 'age_factor',
            'state', 'days_in_state', 'times_infected'
        ]

    def sample_contacts(self, infected_indices):
        # Mueve los arrays a GPU temporalmente
        susceptibles = torch.nonzero((self.state != 1) & (self.state != 3), as_tuple=True)[0]
        if susceptibles.numel() == 0 or infected_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long)

        infected_indices_gpu = infected_indices

        n_contacts_per_infected = (self.contacts_per_day *
                                   (1 - self.lockdown_factor * self.noncompliance[infected_indices_gpu]) *
                                    self.mobility[infected_indices_gpu]).to(torch.uint16)
        n_total_contacts = int(n_contacts_per_infected.sum().item())
        contacts_gpu = susceptibles[torch.randint(0, len(susceptibles), (n_total_contacts,), device=self.cpu)]
        return contacts_gpu  # regresa a CPU

    def infect_contacts(self, contacts_indices):
        if len(contacts_indices) == 0:
            return

        P_contact = self.P_base * self.susceptibility[contacts_indices] * self.noncompliance[contacts_indices] * self.variant_factor * self.mask_factor

        rand = torch.rand(len(contacts_indices), device=self.device)

        new_infections = rand < P_contact
        indices_to_infect = contacts_indices[new_infections]

        self.infections_step += indices_to_infect.size(0)
        self.state[indices_to_infect] = 1
        self.days_in_state[indices_to_infect] = 0
        self.times_infected[indices_to_infect] += 1
        self.susceptibility[indices_to_infect] = 1.0

    def step_infection(self, batch_infected_size=500_000):
        infected_indices = torch.nonzero(self.state == 1,as_tuple=True)[0]

        total_inf = len(infected_indices)
        if total_inf == 0:
            return

        # process infected in batches
        for start in range(0, total_inf, batch_infected_size):
            end = min(start + batch_infected_size, total_inf)
            infected_batch = infected_indices[start:end]

            # SAMPLE CONTACTS ONLY FOR THIS BATCH
            contacts = self.sample_contacts(infected_batch)

            # INFECT ONLY THESE CONTACTS
            self.infect_contacts(contacts)

    def step_leaving(self):
        """
        Remove individuals who leave this shard and return them for redistribution.
        """
        if self.N == 0:
            return None  # no one to leave
        leave_prob = self.leave_prob * self.noncompliance
        # Mask of leaving individuals
        leaving_mask = torch.rand(self.N, device=self.device) < leave_prob

        if leaving_mask.sum() == 0:
            return None  # nobody leaving this step

        # Create a shard of leaving individuals
        leaving_shard = {
            'state': self.state[leaving_mask].cpu(),
            'days_in_state': self.days_in_state[leaving_mask].cpu(),
            'times_infected': self.times_infected[leaving_mask].cpu(),
            'susceptibility': self.susceptibility[leaving_mask].cpu(),
            'noncompliance': self.noncompliance[leaving_mask].cpu(),
            'mobility': self.mobility[leaving_mask].cpu(),
            'age_factor': self.age_factor[leaving_mask].clone().cpu()
        }

        # Keep only those who stay
        staying_mask = ~leaving_mask
        self.state = self.state[staying_mask]
        self.days_in_state = self.days_in_state[staying_mask]
        self.times_infected = self.times_infected[staying_mask]
        self.susceptibility = self.susceptibility[staying_mask]
        self.noncompliance = self.noncompliance[staying_mask]
        self.mobility = self.mobility[staying_mask]
        self.age_factor = self.age_factor[staying_mask]
        self.N = self.state.shape[0]
        self.leaves_step = int(leaving_mask.sum().item())

        return leaving_shard

    def add_new_individuals(self, new_individuals):
        """
        Add new individuals received from another shard.
        `new_individuals` is a dict with the same keys as leaving_shard.
        """
        if new_individuals is None:
            return  # nothing to add

        for attr in self.tensor_attributes:
            # Obtenemos el tensor actual del agente (self.state, self.mobility...)
            current_tensor = getattr(self, attr)
            
            # Obtenemos el tensor que viene llegando
            incoming_tensor = new_individuals[attr]
            
            # Los unimos. Importante: Ambos deben estar en el mismo dispositivo (CPU)
            new_tensor = torch.cat([current_tensor, incoming_tensor], dim=0)
            # Guardamos el resultado de vuelta en el agente
            setattr(self, attr, new_tensor)
        self.N = self.state.shape[0]
        self.incomes_step = new_individuals['state'].size(0)
        
    def remove_dead(self):
        """
        Remove dead individuals permanently from this shard.
        """
        if self.N == 0:
            return

        # Assuming dead state is 3 (or whichever you choose)
        alive_mask = self.state != 3
        self.state = self.state[alive_mask]
        self.days_in_state = self.days_in_state[alive_mask]
        self.times_infected = self.times_infected[alive_mask]
        self.susceptibility = self.susceptibility[alive_mask]
        self.noncompliance = self.noncompliance[alive_mask]
        self.mobility = self.mobility[alive_mask]
        self.age_factor = self.age_factor[alive_mask]
        self.N = self.state.shape[0]

    def step_progression(self):
        """
        Progresses the state of the population:
        - Increments days in state for infected.
        - Applies per-step death probability for infected individuals.
        - Moves survivors to recovered after recovery_day.
        """
        # Boolean mask of infected individuals
        infected_mask = self.state == 1
        if not infected_mask.any():
            return  # no infected to progress

        # Step 1: Increment days in state for infected
        self.days_in_state[infected_mask] += 1

        # Step 2: Death chance per step (only infected)
        infected_indices = torch.nonzero(infected_mask, as_tuple=True)[0]
        death_prob = self.death_prob * self.age_factor[infected_indices]
        rand_vals = torch.rand(infected_indices.size(0), device=self.device)
        dying_indices = infected_indices[rand_vals < death_prob]

        # Apply deaths in-place
        self.state[dying_indices] = 3
        self.deaths_step = dying_indices.size(0)

        # Step 3: Recovery for infected who survived past recovery_day
        ready_to_recover = infected_indices[self.days_in_state[infected_indices] >= self.recovery_day]
        ready_to_recover = ready_to_recover[self.state[ready_to_recover] != 3]  # exclude dead
        self.state[ready_to_recover] = 2

    def to_gpu(self):
        for attr in self.tensor_attributes:
            # getattr obtiene el valor, .to() lo mueve, setattr lo guarda de nuevo
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(self.device))
        self.device = self.device

    def to_cpu(self):
        for attr in self.tensor_attributes:
            # getattr obtiene el valor, .to() lo mueve, setattr lo guarda de nuevo
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(self.cpu))
        self.device = self.cpu
        torch.cuda.empty_cache()

    def step(self):
        self.to_gpu()
        self.step_infection()     # spread disease
        self.step_progression()        # healing, state updates
        self.remove_dead()        # permanently remove deaths
        leaving = self.step_leaving()  # remove leaving individuals
        self.to_cpu()
        return leaving
        


