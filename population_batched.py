

import torch
from tqdm import tqdm
import time
import psutil
class PopulationShardTorch:
    def __init__(self, N, node_id, seed=42, device='cuda'):
        self.device = device
        self.N = N
        self.node_id = node_id

        # Random generator (CPU)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

        # --- Datos de cálculo de infección (CPU) ---
        self.susceptibility = (0.7 + 0.3 * torch.rand(N, device=device, generator=self.rng)).to(torch.float16)
        # Cumplimiento de normas, mascarillas, lockdown
        self.noncompliance = (0.2 + 0.8 * torch.rand(N, device=device, generator=self.rng)).to(torch.float16)
        # Movilidad interna: ajusta número de contactos diarios
        self.mobility = (0.5 + 0.5 * torch.rand(N, device=device, generator=self.rng)).to(torch.float16)

        # --- Datos informativos / progresión (CPU) ---
        self.state = torch.zeros(N, dtype=torch.uint8,device=device)  # 0=S, 1=I, 2=R, 3=M, 4=V
        self.days_in_state = torch.zeros(N, dtype=torch.uint8,device=device)
        self.times_infected = torch.zeros(N, dtype=torch.uint8,device=device)
        self.age_factor = (0.1 + 2 * torch.rand(N, dtype=torch.float16,device=device))

        # --- Parámetros de simulación ---
        self.contacts_per_day = 30
        self.P_base = 0.1
        self.variant_factor = 1.0
        self.mask_factor = 0.5
        self.lockdown_factor = 0.4
        self.min_days_infected = 5
        self.max_days_infected = 14
        self.recovery_day = 7
        self.death_prob = 0.01
        self.leave_prob = 1e-4
        self.deaths = 0

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
        contacts_gpu = susceptibles[torch.randint(0, len(susceptibles), (n_total_contacts,), device='cpu')]
        return contacts_gpu  # regresa a CPU

    def infect_contacts(self, contacts_indices):
        if len(contacts_indices) == 0:
            return

        P_contact = self.P_base * self.susceptibility[contacts_indices] * self.noncompliance[contacts_indices] * self.variant_factor * self.mask_factor

        rand = torch.rand(len(contacts_indices), device=self.device)

        new_infections = rand < P_contact
        indices_to_infect = contacts_indices[new_infections]

        # Actualiza en CPU
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
            'state': self.state[leaving_mask].clone(),
            'days_in_state': self.days_in_state[leaving_mask].clone(),
            'times_infected': self.times_infected[leaving_mask].clone(),
            'susceptibility': self.susceptibility[leaving_mask].clone(),
            'noncompliance': self.noncompliance[leaving_mask].clone(),
            'mobility': self.mobility[leaving_mask].clone(),
            'age_factor': self.age_factor[leaving_mask].clone(),
            'N': int(leaving_mask.sum().item())
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

        return leaving_shard

    def add_new_individuals(self, new_individuals):
        """
        Add new individuals received from another shard.
        `new_individuals` is a dict with the same keys as leaving_shard.
        """
        if new_individuals is None or new_individuals['N'] == 0:
            return  # nothing to add

        self.state = torch.cat([self.state, new_individuals['state']], dim=0)
        self.days_in_state = torch.cat([self.days_in_state, new_individuals['days_in_state']], dim=0)
        self.times_infected = torch.cat([self.times_infected, new_individuals['times_infected']], dim=0)
        self.susceptibility = torch.cat([self.susceptibility, new_individuals['susceptibility']], dim=0)
        self.noncompliance = torch.cat([self.noncompliance, new_individuals['noncompliance']], dim=0)
        self.mobility = torch.cat([self.mobility, new_individuals['mobility']], dim=0)
        self.age_factor = torch.cat([self.age_factor, new_individuals['age_factor']], dim=0)
        self.N = self.state.shape[0]

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
        self.deaths += dying_indices.size(0)

        # Step 3: Recovery for infected who survived past recovery_day
        ready_to_recover = infected_indices[self.days_in_state[infected_indices] >= self.recovery_day]
        ready_to_recover = ready_to_recover[self.state[ready_to_recover] != 3]  # exclude dead
        self.state[ready_to_recover] = 2


    def step(self):
        self.step_infection()     # spread disease
        self.step_progression()        # healing, state updates
        self.remove_dead()        # permanently remove deaths
        leaving = self.step_leaving()  # remove leaving individuals
        return leaving
    
def check_resources(N:int):
    RAM_THRESHOLD = 0.8   # 80% of system RAM
    GPU_THRESHOLD = 0.8   # 80% of GPU memory

    # --- Fields and dtypes ---
    dtype_sizes = {
        'uint8': 1,
        'float16': 2,
        'float32': 4
    }

    fields = {
        'state': 'uint8',
        'days_in_state': 'uint8',
        'times_infected': 'uint8',
        'susceptibility': 'float16',
        'noncompliance': 'float16',
        'mobility': 'float16',
        'age_factor': 'float16'
    }

    # --- Compute estimated memory per shard ---
    total_bytes = N * sum(dtype_sizes[dtype] for dtype in fields.values())
    total_MB = total_bytes / 1e6
    total_GB = total_bytes / 1e9
    print(f"Estimated memory per shard: {total_MB:.2f} MB ({total_GB:.2f} GB)")

    # --- Check system RAM ---
    mem = psutil.virtual_memory()
    available_RAM_GB = mem.available / 1e9
    if total_GB > available_RAM_GB * RAM_THRESHOLD:
        raise MemoryError(f"Shard requires {total_GB:.2f} GB, which exceeds "
                        f"{RAM_THRESHOLD*100}% of available system RAM ({available_RAM_GB:.2f} GB).")

    print(f"Available system RAM: {available_RAM_GB:.2f} GB - OK")

    # --- Check GPU memory if available ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_GPU_GB = torch.cuda.memory_allocated(0) / 1e9
        free_GPU_GB = total_GPU_GB - allocated_GPU_GB

        if total_GB > free_GPU_GB * GPU_THRESHOLD:
            raise MemoryError(f"Shard requires {total_GB:.2f} GB, which exceeds "
                            f"{GPU_THRESHOLD*100}% of free GPU memory ({free_GPU_GB:.2f} GB).")
        
        print(f"Free GPU memory: {free_GPU_GB:.2f} GB - OK")
    return total_GB > available_RAM_GB * RAM_THRESHOLD


if __name__ == '__main__':
    N = int(48e6)  # prueba con 1 millón (48M puede ser excesivo)

    if check_resources(N):
        raise RuntimeError 
    
    # Compute total memory (in MB)
    node_id = 1
    seed = 42
    initial_infected = 5
    steps = 100
    device = 'cuda'

    shard = PopulationShardTorch(N=N, node_id=node_id, seed=seed, device=device)

    infected_indices = torch.randint(0, N, (initial_infected,))
    shard.state[infected_indices] = 1
    shard.days_in_state[infected_indices] = 0

    pbar = tqdm(range(1, steps + 1))

    for step in pbar:
        shard.step()
        # 0=S, 1=I, 2=R, 3=M, 4=V
        num_infected = int((shard.state == 1).sum().item())
        recuperados = int((shard.state == 2).sum().item())
        death = int(shard.deaths)
        alive = shard.state.size(0)
        pbar.set_postfix({'Paso': step, 'Infectados': num_infected, 'Alive': alive, 'deaths':death})
    print(death)
