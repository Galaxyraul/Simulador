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
        self.state = torch.zeros(N, dtype=torch.uint8,device=self.device)  # 0=S, 1=I, 2=R, 3=M, 4=V
        self.days_in_state = torch.zeros(N, dtype=torch.uint8,device=self.device)
        self.times_infected = torch.zeros(N, dtype=torch.uint8,device=self.device)
        self.age_factor = torch.ones(N, dtype=torch.float16,device=self.device)

        # --- Parámetros de simulación ---
        self.contacts_per_day = 30
        self.P_base = 0.05
        self.variant_factor = 1.0
        self.mask_factor = 1.0
        self.lockdown_factor = 0.0

    def sample_contacts(self, infected_indices):
        # Mueve los arrays a GPU temporalmente
        state_gpu = self.state.to(self.device)
        noncompliance_gpu = self.noncompliance.to(self.device)
        mobility_gpu = self.mobility.to(self.device)

        susceptibles = torch.nonzero((state_gpu != 1) & (state_gpu != 3)).squeeze()
        if len(susceptibles) == 0 or len(infected_indices) == 0:
            return torch.tensor([], dtype=torch.long)

        infected_indices_gpu = infected_indices

        n_contacts_per_infected = (self.contacts_per_day *
                                   noncompliance_gpu[infected_indices_gpu] *
                                   (1 - self.lockdown_factor) *
                                   mobility_gpu[infected_indices_gpu]).to(torch.uint16)
        n_total_contacts = int(n_contacts_per_infected.sum().item())
        contacts_gpu = susceptibles[torch.randint(0, len(susceptibles), (n_total_contacts,), device='cpu')]
        return contacts_gpu  # regresa a CPU

    def infect_contacts(self, contacts_indices):
        if len(contacts_indices) == 0:
            return

        # Mueve solo los tensores necesarios a GPU
        sus_gpu = self.susceptibility[contacts_indices].to(self.device)
        nonc_gpu = self.noncompliance[contacts_indices].to(self.device)

        P_contact = self.P_base * sus_gpu * nonc_gpu * self.variant_factor * self.mask_factor

        rand = torch.rand(len(contacts_indices), device=self.device)

        new_infections = rand < P_contact
        indices_to_infect = contacts_indices[new_infections.cpu()]

        # Actualiza en CPU
        self.state[indices_to_infect] = 1
        self.days_in_state[indices_to_infect] = 0
        self.times_infected[indices_to_infect] += 1
        self.susceptibility[indices_to_infect] = 1.0

    def step_infection(self):
        infected_indices = torch.nonzero(self.state == 1).squeeze().to('cpu')
        contacts = self.sample_contacts(infected_indices)
        self.infect_contacts(contacts)


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
        start = time.perf_counter()

        shard.step_infection()

        end = time.perf_counter()
        iter_time = end - start
        num_infected = int((shard.state == 1).sum().item())

        pbar.set_postfix({'Paso': step, 'Infectados': num_infected, 'iter_time (s)': f'{iter_time:.3f}'})
