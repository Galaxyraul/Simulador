import numpy as np
from tqdm import tqdm

class PopulationShard:
    """
    Shard de población para un nodo, usando Structure of Arrays (SoA)
    para máxima eficiencia y vectorización.
    """

    def __init__(self, N, node_id, seed=None):
        self.N = N
        self.node_id = node_id  # único para todo el shard

        # RNG reproducible
        self.rng = np.random.default_rng(seed)

        # --- Datos de cálculo de infección ---
        # Susceptibilidad individual: predisposición genética / inmunidad inicial
        self.susceptibility = (0.7 + 0.3 * self.rng.random(N)).astype(np.float16)
        # Cumplimiento de normas, mascarillas, lockdown
        self.noncompliance = (0.2 + 0.8 * self.rng.random(N)).astype(np.float16)
        # Movilidad interna: ajusta número de contactos diarios
        self.mobility = np.ones(N, dtype=np.float16)

        # --- Datos informativos / progresión ---
        self.state = np.zeros(N, dtype=np.uint8)          # 0=S, 1=I, 2=R, 3=M, 4=V
        self.days_in_state = np.zeros(N, dtype=np.uint8)
        self.times_infected = np.zeros(N, dtype=np.uint8)
        self.age_factor = np.ones(N, dtype=np.float16)

        # --- Parámetros de simulación ---
        self.contacts_per_day = 30           # base de contactos diarios
        self.P_base = 0.05                   # probabilidad base de infección
        self.variant_factor = 1.0            # transmissibilidad variante actual
        self.mask_factor = 1.0               # efectividad de mascarilla global
        self.lockdown_factor = 0.0           # fuerza del lockdown (0-1)

    def sample_contacts(self, infected_indices):
        susceptibles = np.where((self.state != 1) & (self.state != 3))[0]
        n_contagios = (
            self.contacts_per_day *
            self.noncompliance[infected_indices] *
            (1 - self.lockdown_factor) *
            self.mobility[infected_indices]
        ).astype(np.uint64)
        print(n_contagios.size)
        print(n_contagios)
        # Evitar NaN

        # Convertir a entero
        n_contagios = int(np.floor(n_contagios.sum()))

        return self.rng.choice(
            susceptibles,
            size=n_contagios,
            replace=True
        ).astype(np.int32)


    def infect_contacts(self, contacts_indices):
        """
        Aplica la probabilidad de infección para cada contacto.
        """
        if len(contacts_indices) == 0:
            return

        P_contact = (self.P_base *
                     self.susceptibility[contacts_indices] *
                     self.noncompliance[contacts_indices] *
                     self.variant_factor *
                     self.mask_factor)

        rand = self.rng.random(size=len(P_contact))
        new_infections = rand < P_contact

        # Actualizamos estado y días infectado solo si se infecta
        indices_to_infect = contacts_indices[new_infections]
        self.state[indices_to_infect] = 1  # I
        self.days_in_state[indices_to_infect] = 0
        self.times_infected[indices_to_infect] += 1
        # Reinicia susceptibilidad si se desea modelar pérdida de inmunidad después
        self.susceptibility[indices_to_infect] = 1.0

    def step_infection(self):
        """
        Ejecuta un tick de infección completo.
        """
        # Identificar todos los infectados
        infected_indices = np.where(self.state == 1)[0]
        # Muestreo de contactos
        contacts = self.sample_contacts(infected_indices)
        # Aplicar contagios
        self.infect_contacts(contacts)



if __name__ == '__main__':
    # --- Configuración ---
    N = int(48e6)          # número de personas en el nodo
    node_id = 1       # identificador del nodo
    seed = 42         # reproducibilidad

    # Crear el shard
    shard = PopulationShard(N=N, node_id=node_id, seed=seed)

    # Inicializar algunos infectados
    initial_infected = 5
    infected_indices = shard.rng.choice(N, size=initial_infected, replace=False)
    shard.state[infected_indices] = 1  # I
    shard.days_in_state[infected_indices] = 0

    print(f"Paso 0: Infected={initial_infected}")

    # --- Simulación de 10 pasos ---
    for step in tqdm(range(1, 100)):
        shard.step_infection()
        num_infected = np.sum(shard.state == 1)
        print(f"Paso {step}: Infected={num_infected}")

