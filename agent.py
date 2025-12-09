from mesa import Agent
import math
import random

class Person(Agent):
    """
    Epidemic agent representing a person.

    States:
        - "S": Susceptible
        - "V": Vaccinated
        - "I": Infected
        - "R": Recovered
        - "M": Dead
    """

    def __init__(
        self, 
        model, 
        unique_id, 
        state="S", 
        age_group=30, 
        compliance=1.0
    ):
        super().__init__(model)
        self.unique_id = unique_id
        self.state = state
        self.age_group = age_group
        self.immunity = 0.0
        self.times_infected = 0
        self.days_infected = 0
        self.days_since_immunity = 0
        self.compliance = compliance  # 0 = never obeys rules, 1 = always obeys
        self.pos = None  # (x, y) inside node if using intra-node grid
        self.node_index = None  # Current node / municipality
        self.cell_index = None  # Optional: intra-node position
        # Event-buffer attributes
        self.next_state = state
        self.next_node = None
        self.next_cell = None

    # ----------------------------
    # Movement Logic
    # ----------------------------
    def try_move(self):
        # Inter-node travel
        travel_prob = self.model.travel_graph.get(self.node_index, 0.0)
        travel_prob *= self.compliance * self.model.travel_restriction_factor.get(self.node_index, 1.0)
        if random.random() < travel_prob:
            # Sample target node
            target_node = self.model.sample_target_node(self.node_index)
            self.next_node = target_node

    # ----------------------------
    # Infection Dynamics
    # ----------------------------
    def try_infect(self):

        # Sample daily contacts
        internal_mobility = self.model.internal_mobility_factor
        num_contacts = max(1, int(self.model.contacts_per_day * internal_mobility))

        variant = self.model.current_variant
        variant_effect = variant.transmissibility_multiplier 
        mask_effect = (1 - self.model.mask_effectiveness) * self.compliance
        base_prob = self.model.infection_prob * variant_effect * mask_effect

        contacts = self.model.sample_susceptible_agent(self.node_index,num_contacts, exclude_infected=True)

        for contact in contacts: 
            effective_prob = base_prob * (1 - contact.immunity * (1 - variant.immune_escape_factor))
            if random.random() < effective_prob:
                contact.next_state = "I"
                contact.days_infected = 0 
                contact.immunity = 0.0
                contact.times_infected += 1

    # ----------------------------
    # Recovery and Mortality
    # ----------------------------
    def update_health(self):
        self.days_infected += 1
        age_factor = 1.0 + max(0, (self.age_group - 50)/100)
        if random.random() < (self.model.mortality_prob * age_factor):
            self.next_state = "M"
            self.model.space.remove_agent(self)
            return

        if self.days_infected >= self.model.recovery_time:
            self.next_state = "R"
            self.immunity = 1.0
            self.days_since_immunity = 0

    # ----------------------------
    # Waning Immunity
    # ----------------------------
    def decay_immunity(self):
        if self.state in ("R", "V") and self.immunity > 0.0:
            decay_factor = math.exp(-1 / self.model.immunity_decay_days)
            self.immunity *= decay_factor
            self.days_since_immunity += 1
            if self.immunity < 0.05:
                self.immunity = 0.0

    # ----------------------------
    # Vaccination
    # ----------------------------
    def apply_vaccination(self, efficacy=0.8):
        if self.state in ("S", "R") and random.random() < efficacy:
            self.immunity = max(self.immunity, efficacy)
            self.next_state = "V"

    # ----------------------------
    # Step function
    # ----------------------------
    def step(self):
        self.try_move()
        self.decay_immunity()

    def step_infected(self):
        self.try_infect()
        self.update_health()