from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from agent import Person
class EpidemicModel(Model):
    def __init__(self,
                 N=200,
                 width=50,
                 height=50,
                 initial_infected=5,
                 infection_prob=0.03,
                 infection_radius=1.5,
                 recovery_time=10,
                 mortality_prob=0.002,
                 initial_vaccinated_frac=0.0,
                 vaccine_effectiveness=0.8,
                 daily_vaccinations=0,
                 vacc_start_day=10,
                 move_prob=0.8,
                 step_size=1.0,
                 mask_effectiveness=0.0,
                 seed=33):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.N = N
        self.infection_prob = infection_prob
        self.infection_radius = infection_radius
        self.recovery_time = recovery_time
        self.mortality_prob = mortality_prob
        self.vaccine_effectiveness = vaccine_effectiveness
        self.daily_vaccinations = daily_vaccinations
        self.vacc_start_day = vacc_start_day
        self.move_prob = move_prob
        self.step_size = step_size
        self.mask_effectiveness = mask_effectiveness

        self.steps = 0
        self.space = ContinuousSpace(width, height, torus=True)
        for i in range(self.N):
            age = int(self.random.uniform(10, 80))
            if i < initial_infected:
                state = "I"
            elif self.random.random() < initial_vaccinated_frac:
                state = "V"
            else:
                state = "S"
            agent = Person(model=self, unique_id=i, state=state, age=age)
            if state == "V":
                agent.immunity = self.vaccine_effectiveness
            self.agents.add(agent)
            x = self.random.random() * width
            y = self.random.random() * height
            self.space.place_agent(agent, (x, y))

        self.datacollector = DataCollector({
            "Susceptible": lambda m: sum(1 for a in m.agents if a.state == "S"),
            "Vaccinated": lambda m: sum(1 for a in m.agents if a.state == "V"),
            "Infected": lambda m: sum(1 for a in m.agents if a.state == "I"),
            "Recovered": lambda m: sum(1 for a in m.agents if a.state == "R"),
            "Dead": lambda m: sum(1 for a in m.agents if a.state == "M"),
            "TotalAlive": lambda m: sum(1 for a in m.agents if a.state != "M"),
        })
        self.datacollector.collect(self)

    def vaccinate_daily(self):
        if self.steps < self.vacc_start_day:
            return
        if self.daily_vaccinations <= 0:
            return
        susceptibles = [a for a in self.agents if a.state == "S"]
        self.random.shuffle(susceptibles)
        to_vaccinate = susceptibles[:self.daily_vaccinations]
        for a in to_vaccinate:
            a.state = "V"
            a.immunity = self.vaccine_effectiveness

    def step(self):
        self.vaccinate_daily()
        self.agents.shuffle()
        for agent in list(self.agents):
            agent.step()
        self.datacollector.collect(self)
        self.steps += 1
if __name__ == '__main__':
    model = EpidemicModel(seed=33)