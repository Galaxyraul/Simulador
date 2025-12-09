import matplotlib.pyplot as plt
from model import EpidemicModel
from tqdm import tqdm
def run_example():
    model = EpidemicModel(
        N=int(10e4),
        width=50,
        height=50,
        initial_infected=5,
        infection_prob=0.05,
        infection_radius=1.5,
        recovery_time=8,
        mortality_prob=0.002,
        initial_vaccinated_frac=0.10,
        vaccine_effectiveness=0.85,
        daily_vaccinations=5,
        vacc_start_day=5,
        move_prob=0.9,
        step_size=1.0,
        mask_effectiveness=0.2,
        seed=42
    )

    steps = 120
    for _ in tqdm(range(steps),desc='Simulation',total=steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()
    print(df.tail())

    plt.figure(figsize=(10,6))
    plt.plot(df["Susceptible"], label="Susceptible")
    plt.plot(df["Vaccinated"], label="Vaccinated")
    plt.plot(df["Infected"], label="Infected")
    plt.plot(df["Recovered"], label="Recovered")
    plt.plot(df["Dead"], label="Dead")
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title("Epidemic simulation (S,V,I,R,Dead)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_example()
