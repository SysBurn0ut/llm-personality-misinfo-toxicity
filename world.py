import mesa
from tqdm import tqdm
from citizen import Citizen
from personas import build_personas
from utils import clear_cache


class World(mesa.Model):
    '''
    Same persona (name, age, qualification and traits) replicated on all models.
    Population = n_personas * n_models.
    '''

    def __init__(self, args, model_list):
        super().__init__()

        self.n_personas = args.n_personas
        self.model_list = model_list
        self.population = self.n_personas * len(model_list)
        self.step_count = args.no_days
        self.name = args.name

        self.current_step = 0
        self.current_headline = ""

        personas = build_personas(self.n_personas, seed=args.seed, mode=args.persona_mode, margin=args.pole_margin)

        uid = 0
        for p in personas:
            for model_string in model_list:
                citizen = Citizen(
                    model=self,
                    unique_id=uid,
                    persona=p,
                    model_string=model_string,
                )
                self.agents.add(citizen)
                uid += 1

        print(f"[INIT] {self.n_personas} people x {len(model_list)} models "
              f"= {self.population} agents")

    def step(self):
        for citizen in tqdm(list(self.agents),
                            desc=f"Step {self.current_step + 1}"):
            citizen.step()
        self.current_step += 1

    def run_model(self, headline_list):
        for i, headline in enumerate(headline_list):
            if i >= self.step_count:
                break
            print(f"\nHeadline {i + 1}/{min(len(headline_list), self.step_count)}")
            self.current_headline = headline
            self.step()
            clear_cache()
