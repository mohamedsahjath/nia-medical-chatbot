from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class GAResult:
    mask: np.ndarray
    best_fitness: float
    selected_features: int
    total_features: int
    generations: int
    history: list[float]


class GeneticFeatureSelector:
    """Binary Genetic Algorithm for selecting useful TF-IDF features."""

    def __init__(self, population_size=10, generations=5, mutation_rate=0.03,
                 crossover_rate=0.85, random_state=42):
        self.population_size = max(6, population_size)
        self.generations = max(1, generations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(random_state)

    def fit(self, x, y, mandatory_mask=None):
        feature_count = x.shape[1]
        self.mandatory_mask = (
            np.asarray(mandatory_mask, dtype=bool)
            if mandatory_mask is not None else np.zeros(feature_count, dtype=bool)
        )
        population = self.rng.random((self.population_size, feature_count)) < 0.65
        self._repair(population)
        history = []
        for _ in range(self.generations):
            fitness = np.array([self._fitness(x, y, mask) for mask in population])
            history.append(float(fitness.max()))
            next_population = [population[i].copy() for i in fitness.argsort()[-2:]]
            while len(next_population) < self.population_size:
                first = self._tournament(population, fitness)
                second = self._tournament(population, fitness)
                child_a, child_b = self._crossover(first, second)
                next_population.extend([self._mutate(child_a), self._mutate(child_b)])
            population = np.array(next_population[:self.population_size])
            self._repair(population)
        fitness = np.array([self._fitness(x, y, mask) for mask in population])
        best = int(fitness.argmax())
        mask = population[best].copy()
        return GAResult(mask, float(fitness[best]), int(mask.sum()), feature_count,
                        self.generations, history + [float(fitness.max())])

    def _fitness(self, x, y, mask):
        if mask.sum() < 2:
            return 0.0
        labels = np.asarray(y)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in splitter.split(x, labels):
            model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
            model.fit(x[train_index][:, mask], labels[train_index])
            prediction = model.predict(x[test_index][:, mask])
            scores.append(f1_score(labels[test_index], prediction, average="macro", zero_division=0))
        return float(np.mean(scores) + 0.02 * (1.0 - mask.mean()))

    def _tournament(self, population, fitness):
        indexes = self.rng.choice(len(population), size=3, replace=False)
        return population[indexes[np.argmax(fitness[indexes])]].copy()

    def _crossover(self, first, second):
        if self.rng.random() >= self.crossover_rate or len(first) < 2:
            return first.copy(), second.copy()
        point = int(self.rng.integers(1, len(first)))
        return (np.concatenate([first[:point], second[point:]]),
                np.concatenate([second[:point], first[point:]]))

    def _mutate(self, chromosome):
        return np.logical_xor(chromosome,
                              self.rng.random(len(chromosome)) < self.mutation_rate)

    def _repair(self, population):
        minimum = min(5, population.shape[1])
        for chromosome in population:
            chromosome[self.mandatory_mask] = True
            if chromosome.sum() < minimum:
                indexes = self.rng.choice(population.shape[1], size=minimum, replace=False)
                chromosome[indexes] = True
