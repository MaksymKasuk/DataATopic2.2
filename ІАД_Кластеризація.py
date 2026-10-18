"""Практична робота 6"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "soft_computing_results"


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def run_kmeans_example() -> dict:
    """Кластеризація Iris і кількісна оцінка результату."""
    iris = load_iris(as_frame=True)
    x = iris.data.copy()
    y = iris.target.to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    rows = []
    fitted = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_SEED)
        labels = model.fit_predict(x_scaled)
        fitted[k] = (model, labels)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(x_scaled, labels),
            "adjusted_rand": adjusted_rand_score(y, labels),
        })

    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(RESULTS_DIR / "kmeans_evaluation.csv", index=False)

    model, labels = fitted[3]
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    centers = pd.DataFrame(centers_original, columns=x.columns)
    centers.insert(0, "cluster", np.arange(3))
    centers.to_csv(RESULTS_DIR / "kmeans_centers.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x_scaled)
    centers_pca = pca.transform(model.cluster_centers_)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(evaluation["k"], evaluation["inertia"], marker="o", color="#2C7FB8")
    axes[0].set(title="Метод ліктя", xlabel="Кількість кластерів k", ylabel="Внутрішньокластерна сума квадратів")
    axes[1].plot(evaluation["k"], evaluation["silhouette"], marker="o", color="#D95F0E")
    axes[1].axvline(3, color="gray", linestyle=":", linewidth=1)
    axes[1].set(title="Силуетний коефіцієнт", xlabel="Кількість кластерів k", ylabel="Середній силует")
    save_figure(fig, "figure_1_kmeans_selection.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap="viridis", s=42, alpha=0.82)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=190, edgecolor="white", label="Центроїди")
    ax.set(title="Кластери k means у просторі двох головних компонент", xlabel="Перша головна компонента", ylabel="Друга головна компонента")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Номер кластера")
    save_figure(fig, "figure_2_kmeans_clusters.png")

    return {
        "x": x,
        "x_scaled": x_scaled,
        "y": y,
        "target_names": iris.target_names,
        "labels": labels,
        "evaluation": evaluation,
        "explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
    }


class KohonenSOM:
    """Мінімальна реалізація двовимірної самоорганізовної карти."""

    def __init__(self, rows: int, cols: int, n_features: int, seed: int = RANDOM_SEED):
        self.rows = rows
        self.cols = cols
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.6, size=(rows, cols, n_features))
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([rr, cc], axis=-1)
        self.quantization_history: list[float] = []

    def winner(self, sample: np.ndarray) -> tuple[int, int]:
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return tuple(np.unravel_index(np.argmin(distances), distances.shape))

    def train(self, x: np.ndarray, epochs: int = 120, learning_rate: float = 0.45) -> None:
        initial_radius = max(self.rows, self.cols) / 2
        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            lr = learning_rate * math.exp(-2.5 * progress)
            radius = max(0.55, initial_radius * math.exp(-2.8 * progress))
            for idx in self.rng.permutation(len(x)):
                sample = x[idx]
                bmu = np.asarray(self.winner(sample))
                grid_d2 = np.sum((self.grid - bmu) ** 2, axis=2)
                influence = np.exp(-grid_d2 / (2 * radius ** 2))[..., None]
                self.weights += lr * influence * (sample - self.weights)
            self.quantization_history.append(self.quantization_error(x))

    def quantization_error(self, x: np.ndarray) -> float:
        return float(np.mean([np.linalg.norm(sample - self.weights[self.winner(sample)]) for sample in x]))

    def topographic_error(self, x: np.ndarray) -> float:
        errors = 0
        flat = self.weights.reshape(-1, self.n_features)
        for sample in x:
            order = np.argsort(np.linalg.norm(flat - sample, axis=1))[:2]
            a = np.array(np.unravel_index(order[0], (self.rows, self.cols)))
            b = np.array(np.unravel_index(order[1], (self.rows, self.cols)))
            if np.max(np.abs(a - b)) > 1:
                errors += 1
        return errors / len(x)

    def u_matrix(self) -> np.ndarray:
        umat = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                distances = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        distances.append(np.linalg.norm(self.weights[r, c] - self.weights[nr, nc]))
                umat[r, c] = np.mean(distances)
        return umat


def run_som_example(x_scaled: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> dict:
    som = KohonenSOM(8, 8, x_scaled.shape[1])
    som.train(x_scaled, epochs=140, learning_rate=0.45)
    bmus = np.array([som.winner(sample) for sample in x_scaled])

    hits = np.zeros((som.rows, som.cols), dtype=int)
    label_map: dict[tuple[int, int], int] = {}
    for r, c in bmus:
        hits[r, c] += 1
    for r in range(som.rows):
        for c in range(som.cols):
            members = y[(bmus[:, 0] == r) & (bmus[:, 1] == c)]
            if len(members):
                label_map[(r, c)] = Counter(members).most_common(1)[0][0]

    correct = sum(label_map[tuple(bmus[i])] == y[i] for i in range(len(y)))
    purity = correct / len(y)
    quantization_error = som.quantization_error(x_scaled)
    topographic_error = som.topographic_error(x_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sns.heatmap(som.u_matrix(), cmap="mako", ax=axes[0], square=True, cbar_kws={"label": "Середня відстань"})
    axes[0].set(title="U матриця карти Кохонена", xlabel="Стовпець нейрона", ylabel="Рядок нейрона")
    sns.heatmap(hits, cmap="YlGnBu", annot=True, fmt="d", ax=axes[1], square=True, cbar_kws={"label": "Кількість об'єктів"})
    axes[1].set(title="Карта щільності попадань", xlabel="Стовпець нейрона", ylabel="Рядок нейрона")
    save_figure(fig, "figure_3_som_maps.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(np.arange(1, len(som.quantization_history) + 1), som.quantization_history, color="#7A0177")
    ax.set(title="Навчання карти Кохонена", xlabel="Епоха", ylabel="Квантизаційна помилка")
    save_figure(fig, "figure_4_som_learning.png")

    neuron_rows = []
    for (r, c), label in sorted(label_map.items()):
        neuron_rows.append({"row": r, "column": c, "hits": int(hits[r, c]), "dominant_class": str(target_names[label])})
    pd.DataFrame(neuron_rows).to_csv(RESULTS_DIR / "som_neurons.csv", index=False)
    return {
        "quantization_error": quantization_error,
        "topographic_error": topographic_error,
        "purity": purity,
        "occupied_neurons": len(label_map),
    }


def triangular_membership(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    left = (values - a) / max(b - a, 1e-12)
    right = (c - values) / max(c - b, 1e-12)
    return np.maximum(0, np.minimum(left, right))


def fuzzy_fan_speed(temperature: float) -> tuple[float, dict[str, float]]:
    universe = np.linspace(0, 100, 1001)
    cold = float(triangular_membership(temperature, 0, 10, 22))
    comfortable = float(triangular_membership(temperature, 18, 24, 30))
    hot = float(triangular_membership(temperature, 26, 35, 45))
    low = triangular_membership(universe, 0, 20, 45)
    medium = triangular_membership(universe, 30, 55, 80)
    high = triangular_membership(universe, 65, 85, 100)
    aggregated = np.maximum.reduce([
        np.minimum(cold, low),
        np.minimum(comfortable, medium),
        np.minimum(hot, high),
    ])
    crisp = float(np.trapezoid(universe * aggregated, universe) / np.trapezoid(aggregated, universe))
    return crisp, {"cold": cold, "comfortable": comfortable, "hot": hot}


def run_fuzzy_example() -> dict:
    temperatures = np.linspace(0, 45, 301)
    cold = triangular_membership(temperatures, 0, 10, 22)
    comfortable = triangular_membership(temperatures, 18, 24, 30)
    hot = triangular_membership(temperatures, 26, 35, 45)
    evaluated = [fuzzy_fan_speed(float(t))[0] for t in np.linspace(12, 40, 100)]
    target_temperature = 29.0
    speed, memberships = fuzzy_fan_speed(target_temperature)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(temperatures, cold, label="Холодно")
    axes[0].plot(temperatures, comfortable, label="Комфортно")
    axes[0].plot(temperatures, hot, label="Спекотно")
    axes[0].axvline(target_temperature, color="black", linestyle=":")
    axes[0].set(title="Нечіткі множини температури", xlabel="Температура, °C", ylabel="Ступінь належності")
    axes[0].legend()
    axes[1].plot(np.linspace(12, 40, 100), evaluated, color="#E6550D")
    axes[1].scatter([target_temperature], [speed], color="black", zorder=4)
    axes[1].set(title="Нечітке керування вентилятором", xlabel="Температура, °C", ylabel="Швидкість, %")
    save_figure(fig, "figure_5_fuzzy_control.png")
    return {"temperature": target_temperature, "fan_speed": speed, "memberships": memberships}


def objective(x: np.ndarray) -> np.ndarray:
    return x * np.sin(10 * np.pi * x) + 1.0


def run_genetic_example() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    population = rng.uniform(0, 1, 80)
    history = []
    for _ in range(100):
        fitness = objective(population)
        history.append(float(fitness.max()))
        order = np.argsort(fitness)[::-1]
        elite = population[order[:8]]
        shifted = fitness - fitness.min() + 1e-9
        probabilities = shifted / shifted.sum()
        children = list(elite)
        while len(children) < len(population):
            p1, p2 = rng.choice(population, size=2, replace=True, p=probabilities)
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            if rng.random() < 0.18:
                child += rng.normal(0, 0.035)
            children.append(float(np.clip(child, 0, 1)))
        population = np.asarray(children)
    fitness = objective(population)
    best_idx = int(np.argmax(fitness))
    best_x = float(population[best_idx])
    best_value = float(fitness[best_idx])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(np.arange(1, len(history) + 1), history, color="#238B45")
    ax.set(title="Збіжність генетичного алгоритму", xlabel="Покоління", ylabel="Найкраще значення функції")
    save_figure(fig, "figure_6_genetic_algorithm.png")
    return {"best_x": best_x, "best_value": best_value}


CITIES = {
    "Київ": (50.4501, 30.5234),
    "Львів": (49.8397, 24.0297),
    "Одеса": (46.4825, 30.7233),
    "Харків": (49.9935, 36.2304),
    "Дніпро": (48.4647, 35.0462),
    "Запоріжжя": (47.8388, 35.1396),
    "Вінниця": (49.2331, 28.4682),
    "Полтава": (49.5883, 34.5514),
    "Чернігів": (51.4982, 31.2893),
    "Черкаси": (49.4444, 32.0598),
    "Тернопіль": (49.5535, 25.5948),
    "Івано Франківськ": (48.9226, 24.7111),
}


def haversine_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


def distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        matrix[i, j] = matrix[j, i] = haversine_distance(coords[i], coords[j])
    return matrix


def route_length(route: list[int] | np.ndarray, distances: np.ndarray) -> float:
    route = list(route)
    return float(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


def nearest_neighbor_tour(distances: np.ndarray, start: int = 0) -> list[int]:
    unvisited = set(range(len(distances))) - {start}
    route = [start]
    while unvisited:
        current = route[-1]
        nxt = min(unvisited, key=lambda j: distances[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
    return route


def held_karp(distances: np.ndarray) -> tuple[float, list[int]]:
    """Точний динамічний алгоритм для порівняння на 12 містах."""
    n = len(distances)
    dp: dict[tuple[int, int], tuple[float, int]] = {}
    for k in range(1, n):
        dp[(1 << k, k)] = (distances[0, k], 0)
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            bits = sum(1 << k for k in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                candidates = [(dp[(prev_bits, m)][0] + distances[m, k], m) for m in subset if m != k]
                dp[(bits, k)] = min(candidates)
    bits = sum(1 << k for k in range(1, n))
    total, last = min((dp[(bits, k)][0] + distances[k, 0], k) for k in range(1, n))
    reverse_route = [last]
    current_bits = bits
    while len(reverse_route) < n - 1:
        _, previous = dp[(current_bits, last)]
        current_bits &= ~(1 << last)
        last = previous
        if last != 0:
            reverse_route.append(last)
    route = [0] + list(reversed(reverse_route))
    return float(total), route


def ant_colony_tsp(distances: np.ndarray, n_ants: int = 50, iterations: int = 220,
                   alpha: float = 1.0, beta: float = 4.0, evaporation: float = 0.35) -> tuple[list[int], float, list[float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(distances)
    pheromone = np.ones((n, n), dtype=float)
    heuristic = 1 / np.maximum(distances, 1e-12)
    np.fill_diagonal(heuristic, 0)
    best_route: list[int] | None = None
    best_length = float("inf")
    history = []

    for _ in range(iterations):
        routes = []
        lengths = []
        for _ant in range(n_ants):
            route = [0]
            unvisited = set(range(1, n))
            while unvisited:
                current = route[-1]
                candidates = np.array(sorted(unvisited))
                desirability = (pheromone[current, candidates] ** alpha) * (heuristic[current, candidates] ** beta)
                probabilities = desirability / desirability.sum()
                nxt = int(rng.choice(candidates, p=probabilities))
                route.append(nxt)
                unvisited.remove(nxt)
            length = route_length(route, distances)
            routes.append(route)
            lengths.append(length)
            if length < best_length:
                best_route, best_length = route.copy(), length

        pheromone *= 1 - evaporation
        for route, length in zip(routes, lengths):
            deposit = 1.0 / length
            for i in range(n):
                a, b = route[i], route[(i + 1) % n]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit
        for i in range(n):
            a, b = best_route[i], best_route[(i + 1) % n]
            pheromone[a, b] += 2.0 / best_length
            pheromone[b, a] += 2.0 / best_length
        history.append(best_length)
    return best_route, float(best_length), history


def run_ant_colony_example() -> dict:
    names = list(CITIES)
    coords = list(CITIES.values())
    distances = distance_matrix(coords)
    aco_route, aco_length, history = ant_colony_tsp(distances)
    nn_route = nearest_neighbor_tour(distances)
    nn_length = route_length(nn_route, distances)
    exact_length, exact_route = held_karp(distances)

    route_names = [names[i] for i in aco_route] + [names[aco_route[0]]]
    pd.DataFrame({"order": np.arange(1, len(route_names) + 1), "city": route_names}).to_csv(
        RESULTS_DIR / "aco_route.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lons = np.array([coords[i][1] for i in aco_route + [aco_route[0]]])
    lats = np.array([coords[i][0] for i in aco_route + [aco_route[0]]])
    axes[0].plot(lons, lats, marker="o", color="#08519C", linewidth=1.8)
    for name, (lat, lon) in CITIES.items():
        axes[0].annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set(title="Маршрут мурашиного алгоритму", xlabel="Довгота", ylabel="Широта")
    axes[0].set_aspect(1.35)
    axes[1].plot(np.arange(1, len(history) + 1), history, color="#CB181D")
    axes[1].axhline(exact_length, color="black", linestyle=":", label="Точний оптимум")
    axes[1].set(title="Збіжність мурашиної колонії", xlabel="Ітерація", ylabel="Найкраща довжина, км")
    axes[1].legend()
    save_figure(fig, "figure_7_aco_tsp.png")

    return {
        "aco_route": route_names,
        "aco_length_km": aco_length,
        "nearest_neighbor_length_km": nn_length,
        "exact_length_km": exact_length,
        "aco_gap_percent": 100 * (aco_length / exact_length - 1),
        "nearest_neighbor_gap_percent": 100 * (nn_length / exact_length - 1),
        "exact_route": [names[i] for i in exact_route] + [names[exact_route[0]]],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    kmeans = run_kmeans_example()
    som = run_som_example(kmeans["x_scaled"], kmeans["y"], kmeans["target_names"])
    fuzzy = run_fuzzy_example()
    genetic = run_genetic_example()
    aco = run_ant_colony_example()

    selected = kmeans["evaluation"].loc[kmeans["evaluation"]["k"] == 3].iloc[0]
    metrics = {
        "kmeans": {
            "k": 3,
            "inertia": float(selected["inertia"]),
            "silhouette": float(selected["silhouette"]),
            "adjusted_rand": float(selected["adjusted_rand"]),
            "pca_explained_variance_2d": kmeans["explained_variance_2d"],
        },
        "som": som,
        "fuzzy": fuzzy,
        "genetic": genetic,
        "aco": aco,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Практичну частину виконано.")
    print(f"k means: silhouette = {metrics['kmeans']['silhouette']:.3f}, ARI = {metrics['kmeans']['adjusted_rand']:.3f}")
    print(f"SOM: квантизаційна помилка = {som['quantization_error']:.3f}, топографічна = {som['topographic_error']:.3f}")
    print(f"Нечіткий регулятор: {fuzzy['temperature']:.1f} °C -> {fuzzy['fan_speed']:.1f} %")
    print(f"Генетичний алгоритм: x = {genetic['best_x']:.4f}, f(x) = {genetic['best_value']:.4f}")
    print(f"ACO: {aco['aco_length_km']:.1f} км, відхилення від точного оптимуму = {aco['aco_gap_percent']:.2f} %")
    print(f"Результати: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
