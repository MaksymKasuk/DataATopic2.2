"""Комп'ютерний практикум № 5. Регресія Кокса.

Сценарій підготовлено для Google Colab і локального запуску.
Він завантажує набір Rossi, будує модель пропорційних ризиків Кокса,
оцінює якість, перевіряє припущення пропорційності та зберігає таблиці й графіки.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path


def ensure_lifelines() -> None:
    try:
        import lifelines  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lifelines==0.30.0"])


ensure_lifelines()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index, k_fold_cross_validation
from scipy.stats import combine_pvalues
from sklearn.model_selection import train_test_split


SEED = 42
OUTPUT_DIR = Path(os.environ.get("COX_OUTPUT_DIR", "cox_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
LABELS_UA = {
    "fin": "Фінансова допомога",
    "age": "Вік",
    "race": "Темношкіра раса",
    "wexp": "Досвід повної зайнятості",
    "mar": "Шлюб",
    "paro": "Умовно дострокове звільнення",
    "prio": "Кількість попередніх засуджень",
}


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close()


def bootstrap_c_index(df: pd.DataFrame, risk: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        try:
            c_value = concordance_index(sample["week"], -risk[idx], sample["arrest"])
            if np.isfinite(c_value):
                values.append(float(c_value))
        except ZeroDivisionError:
            continue
    return tuple(np.quantile(values, [0.025, 0.975]))


def coefficient_table(model: CoxPHFitter) -> pd.DataFrame:
    summary = model.summary.reset_index().rename(columns={"covariate": "variable"})
    out = summary[["variable", "coef", "se(coef)", "z", "p", "exp(coef)",
                   "exp(coef) lower 95%", "exp(coef) upper 95%"]].copy()
    out.insert(1, "variable_ua", out["variable"].map(LABELS_UA))
    out.columns = ["variable", "variable_ua", "beta", "se", "z", "p_value",
                   "hazard_ratio", "ci95_low", "ci95_high"]
    out["risk_change_percent"] = (out["hazard_ratio"] - 1) * 100
    return out


def main() -> None:
    configure_plotting()
    df = load_rossi().copy()
    df.to_csv(OUTPUT_DIR / "rossi_data.csv", index=False)

    descriptive = df.describe().T
    descriptive.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    events = int(df["arrest"].sum())
    censored = int((1 - df["arrest"]).sum())

    train, test = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["arrest"]
    )
    train = train.copy()
    test = test.copy()

    train_model = CoxPHFitter()
    train_model.fit(train, duration_col="week", event_col="arrest", formula=" + ".join(PREDICTORS))
    test_risk = train_model.predict_partial_hazard(test).to_numpy().ravel()
    test_c = float(concordance_index(test["week"], -test_risk, test["arrest"]))

    cv_scores = k_fold_cross_validation(
        CoxPHFitter(), df, duration_col="week", event_col="arrest",
        k=5, scoring_method="concordance_index", seed=SEED
    )

    model = CoxPHFitter()
    model.fit(df, duration_col="week", event_col="arrest", formula=" + ".join(PREDICTORS))
    coef = coefficient_table(model)
    coef.to_csv(OUTPUT_DIR / "cox_coefficients.csv", index=False)

    lr = model.log_likelihood_ratio_test()
    full_risk = model.predict_partial_hazard(df).to_numpy().ravel()
    ci_low, ci_high = bootstrap_c_index(df, full_risk)

    ph = proportional_hazard_test(model, df, time_transform="rank")
    ph_table = ph.summary.reset_index().rename(columns={"index": "variable"})
    ph_table.insert(1, "variable_ua", ph_table["variable"].map(LABELS_UA))
    ph_table.to_csv(OUTPUT_DIR / "proportional_hazards_test.csv", index=False)
    _, ph_global_p = combine_pvalues(ph_table["p"].to_numpy(), method="fisher")

    metrics = {
        "n": int(len(df)),
        "events": events,
        "censored": censored,
        "event_rate": float(events / len(df)),
        "log_likelihood": float(model.log_likelihood_),
        "partial_aic": float(model.AIC_partial_),
        "likelihood_ratio_chi2": float(lr.test_statistic),
        "likelihood_ratio_df": int(lr.degrees_freedom),
        "likelihood_ratio_p": float(lr.p_value),
        "c_index_full": float(model.concordance_index_),
        "c_index_bootstrap_ci95_low": float(ci_low),
        "c_index_bootstrap_ci95_high": float(ci_high),
        "c_index_holdout": test_c,
        "c_index_cv_mean": float(np.mean(cv_scores)),
        "c_index_cv_sd": float(np.std(cv_scores, ddof=1)),
        "ph_global_fisher_p": float(ph_global_p),
        "ph_min_variable_p": float(ph_table["p"].min()),
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Рисунок 1. Розподіл події та тривалості спостереження.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    counts = pd.Series({"Повторний арешт": events, "Цензуровано": censored})
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False,
                palette=["#C44E52", "#4C72B0"], ax=axes[0])
    axes[0].set_title("Статус спостережень")
    axes[0].set_ylabel("Кількість осіб")
    axes[0].set_xlabel("")
    sns.histplot(data=df, x="week", hue="arrest", bins=18, multiple="stack",
                 palette={0: "#4C72B0", 1: "#C44E52"}, ax=axes[1])
    axes[1].set_title("Тривалість спостереження")
    axes[1].set_xlabel("Тижні")
    axes[1].set_ylabel("Кількість осіб")
    savefig("figure_1_data_overview.png")

    # Рисунок 2. Емпіричні криві Каплана Мейєра за фінансовою допомогою.
    plt.figure(figsize=(7.2, 4.8))
    kmf = KaplanMeierFitter()
    for value, label, color in [(0, "Допомогу не отримував", "#C44E52"),
                                (1, "Допомогу отримував", "#4C72B0")]:
        mask = df["fin"] == value
        kmf.fit(df.loc[mask, "week"], event_observed=df.loc[mask, "arrest"], label=label)
        kmf.plot_survival_function(ci_show=True, color=color)
    plt.title("Криві Каплана Мейєра за фінансовою допомогою")
    plt.xlabel("Тижні після звільнення")
    plt.ylabel("Ймовірність залишитися без повторного арешту")
    savefig("figure_2_km_financial_aid.png")

    # Рисунок 3. Відношення ризиків та 95% довірчі інтервали.
    forest = coef.sort_values("hazard_ratio").copy()
    y = np.arange(len(forest))
    plt.figure(figsize=(8.5, 5.2))
    plt.errorbar(
        forest["hazard_ratio"], y,
        xerr=[forest["hazard_ratio"] - forest["ci95_low"],
              forest["ci95_high"] - forest["hazard_ratio"]],
        fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=3
    )
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, forest["variable_ua"])
    plt.xscale("log")
    plt.xlabel("Відношення ризиків, логарифмічна шкала")
    plt.title("Вплив предикторів у моделі Кокса")
    savefig("figure_3_hazard_ratios.png")

    # Рисунок 4. Прогнозовані криві для трьох профілів ризику.
    ordered = np.argsort(full_risk)
    selected = [ordered[int(0.10 * (len(df) - 1))],
                ordered[int(0.50 * (len(df) - 1))],
                ordered[int(0.90 * (len(df) - 1))]]
    profiles = df.iloc[selected][PREDICTORS]
    survival = model.predict_survival_function(profiles)
    survival.columns = ["Низький ризик", "Медіанний ризик", "Високий ризик"]
    ax = survival.plot(figsize=(7.2, 4.8), color=["#55A868", "#4C72B0", "#C44E52"])
    ax.set_title("Прогнозована ймовірність відсутності повторного арешту")
    ax.set_xlabel("Тижні після звільнення")
    ax.set_ylabel("Ймовірність")
    savefig("figure_4_predicted_survival.png")

    # Рисунок 5. Діагностика пропорційності ризиків.
    ph_plot = ph_table.sort_values("p")
    plt.figure(figsize=(8.2, 4.8))
    colors = np.where(ph_plot["p"] < 0.05, "#C44E52", "#55A868")
    plt.barh(ph_plot["variable_ua"], -np.log10(ph_plot["p"]), color=colors)
    plt.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1,
                label="Рівень 0,05")
    plt.xlabel("-log10(p)")
    plt.title("Тест пропорційності ризиків за залишками Шенфельда")
    plt.legend()
    savefig("figure_5_ph_test.png")

    print("Модель Кокса успішно побудовано.")
    print(pd.Series(metrics).round(4).to_string())
    print("\nКоефіцієнти моделі:")
    print(coef.round(4).to_string(index=False))
    print(f"\nРезультати збережено у: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
