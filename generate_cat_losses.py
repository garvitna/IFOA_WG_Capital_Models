from pal import distributions, FrequencySeverityModel, FreqSevSims
from pal.variables import ProteusVariable, StochasticScalar
import pandas as pd
import numpy as np

lobs = ["Specialty", "Property", "Liability"]

market_loss_severity = FrequencySeverityModel(
    distributions.Poisson(mean=10),
    distributions.Uniform(0, 1),
).generate()
sigma = {"Specialty": 0.75, "Property": 1.25, "Liability": 0.5}
scale = {"Specialty": 0.1, "Property": 0.25, "Liability": 0.05}

premium = {"Specialty": 1.9e9, "Property": 3e9, "Liability": 1.9e9}
individual_cat_losses_by_lob = ProteusVariable(
    "lob",
    {
        lob: FreqSevSims(
            sim_index=market_loss_severity.sim_index,
            values=(
                distributions.LogNormal(-0.5 * sigma[lob] ** 2, sigma[lob]).invcdf(
                    StochasticScalar(market_loss_severity.values)
                )
                * distributions.LogNormal(-0.5 * 0.1, 0.1).generate(n_sims=len(market_loss_severity.values))
            ).values
            * scale[lob]
            / 10
            * premium[lob],
            n_sims=10000,
        )
        for lob in lobs
    },
)

df = pd.DataFrame(
    {
        "Sim": individual_cat_losses_by_lob["Property"].sim_index + 1,
        "EventId": np.arange(len(individual_cat_losses_by_lob["Property"].sim_index)) + 1,
        "Specialty": individual_cat_losses_by_lob["Specialty"].values,
        "Property": individual_cat_losses_by_lob["Property"].values,
        "Liability": individual_cat_losses_by_lob["Liability"].values,
    }
)

df.to_csv("data/cat_yelt.csv", index=False)
