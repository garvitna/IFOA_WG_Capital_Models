import pandas as pd
import numpy as np
from pal import distributions
from pal.variables import ProteusVariable


def generate_reserve_risk(
    mean_by_lob: pd.Series,
    sigma_by_lob: pd.Series,
) -> ProteusVariable:
    """Generates distribution of ultimate future reserve cashflows by line of business."""

    total_reserve_cashflows_by_lob = ProteusVariable(
        "lob",
        {
            lob: mean_by_lob[lob]
            * distributions.LogNormal(
                -0.5 * sigma_by_lob[lob] ** 2,
                sigma_by_lob[lob],
            ).generate()
            for lob in mean_by_lob.index
        },
    )
    return total_reserve_cashflows_by_lob
