import pandas as pd
import numpy as np
from pal import distributions
from pal.variables import ProteusVariable


def generate_reserve_risk(
    mean_by_lob: pd.Series,
    sigma_by_lob: pd.Series,
    payment_pattern_by_lob: pd.DataFrame,
) -> ProteusVariable:
    """Generates distribution of ultimate future reserve cashflows by line of business."""
    total_reserve_cashflows_by_lob = ProteusVariable(dim_name="lob", values={})  # accumulate reserve risk per LoB
    lobs = mean_by_lob.index

    for lob in lobs:
        incremental_payment_pattern = np.diff(payment_pattern_by_lob[lob].values, prepend=0)
        actual_cashflows = ProteusVariable(
            "year",
            {
                yr: incremental_payment_pattern[idx]
                * mean_by_lob[lob]
                * distributions.LogNormal(
                    -0.5 * sigma_by_lob[lob] ** 2,
                    sigma_by_lob[lob],
                ).generate()
                for idx, yr in enumerate(payment_pattern_by_lob[lob].index)
            },
        )
        total_reserve_cashflows_by_lob.values[lob] = actual_cashflows.sum()

    return total_reserve_cashflows_by_lob
