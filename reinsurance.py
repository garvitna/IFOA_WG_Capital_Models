import numpy as np
import pandas as pd
from pal.variables import ProteusVariable
from pal.frequency_severity import FreqSevSims


def apply_quota_share(losses_by_lob: ProteusVariable, quota_share_cession: pd.Series):
    """
    Apply quota share reinsurance to the losses by line of business (LOB).

    Parameters:
        losses_by_lob: dict, losses by LOB.
        Quota_Share: pd.Series, quota share percentages for each LOB.

    Returns:
        ceded_losses_by_lob: dict, ceded losses by LOB.
    """
    return ProteusVariable(
        "lob", {lob: losses_by_lob[lob] * quota_share_cession[lob] for lob in quota_share_cession.index}
    )


def apply_reinsurance(
    attritional_losses_by_lob: ProteusVariable,
    individual_large_losses_by_lob: ProteusVariable,
    individual_cat_losses_by_lob: ProteusVariable,
    Quota_Share: pd.Series,
    Large_XoL_Retention: pd.Series,
    Large_XoL_Limit: pd.Series,
    CAT_XoL_Retention: pd.Series,
    CAT_XoL_Limit: pd.Series,
) -> ProteusVariable:
    """
    Apply a simple reinsurance to the losses by line of business (LOB).

    The reinsurance structure comprises of a quota share with varying cession rates for each line of business (LOB),
    an individual excess of loss (XoL) for large losses with a retention and limit for each LOB,
    and a catastrophe (CAT) excess of loss program comprised of a number of layers with an event retention and limit for all LOBs.

    Parameters:
        attritional_losses_by_lob: dict, attritional losses by LOB.
        individual_large_losses_by_lob: dict, individual large losses by LOB.
        individual_cat_losses_by_lob: dict, individual CAT losses by LOB.
        Quota_Share: pd.Series, quota share percentages for each LOB.
        Large_XoL_Retention: pd.Series, large excess of loss retention amounts for each LOB.
        Large_XoL_Limit: pd.Series, large excess of loss limit amounts for each LOB.
        CAT_XoL_Retention: pd.Series, CAT excess of loss retention amount for each layer.
        CAT_XoL_Limit: pd.Series, CAT excess of loss limit amount for each layer.

    Returns:
        ceded_losses_by_lob: ProteusVariable, ceded losses by LOB, structured by loss type (Attritional, Large, Catastrophe).
    """

    qs_ceded_attritional_losses_by_lob = apply_quota_share(attritional_losses_by_lob, Quota_Share)
    qs_ceded_individual_large_losses_by_lob = apply_quota_share(individual_large_losses_by_lob, Quota_Share)
    qs_ceded_individual_cat_losses_by_lob = apply_quota_share(individual_cat_losses_by_lob, Quota_Share)

    net_qs_individual_large_losses_by_lob = individual_large_losses_by_lob - qs_ceded_individual_large_losses_by_lob
    net_qs_individual_cat_losses_by_lob = individual_cat_losses_by_lob - qs_ceded_individual_cat_losses_by_lob
    # Apply the XoL to the individual large losses
    xl_ceded_individual_large_losses_by_lob = ProteusVariable(
        "lob",
        {
            lob: np.minimum(
                np.maximum(net_qs_individual_large_losses_by_lob[lob] - Large_XoL_Retention[lob], 0),
                Large_XoL_Limit[lob],
            )
            for lob in net_qs_individual_large_losses_by_lob.values.keys()
        },
    )
    total_ceded_individual_large_losses_by_lob = (
        qs_ceded_individual_large_losses_by_lob + xl_ceded_individual_large_losses_by_lob
    )

    net_qs_total_cat_event_losses: FreqSevSims = net_qs_individual_cat_losses_by_lob.sum()
    # cat program
    cat_layers = CAT_XoL_Retention.index
    xl_ceded_cat_event_losses_by_lob = 0.0
    for layer in cat_layers:
        xl_ceded_cat_event_losses = np.minimum(
            np.maximum(net_qs_total_cat_event_losses - CAT_XoL_Retention[layer], 0), CAT_XoL_Limit[layer]
        )
        xl_cat_recovery_ratio = xl_ceded_cat_event_losses / net_qs_total_cat_event_losses
        xl_ceded_cat_event_losses_by_lob += xl_ceded_cat_event_losses * xl_cat_recovery_ratio
    total_ceded_individual_cat_losses_by_lob = qs_ceded_individual_cat_losses_by_lob + xl_ceded_cat_event_losses_by_lob

    return ProteusVariable(
        "loss_type",
        {
            "Attritional": qs_ceded_attritional_losses_by_lob,
            "Large": total_ceded_individual_large_losses_by_lob,
            "Catastrophe": total_ceded_individual_cat_losses_by_lob,
        },
    )
