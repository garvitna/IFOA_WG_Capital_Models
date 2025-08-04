import numpy as np
import pandas as pd
from pal.variables import ProteusVariable
from pal.frequency_severity import FreqSevSims
from pal.contracts import XoL, XoLTower


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
    non_cat_losses_by_lob: ProteusVariable,
    individual_cat_losses_by_lob: ProteusVariable,
    Quota_Share: pd.Series,
    CAT_XoL_Retention: pd.Series,
    CAT_XoL_Limit: pd.Series,
) -> ProteusVariable:
    """
    Apply a simple reinsurance to the losses by line of business (LOB).

    The reinsurance structure comprises of a quota share with varying cession rates for each line of business (LOB),
    an individual excess of loss (XoL) for large losses with a retention and limit for each LOB,
    and a catastrophe (CAT) excess of loss program comprised of a number of layers with an event retention and limit for all LOBs.

    Parameters:
        non_cat_losses_by_lob: ProteusVariable, non_cat_losses_by_lob LOB.
        individual_cat_losses_by_lob: ProteusVariable, individual CAT losses by LOB.
        Quota_Share: pd.Series, quota share percentages for each LOB.
        CAT_XoL_Retention: pd.Series, CAT excess of loss retention amount for each layer.
        CAT_XoL_Limit: pd.Series, CAT excess of loss limit amount for each layer.

    Returns:
        ceded_losses_by_lob: ProteusVariable, ceded losses by LOB, structured by loss type (Attritional, Large, Catastrophe).
    """

    qs_ceded_non_cat_losses_by_lob = apply_quota_share(non_cat_losses_by_lob, Quota_Share)
    qs_ceded_individual_cat_losses_by_lob = apply_quota_share(individual_cat_losses_by_lob, Quota_Share)

    net_qs_individual_cat_losses_by_lob = individual_cat_losses_by_lob - qs_ceded_individual_cat_losses_by_lob

    net_qs_total_cat_event_losses: FreqSevSims = net_qs_individual_cat_losses_by_lob.sum()
    # cat program
    cat_layers = CAT_XoL_Retention.index
    cat_xol_tower = XoLTower(
        CAT_XoL_Limit.values,
        CAT_XoL_Retention.values,
        premium=[0 for _ in cat_layers],  # Premium is not used in this calculation
        name=cat_layers,
    )
    xl_ceded_cat_event_losses = cat_xol_tower.apply(net_qs_total_cat_event_losses).recoveries
    xl_cat_recovery_ratio = xl_ceded_cat_event_losses / net_qs_total_cat_event_losses
    xl_ceded_cat_event_losses_by_lob = net_qs_individual_cat_losses_by_lob * xl_cat_recovery_ratio
    total_ceded_individual_cat_losses_by_lob = qs_ceded_individual_cat_losses_by_lob + xl_ceded_cat_event_losses_by_lob

    return ProteusVariable(
        "loss_type",
        {
            "Non-Catastrophe": qs_ceded_non_cat_losses_by_lob,
            "Catastrophe": total_ceded_individual_cat_losses_by_lob,
        },
    )
