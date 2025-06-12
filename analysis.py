import plotly.graph_objects as go
from pal.variables import ProteusVariable, StochasticScalar
from pal import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_percentile_table(
    risk_by_risk_type: ProteusVariable, losses_by_loss_type: ProteusVariable, percentiles: float
) -> pd.DataFrame:
    combined_risk: StochasticScalar = risk_by_risk_type.sum()
    underwriting_risk = risk_by_risk_type["Underwriting Risk"]
    reserve_risk = risk_by_risk_type["Reserve Risk"]
    market_risk = risk_by_risk_type["Market Risk"]
    credit_risk = risk_by_risk_type["Credit Risk"]
    operational_risk = risk_by_risk_type["Operational Risk"]
    net_non_catastrophe_losses = losses_by_loss_type["Non-Catastrophe"]
    net_aggregate_cat_losses = losses_by_loss_type["Catastrophe"]

    # Table of VaR and TVaR at different percentiles
    Combined_Risk_VaR = combined_risk.percentile(percentiles)
    Combined_Risk_TVaR = combined_risk.tvar(percentiles)
    underwriting_risk_VaR = underwriting_risk.percentile(percentiles)
    underwriting_risk_TVaR = underwriting_risk.tvar(percentiles)
    reserve_risk_VaR = reserve_risk.percentile(percentiles)
    reserve_risk_TVaR = reserve_risk.tvar(percentiles)
    Market_risk_VaR = market_risk.percentile(percentiles)
    Market_risk_TVaR = market_risk.tvar(percentiles)
    credit_risk_VaR = credit_risk.percentile(percentiles)
    credit_risk_TVaR = credit_risk.tvar(percentiles)
    operational_risk_VaR = operational_risk.percentile(percentiles)
    operational_risk_TVaR = operational_risk.tvar(percentiles)
    net_non_catastrophe_losses_VaR = net_non_catastrophe_losses.percentile(percentiles)
    net_non_catastrophe_losses_TVaR = net_non_catastrophe_losses.tvar(percentiles)
    net_CAT_losses_VaR = net_aggregate_cat_losses.percentile(percentiles)
    net_CAT_losses_TVaR = net_aggregate_cat_losses.tvar(percentiles)
    percentile_data = pd.DataFrame(
        {
            "Percentile": percentiles,
            "Combined Risk VaR": Combined_Risk_VaR,
            "Combined Risk TVaR": Combined_Risk_TVaR,
            "Underwriting Risk VaR": underwriting_risk_VaR,
            "Underwriting Risk TVaR": underwriting_risk_TVaR,
            "Reserve Risk VaR": reserve_risk_VaR,
            "Reserve Risk TVaR": reserve_risk_TVaR,
            "Market Risk VaR": Market_risk_VaR,
            "Market Risk TVaR": Market_risk_TVaR,
            "Credit Risk VaR": credit_risk_VaR,
            "Credit Risk TVaR": credit_risk_TVaR,
            "Operational Risk VaR": operational_risk_VaR,
            "Operational Risk TVaR": operational_risk_TVaR,
            "Net Non-Catastrophe Losses VaR": net_non_catastrophe_losses_VaR,
            "Net Non-Catastrophe Losses TVaR": net_non_catastrophe_losses_VaR,
            "Net CAT Losses VaR": net_CAT_losses_VaR,
            "Net CAT Losses TVaR": net_CAT_losses_TVaR,
        }
    )
    return percentile_data


# Outputting the results to an excel file


def create_risk_scatter_plot(risks: ProteusVariable) -> go.Figure:
    """Create a scatter plot of the various risk types"""
    combined_risk = risks.sum()

    fig = go.Figure()
    for risk_type in risks.values.keys():
        fig.add_trace(
            go.Scattergl(
                x=np.array(combined_risk),
                y=np.array(risks[risk_type].values),
                mode="markers",
                name=risk_type,
            )
        )
    fig.update_layout(
        title="Scatter Plot of Various Risks",
        xaxis_title="Combined Risks",
        yaxis_title="Risk Values",
    )
    return fig


def create_jep(risk_by_risk_type: ProteusVariable, selected_percentile: float) -> pd.DataFrame:
    risk_types = risk_by_risk_type.values.keys()
    thresholds = risk_by_risk_type.percentile(selected_percentile)

    joint_exceedance_matrix = np.zeros((len(thresholds), len(thresholds)))

    for i, risk1 in enumerate(risk_types):
        for j, risk2 in enumerate(risk_types):
            exceed_i = risk_by_risk_type[risk1] > thresholds[risk1]
            exceed_j = risk_by_risk_type[risk2] > thresholds[risk2]
            joint_exceedance_matrix[i, j] = (exceed_i & exceed_j).mean()

    joint_exceedance_df = pd.DataFrame(joint_exceedance_matrix, index=risk_types, columns=risk_types)
    return joint_exceedance_df


def export_to_excel(
    risk_by_risk_type: ProteusVariable,
    selected_percentile: float,
    losses_by_loss_type: ProteusVariable,
    percentiles: pd.Series,
    correlation_matrix: pd.DataFrame,
    jep: pd.DataFrame,
    output_filename: str,
):

    risk_types = risk_by_risk_type.values.keys()
    risk_by_risk_type_with_total = ProteusVariable(
        "risk_type",
        dict(
            **{risk_type: risk_by_risk_type[risk_type] for risk_type in risk_types},
            **{"Combined Risk": risk_by_risk_type.sum()}
        ),
    )
    risk_types_with_total = risk_by_risk_type_with_total.values.keys()
    total_risks = pd.DataFrame(
        {
            "Sim_Index": np.arange(1, config.n_sims + 1),
        }
    )
    for risk_type in risk_types_with_total:
        total_risks[risk_type] = risk_by_risk_type_with_total[risk_type].values

    tvar = pd.Series(
        {risk: risk_by_risk_type_with_total[risk].tvar(selected_percentile) for risk in risk_types_with_total},
        name="Risk Type",
    )
    var = pd.Series(
        {risk: risk_by_risk_type_with_total[risk].percentile(selected_percentile) for risk in risk_types_with_total},
        name="Risk Type",
    )
    capital_requirement = var["Combined Risk"]
    percentile_data = create_percentile_table(risk_by_risk_type, losses_by_loss_type, percentiles)
    with pd.ExcelWriter(output_filename) as writer:
        pd.DataFrame({"Selected_Percentile": [selected_percentile], "Capital": [capital_requirement]}).to_excel(
            writer, sheet_name="Capital_Requirement", index=False
        )
        total_risks.to_excel(writer, sheet_name="Total Risks", index=False)
        percentile_data.to_excel(writer, sheet_name="VaR and TVaR", index=False)
        correlation_matrix.to_excel(writer, sheet_name="Spearman_Corr_Matrix", index=True)
        var.to_excel(writer, sheet_name="VaR by Risk Type", index=True)
        tvar.to_excel(writer, sheet_name="TVaR by Risk Type", index=True)
        jep.to_excel(writer, sheet_name="Joint_Exceedance_Probabilities", index=True)


def produce_analysis(
    underwriting_risk: StochasticScalar,
    reserve_risk: StochasticScalar,
    market_risk: StochasticScalar,
    credit_risk: StochasticScalar,
    operational_risk: StochasticScalar,
    net_losses_by_loss_type: ProteusVariable,
    output_filename: str,
):
    """Produce the analysis of the risk factors."""
    # Combine all risks
    combined_risk = underwriting_risk + reserve_risk + market_risk + credit_risk + operational_risk
    # Show CDFs
    underwriting_risk.show_cdf(title="Underwriting Risk")
    reserve_risk.show_cdf(title="Reserve Risk")
    market_risk.show_cdf(title="Market Risk")
    credit_risk.show_cdf(title="Credit Risk")
    operational_risk.show_cdf(title="Operational Risk")
    combined_risk.show_cdf(title="Combined Risk")

    # percentiles
    selected_percentile = 99.5
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99.5]

    risk_by_risk_type = ProteusVariable(
        "risk_type",
        {
            "Underwriting Risk": underwriting_risk,
            "Reserve Risk": reserve_risk,
            "Market Risk": market_risk,
            "Credit Risk": credit_risk,
            "Operational Risk": operational_risk,
        },
    )

    fig = create_risk_scatter_plot(risk_by_risk_type)

    fig.show()

    spearman_correlation_matrix = pd.DataFrame(
        data=risk_by_risk_type.correlation_matrix(),
        index=risk_by_risk_type.values.keys(),
        columns=risk_by_risk_type.values.keys(),
    )

    # Plotting Pearson Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        spearman_correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=risk_by_risk_type.values.keys(),
        yticklabels=risk_by_risk_type.values.keys(),
    )
    plt.title("Correlation Heatmap of Risk Factors")
    plt.show()

    # Plotting joint exceedance probabilities

    jep = create_jep(risk_by_risk_type, selected_percentile)
    plt.figure(figsize=(8, 6))
    sns.heatmap(jep, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title("Pairwise Joint Exceedance Probabilities")
    plt.show()

    export_to_excel(
        risk_by_risk_type,
        selected_percentile,
        net_losses_by_loss_type,
        percentiles,
        spearman_correlation_matrix,
        jep,
        output_filename,
    )
