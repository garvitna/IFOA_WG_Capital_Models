import pal
from pal import config, distributions
from pal.variables import ProteusVariable
from pal.stochastic_scalar import StochasticScalar
import numpy as np
import pandas as pd


from catastrophes import load_yelt
from market_risks import generate_market_risk_factors
from reinsurance import apply_reinsurance
from reserves import generate_reserve_risk
from operational_risk import generate_operational_risk
from analysis import produce_analysis

pal.config.n_sims = 10000
pal.set_random_seed(13892911)

file_path = "Driver_Based_Model.xlsx"

underwriting_params = pd.read_excel(file_path, sheet_name="Underwriting", index_col="lob")
reserve_params = pd.read_excel(file_path, sheet_name="Reserve", index_col="lob")
credit_operational_cat = pd.read_excel(file_path, sheet_name="Credit_Operational_CAT")
market_df = pd.read_excel(file_path, sheet_name="Market")
market_class_impact = pd.read_excel(file_path, sheet_name="Market_Class_Impact", index_col="lob")
market_corr = pd.read_excel(file_path, sheet_name="Market_Correlation", index_col=0)
credit_operational_market_ec = pd.read_excel(file_path, sheet_name="Credit_Operational_Market_EC")
credit_df = pd.read_excel(file_path, sheet_name="Credit")
operational_df = pd.read_excel(file_path, sheet_name="Operational")
underwriting_cycle_df = pd.read_excel(file_path, sheet_name="Underwriting_Cycle")
underwriting_cycle_params = dict(zip(underwriting_cycle_df["parameter_name"], underwriting_cycle_df["value"]))
underwriting_cycle_class_impact_df = pd.read_excel(file_path, sheet_name="Underwriting_Cycle_Class", index_col="lob")
credit_operational_uc = pd.read_excel(file_path, sheet_name="Credit_Operational_UC")
inflation_class_impact = pd.read_excel(file_path, sheet_name="Inflation_Class", index_col="lob")
reinsurance_params = pd.read_excel(file_path, sheet_name="Reinsurance", index_col="lob")
cat_reinsurance_df = pd.read_excel(file_path, sheet_name="Cat Reinsurance", index_col="layer")

expense_df = pd.read_excel(file_path, sheet_name="Expense Inputs")
discount_curve = pd.read_excel(file_path, sheet_name="Yield Curve")
payment_pattern_df = pd.read_excel(file_path, sheet_name="Payment Pattern")


lobs = underwriting_params.index
net_premium_by_lob = underwriting_params["Net Premium"]
Gamma_Alpha_Parameters = underwriting_params["Gamma_Alpha_Parameter"]
Gamma_Theta_Parameters = underwriting_params["Gamma_Theta_Parameter"]
reserve_params_mean = reserve_params["Reserve_Mean"]
reserve_params_sigma = reserve_params["Reserve_Sigma"]
Credit_Impact_CAT_Threshold = credit_operational_cat.loc[
    credit_operational_cat["parameter_name"] == "Credit_Impact_CAT_Threshold", "value"
].values[0]
Operational_Impact_CAT_Threshold = credit_operational_cat.loc[
    credit_operational_cat["parameter_name"] == "Operational_Impact_CAT_Threshold", "value"
].values[0]
Credit_Impact_CAT = credit_operational_cat["Credit_Impact_CAT"]
Operational_Impact_CAT = credit_operational_cat["Operational_Impact_CAT"]
Economic_Cycle_Class_Impact = market_class_impact["Economic_Cycle_Impact"]
Currency_Exchange_Class_Impact = market_class_impact["Currency_Exchange_Impact"]
Credit_Impact_EC_Threshold = credit_operational_market_ec.loc[
    credit_operational_market_ec["parameter_name"] == "Credit_Impact_EC_Threshold", "value"
].values[0]
Operational_Impact_EC_Threshold = credit_operational_market_ec.loc[
    credit_operational_market_ec["parameter_name"] == "Operational_Impact_EC_Threshold", "value"
].values[0]
Market_Impact_EC_Threshold = credit_operational_market_ec.loc[
    credit_operational_market_ec["parameter_name"] == "Market_Impact_EC_Threshold", "value"
].values[0]
Credit_Impact_EC = credit_operational_market_ec["Credit_Impact_EC"]
Operational_Impact_EC = credit_operational_market_ec["Operational_Impact_EC"]
Market_Impact_EC = credit_operational_market_ec["Market_Impact_EC"]
Underwriting_Cycle_mu = underwriting_cycle_params["Underwriting_Cycle_mu"]
Underwriting_Cycle_sigma = underwriting_cycle_params["Underwriting_Cycle_sigma"]
underwriting_cycle_class_impact = ProteusVariable.from_series(
    underwriting_cycle_class_impact_df["Underwriting_Cycle_Impact"]
)
Credit_Impact_UC_Threshold = credit_operational_uc.loc[
    credit_operational_uc["parameter_name"] == "Credit_Impact_UC_Threshold", "value"
].values[0]
Operational_Impact_UC_Threshold = credit_operational_uc.loc[
    credit_operational_uc["parameter_name"] == "Operational_Impact_UC_Threshold", "value"
].values[0]
Credit_Impact_UC = credit_operational_uc["Credit_Impact_UC"]
Operational_Impact_UC = credit_operational_uc["Operational_Impact_UC"]
Inflation_class_impact = inflation_class_impact["Inflation_Impact"]
Quota_Share = reinsurance_params["Quota_Share"]
Payment_Pattern = payment_pattern_df.set_index("Year")
CAT_XoL_Retention = cat_reinsurance_df["CAT_XoL_Retention"]
CAT_XoL_Limit = cat_reinsurance_df["CAT_XoL_Limit"]


# Process Expense Parameters
expense_params = dict(zip(expense_df["parameter_name"], expense_df["value"]))
Expense_mu = expense_params["Expense_mu"]
Expense_sigma = expense_params["Expense_sigma"]

# Process Market Parameters
market_params = {k: dict(zip(v["parameter_name"], v["value"])) for k, v in market_df.groupby("main_category")}
# Process Credit Parameters
credit_params = {k: dict(zip(v["parameter_name"], v["value"])) for k, v in credit_df.groupby("main_category")}
# Process Operational Parameters
operational_params = dict(zip(operational_df["parameter_name"], operational_df["value"]))
# Process Expense Parameters
expense_params = dict(zip(expense_df["parameter_name"], expense_df["value"]))

# Underwriting Risk

# Generate the non_cat losses by class
non_cat_losses_by_lob_ind = ProteusVariable(
    "lob",
    values={
        lob: distributions.Gamma(alpha=Gamma_Alpha_Parameters[lob], theta=Gamma_Theta_Parameters[lob]).generate()
        for lob in lobs
    },
)


# Generate the CAT losses by class

individual_cat_losses_by_lob = load_yelt("data/cat_yelt.csv", 10000)

# Generate the underwriting cycle by class
# As James Toller mentioned, I have used lognormal distribution instead of normal distribution for the underwriting cycle

underwriting_cycle: StochasticScalar = (
    distributions.LogNormal(mu=Underwriting_Cycle_mu, sigma=Underwriting_Cycle_sigma).generate() - 1
)

# Apply the underwriting cycle to the non_cat losses

non_cat_losses_by_lob = non_cat_losses_by_lob_ind * (1 + (underwriting_cycle_class_impact * underwriting_cycle))

# Market Drivers
market_risk_factor_changes = generate_market_risk_factors(market_params, market_corr)

# apply stochastic inflation

stochastic_inflation = market_risk_factor_changes["inflation_changes"]
inflation_class_impact = ProteusVariable.from_series(Inflation_class_impact)
# non_cat
inflated_non_cat_losses_by_lob = non_cat_losses_by_lob * (1 + inflation_class_impact * stochastic_inflation)
# CAT
inflated_individual_cat_losses_by_lob = individual_cat_losses_by_lob * (
    1 + inflation_class_impact * stochastic_inflation
)

# Initialize updated_inflated_total_losses_by_lob by applying  economic cycle changes
economic_cycle_changes = market_risk_factor_changes["economic_cycle_changes"]
economic_cycle_class_impact = ProteusVariable.from_series(Economic_Cycle_Class_Impact)
# non_cat
updated_inflated_non_cat_losses_by_lob = inflated_non_cat_losses_by_lob * (
    1 + economic_cycle_changes * economic_cycle_class_impact
)
updated_inflated_non_cat_losses_by_lob.show_cdf()


# CAT
updated_inflated_CAT_losses_by_lob = inflated_individual_cat_losses_by_lob * (
    1 + economic_cycle_changes * economic_cycle_class_impact
)

gross_aggregate_cat_losses_by_lob = ProteusVariable(
    "lob", {lob: updated_inflated_CAT_losses_by_lob[lob].aggregate() for lob in lobs}
)

# Netting down the losses
ceded_losses_by_lob = apply_reinsurance(
    updated_inflated_non_cat_losses_by_lob,
    updated_inflated_CAT_losses_by_lob,
    Quota_Share,
    CAT_XoL_Retention,
    CAT_XoL_Limit,
)
ceded_individual_cat_losses_by_lob = ceded_losses_by_lob["Catastrophe"]
net_non_cat_losses_by_lob = updated_inflated_non_cat_losses_by_lob - ceded_losses_by_lob["Non-Catastrophe"]
net_individual_cat_losses_by_lob = updated_inflated_CAT_losses_by_lob - ceded_individual_cat_losses_by_lob
net_aggregate_cat_losses_by_lob = ProteusVariable(
    "lob", {lob: net_individual_cat_losses_by_lob[lob].aggregate() for lob in lobs}
)

# calculate the total net losses
total_net_losses_by_lob = net_non_cat_losses_by_lob + net_aggregate_cat_losses_by_lob

net_losses_by_loss_type = ProteusVariable(
    "loss_type",
    {
        "Non-Catastrophe": net_non_cat_losses_by_lob.sum(),
        "Catastrophe": net_aggregate_cat_losses_by_lob.sum(),
    },
)

# calculate the total gross losses
total_gross_losses_by_lob = updated_inflated_non_cat_losses_by_lob + gross_aggregate_cat_losses_by_lob


# create the total losses

total_gross_losses = total_gross_losses_by_lob.sum()
total_net_losses = net_non_cat_losses_by_lob.sum() + net_aggregate_cat_losses_by_lob.sum()

total_ceded_losses = total_gross_losses - total_net_losses

# reserve risk
future_ultimate_reserves_by_lob = generate_reserve_risk(reserve_params_mean, reserve_params_sigma)

# Apply Stochastic Inflation to Reserve Risk
inflated_reserve_payments_by_lob = future_ultimate_reserves_by_lob * (
    1 + (stochastic_inflation * inflation_class_impact)
)

# apply Market Drivers to Reserve Risk
updated_reserve_payments_by_lob = (
    inflated_reserve_payments_by_lob
    * (1 + economic_cycle_changes * economic_cycle_class_impact)
    * (1 + underwriting_cycle * underwriting_cycle_class_impact)
)

updated_reserve_risk_by_lob = updated_reserve_payments_by_lob - updated_reserve_payments_by_lob.mean()

reserve_risk: StochasticScalar = updated_reserve_risk_by_lob.sum()

# Expense Risk
total_net_premium = net_premium_by_lob.sum()
expense_factor = distributions.Normal(mu=Expense_mu, sigma=Expense_sigma).generate()
expense_risk: StochasticScalar = expense_factor * total_net_premium
# Market
asset_risk_factor_sensitivity = ProteusVariable(
    "risk_factor", {risk: sensitivity for risk, sensitivity in market_params["asset_risk_factor_sensitivity"].items()}
)
investment_return_by_risk_type = asset_risk_factor_sensitivity * market_risk_factor_changes
initial_assets = market_params["initial_assets"]["mu"]
investment_income_by_risk_type = initial_assets * investment_return_by_risk_type
market_risk_init: StochasticScalar = -investment_income_by_risk_type.sum()

# Applying Economic Cycle and Currency Exchange to Market Risk

market_risk = market_risk_init * np.where(
    economic_cycle_changes.values < Market_Impact_EC_Threshold, Market_Impact_EC[1], Market_Impact_EC[0]
)


# Credit Risk
primary_exposure = total_ceded_losses
reinsurer_names = credit_params["reinsurer_rating"].keys()

gross_cat_losses: StochasticScalar = gross_aggregate_cat_losses_by_lob.sum()
credit_risk: StochasticScalar = StochasticScalar(np.zeros(config.n_sims))
for reinsurer in reinsurer_names:
    reinsurer_uniform = distributions.Uniform(0, 1).generate()
    # calculate the probability of default for each reinsurer based on their rating
    rating = credit_params["reinsurer_rating"][reinsurer]
    base_default_probability = credit_params["probability_of_default"][rating]
    default_probability = StochasticScalar(
        np.maximum(
            np.minimum(
                base_default_probability
                * np.where(
                    underwriting_cycle.values > Credit_Impact_UC_Threshold, Credit_Impact_UC[1], Credit_Impact_UC[0]
                )
                * np.where(
                    gross_cat_losses.values > Credit_Impact_CAT_Threshold, Credit_Impact_CAT[1], Credit_Impact_CAT[0]
                )
                * np.where(
                    economic_cycle_changes.values < Credit_Impact_EC_Threshold, Credit_Impact_EC[1], Credit_Impact_EC[0]
                ),
                1,
            ),
            0,
        )
    )
    default_indicator = reinsurer_uniform < default_probability
    lgd = (
        distributions.Beta(
            credit_params["loss_given_default"]["alpha"],
            credit_params["loss_given_default"]["beta"],
        ).generate()
        * default_indicator
    )
    reinsurer_exposure = total_ceded_losses * credit_params["reinsurer_share"][reinsurer]
    credit_risk = credit_risk + reinsurer_exposure * lgd


# Operational Risk
operational_risk_formula = generate_operational_risk(operational_params)

# Applying Underwriting Cycle to Operational Risk
operational_risk_UC = operational_risk_formula * np.where(
    underwriting_cycle.values > Operational_Impact_UC_Threshold, Operational_Impact_UC[1], Operational_Impact_UC[0]
)

# Applying Economic Cycle to Operational Risk
operational_risk_EC = operational_risk_UC * np.where(
    economic_cycle_changes.values < Operational_Impact_EC_Threshold, Operational_Impact_EC[1], Operational_Impact_EC[0]
)

# Applying CAT Losses to Operational Risk
operational_risk = operational_risk_EC * np.where(
    gross_cat_losses.values > Operational_Impact_CAT_Threshold, Operational_Impact_CAT[1], Operational_Impact_CAT[0]
)


underwriting_risk: StochasticScalar = total_net_losses - total_net_premium + expense_risk

# produce analysis

produce_analysis(
    underwriting_risk=underwriting_risk,
    reserve_risk=reserve_risk,
    market_risk=market_risk,
    credit_risk=credit_risk,
    operational_risk=operational_risk,
    net_losses_by_loss_type=net_losses_by_loss_type,
    output_filename="Driver_Based_Model_Output.xlsx",
)
