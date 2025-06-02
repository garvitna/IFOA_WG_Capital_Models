from pal import config, distributions
from pal.frequency_severity import FrequencySeverityModel
from pal import copulas
from pal.variables import ProteusVariable, StochasticScalar
import numpy as np
import pandas as pd

from catastrophes import load_yelt
from reinsurance import apply_reinsurance
from market_risks import generate_market_risk_factors
from reserves import generate_reserve_risk
from operational_risk import generate_operational_risk


from analysis import produce_analysis

config.n_sims = 10000

file_path = "Copula_Based_Model.xlsx"

underwriting_df = pd.read_excel(file_path, sheet_name="Underwriting", index_col="lob")

reserve_df = pd.read_excel(file_path, sheet_name="Reserve", index_col="lob")
market_df = pd.read_excel(file_path, sheet_name="Market")
credit_df = pd.read_excel(file_path, sheet_name="Credit")
operational_df = pd.read_excel(file_path, sheet_name="Operational")
reinsurance_df = pd.read_excel(file_path, sheet_name="Reinsurance", index_col="lob")
cat_reinsurance_df = pd.read_excel(file_path, sheet_name="Cat Reinsurance", index_col="layer")
corr_df = pd.read_excel(file_path, sheet_name="Correlation Matrix", index_col=0)
market_corr = pd.read_excel(file_path, sheet_name="Market_Correlation", index_col=0)
corr_input_df = pd.read_excel(file_path, sheet_name="Correlation Inputs")
expense_df = pd.read_excel(file_path, sheet_name="Expense Inputs")
discount_curve = pd.read_excel(file_path, sheet_name="Yield Curve")
payment_pattern_df = pd.read_excel(file_path, sheet_name="Payment Pattern", index_col=0)

# Process Market Parameters
market_params = {k: dict(zip(v["parameter_name"], v["value"])) for k, v in market_df.groupby("main_category")}
# Process Credit Parameters
credit_params = {k: dict(zip(v["parameter_name"], v["value"])) for k, v in credit_df.groupby("main_category")}
# Process Operational Parameters
operational_params = dict(zip(operational_df["parameter_name"], operational_df["value"]))
# Process Correlation Inputs Parameters
corr_inputs_params = dict(zip(corr_input_df["parameter_name"], corr_input_df["value"]))
# Process Expense Parameters
expense_params = dict(zip(expense_df["parameter_name"], expense_df["value"]))
# Underwriting and Reserve parameters already loaded as DataFrames
# Reinsurance parameters already loaded as DataFrames

# Making individual lists for each parameter
lobs = underwriting_df.index
net_premium = underwriting_df["Net Premium"]
Poisson_Parameters = underwriting_df["Poisson_Parameter"]
GDP_Shape_Parameters = underwriting_df["GDP_Shape_Parameter"]
GDP_Scale_Parameters = underwriting_df["GDP_Scale_Parameter"]
GDP_Loc_Parameters = underwriting_df["GDP_Loc_Parameter"]
Gamma_Alpha_Parameters = underwriting_df["Gamma_Alpha_Parameter"]
Gamma_Theta_Parameters = underwriting_df["Gamma_Theta_Parameter"]
reserve_params_mean = reserve_df["Reserve_Mean"]
reserve_params_sigma = reserve_df["Reserve_Sigma"]
Quota_Share = reinsurance_df["Quota_Share"]
Large_XoL_Retention = reinsurance_df["Large_XoL_Retention"]
Large_XoL_Limit = reinsurance_df["Large_XoL_Limit"]
CAT_XoL_Retention = cat_reinsurance_df["CAT_XoL_Retention"]
CAT_XoL_Limit = cat_reinsurance_df["CAT_XoL_Limit"]
correlation_matrix = corr_df.values
Losses_Aggregation_Theta = corr_inputs_params["Losses_Aggregation_Theta"]
Risk_Aggregation_Theta = corr_inputs_params["Risk_Aggregation_Theta"]
Expense_mu = expense_params["Expense_mu"]
Expense_sigma = expense_params["Expense_sigma"]
year = discount_curve["Year"]
Yield_Curve = discount_curve["Yield_Curve"]
Payment_Pattern = payment_pattern_df

# Create the frequency and severity models for each class

# Underwriting Risk

# Generate the individual large losses by class
individual_large_losses_by_lob = ProteusVariable(
    dim_name="lob",
    values={
        lob: FrequencySeverityModel(
            distributions.Poisson(mean=Poisson_Parameters[lob]),
            distributions.GPD(
                shape=GDP_Shape_Parameters[lob],
                scale=GDP_Scale_Parameters[lob],
                loc=GDP_Loc_Parameters[lob],
            ),
        ).generate()
        for lob in lobs
    },
)
# Generate the attritional losses by class
attritional_losses_by_lob = ProteusVariable(
    "lob",
    values={
        lob: distributions.Gamma(alpha=Gamma_Alpha_Parameters[lob], theta=Gamma_Theta_Parameters[lob]).generate()
        for lob in lobs
    },
)

# create the aggregate losses by lob
aggregate_large_losses_by_lob = ProteusVariable(
    "lob", {name: individual_large_losses_by_lob[name].aggregate() for name in lobs}
)
# Import the Catastrophe YELT data
individual_cat_losses_by_lob = load_yelt("data/cat_yelt.csv", 10000)
aggregate_cat_losses_by_lob = ProteusVariable(
    "lob", {lob: individual_cat_losses_by_lob[lob].aggregate() for lob in lobs}
)

# correlate the attritional, large losses by lob. Use a pairwise copula to do this
for lob in lobs:
    copulas.GumbelCopula(theta=Losses_Aggregation_Theta, n=2).apply(
        [
            aggregate_large_losses_by_lob[lob],
            attritional_losses_by_lob[lob],
        ]
    )

non_cat_underwriting_losses_by_lob = aggregate_large_losses_by_lob + attritional_losses_by_lob

# correlate the non-cat losses of various LoBs. Use a copula to do this
copulas.StudentsTCopula(correlation_matrix, len(lobs)).apply(non_cat_underwriting_losses_by_lob)
gross_underwriting_loss_by_lob = non_cat_underwriting_losses_by_lob + aggregate_cat_losses_by_lob

# Netting down the losses
ceded_losses_by_loss_type_and_lob = apply_reinsurance(
    attritional_losses_by_lob,
    individual_large_losses_by_lob,
    individual_cat_losses_by_lob,
    Quota_Share,
    Large_XoL_Retention,
    Large_XoL_Limit,
    CAT_XoL_Retention,
    CAT_XoL_Limit,
)
net_attritional_losses_by_lob = attritional_losses_by_lob - ceded_losses_by_loss_type_and_lob["Attritional"]
net_individual_large_losses_by_lob = individual_large_losses_by_lob - ceded_losses_by_loss_type_and_lob["Large"]
net_individual_cat_losses_by_lob = individual_cat_losses_by_lob - ceded_losses_by_loss_type_and_lob["Catastrophe"]
net_aggregate_large_losses_by_lob = ProteusVariable(
    "lob",
    values={lob: net_individual_large_losses_by_lob[lob].aggregate() for lob in lobs},
)

net_aggregate_cat_losses_by_lob = ProteusVariable(
    "lob", {lob: net_individual_cat_losses_by_lob[lob].aggregate() for lob in lobs}
)
net_aggregate_cat_losses: StochasticScalar = net_aggregate_cat_losses_by_lob.sum()


# calculate the total losses
total_gross_losses_by_lob = aggregate_large_losses_by_lob + attritional_losses_by_lob + aggregate_cat_losses_by_lob
total_net_losses_by_lob = (
    net_aggregate_large_losses_by_lob + net_attritional_losses_by_lob + net_aggregate_cat_losses_by_lob
)

net_aggregate_large_losses: StochasticScalar = net_aggregate_large_losses_by_lob.sum()
net_attritional_losses: StochasticScalar = net_attritional_losses_by_lob.sum()
net_losses_by_loss_type = ProteusVariable(
    "loss_type",
    {
        "Attritional": net_attritional_losses,
        "Large": net_aggregate_large_losses,
        "Catastrophe": net_aggregate_cat_losses,
    },
)
total_gross_losses = total_gross_losses_by_lob.sum()
total_net_losses: StochasticScalar = net_losses_by_loss_type.sum()

total_ceded_losses: StochasticScalar = total_gross_losses - total_net_losses

# reserve risk

future_ultimate_reserves_by_lob = generate_reserve_risk(reserve_params_mean, reserve_params_sigma, Payment_Pattern)
reserve_risk_by_lob = future_ultimate_reserves_by_lob - future_ultimate_reserves_by_lob.mean()
copulas.StudentsTCopula(correlation_matrix, len(lobs)).apply(reserve_risk_by_lob)
reserve_risk: StochasticScalar = reserve_risk_by_lob.sum()


# Market Risk
market_risk_factor_changes = generate_market_risk_factors(market_params, market_corr)
# Market
asset_risk_factor_sensitivity = ProteusVariable(
    "risk_factor", {risk: sensitivity for risk, sensitivity in market_params["asset_risk_factor_sensitivity"].items()}
)
investment_return_by_risk_type = asset_risk_factor_sensitivity * market_risk_factor_changes
initial_assets = market_params["initial_assets"]["mu"]
investment_income_by_risk_type = initial_assets * investment_return_by_risk_type
market_risk: StochasticScalar = -investment_income_by_risk_type.sum()

# Credit Risk
reinsurer_names = credit_params["reinsurer_rating"].keys()

gross_cat_losses: StochasticScalar = aggregate_cat_losses_by_lob.sum()
credit_risk: StochasticScalar = StochasticScalar(np.zeros(config.n_sims))
for reinsurer in reinsurer_names:
    # apply a Clayton copula to the reinsurer uniforms and the overall catastrophe losses
    reinsurer_uniform = distributions.Uniform(0, 1).generate()
    copulas.ClaytonCopula(credit_params["cat_copula_parameter"]["theta"], n=2).apply(
        [-gross_cat_losses, reinsurer_uniform]
    )
    # calculate the probability of default for each reinsurer based on their rating
    rating = credit_params["reinsurer_rating"][reinsurer]
    default_probability = credit_params["probability_of_default"][rating]
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

# operational risk
operational_risk = generate_operational_risk(operational_params)

# combining risk
# did not include credit and expense risks as they are dependent on the total net losses and reserve risk

copulas.GumbelCopula(Risk_Aggregation_Theta, n=4).apply([total_net_losses, reserve_risk, market_risk, operational_risk])

# Expense Risk
expense_factor = distributions.Normal(Expense_mu, Expense_sigma).generate()
total_net_premium = net_premium.sum()
expense_risk = expense_factor * (total_net_premium)

underwriting_risk: StochasticScalar = total_net_losses + expense_risk - total_net_premium
combined_risk: StochasticScalar = underwriting_risk + reserve_risk + market_risk + credit_risk + operational_risk


# produce analysis

produce_analysis(
    underwriting_risk=underwriting_risk,
    reserve_risk=reserve_risk,
    market_risk=market_risk,
    credit_risk=credit_risk,
    operational_risk=operational_risk,
    net_losses_by_loss_type=net_losses_by_loss_type,
    output_filename="Copula_Based_Model_Output.xlsx",
)
