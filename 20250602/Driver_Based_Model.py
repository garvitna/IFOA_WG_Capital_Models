from pcm import config,distributions
from pcm.frequency_severity import FrequencySeverityModel
from pcm import copulas
from pcm.variables import ProteusVariable
from pcm.stochastic_scalar import StochasticScalar
import plotly.graph_objects as go  # noqa
import numpy as np
import pandas as pd
from math import erf, sqrt
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

config.n_sims = 10000  # Set the number of simulations

file_path = 'Driver_Based_Model.xlsx'

underwriting_params = pd.read_excel(file_path, sheet_name='Underwriting')
CAT_df = pd.read_excel(file_path, sheet_name='Catastrophe')
reserve_params = pd.read_excel(file_path, sheet_name='Reserve')
cat_class_impact = pd.read_excel(file_path, sheet_name='CAT_Class_Impact')
credit_operational_cat = pd.read_excel(file_path, sheet_name='Credit_Operational_CAT')
market_df = pd.read_excel(file_path, sheet_name='Market')
market_class_impact = pd.read_excel(file_path, sheet_name='Market_Class_Impact')
credit_operational_market_ec = pd.read_excel(file_path, sheet_name='Credit_Operational_Market_EC')
credit_df = pd.read_excel(file_path, sheet_name='Credit')
operational_df = pd.read_excel(file_path, sheet_name='Operational')
underwriting_cycle_df = pd.read_excel(file_path, sheet_name='Underwriting_Cycle')
underwriting_cycle_params = dict(zip(underwriting_cycle_df['parameter_name'], underwriting_cycle_df['value']))
underwriting_cycle_class_impact = pd.read_excel(file_path, sheet_name='Underwriting_Cycle_Class')
credit_operational_uc = pd.read_excel(file_path, sheet_name='Credit_Operational_UC')
inflation_df = pd.read_excel(file_path, sheet_name='Inflation')
inflation_params = dict(zip(inflation_df['parameter_name'], inflation_df['value']))
inflation_class_impact = pd.read_excel(file_path, sheet_name='Inflation_Class')
reinsurance_params = pd.read_excel(file_path, sheet_name='Reinsurance')
expense_df = pd.read_excel(file_path, sheet_name='Expense Inputs')
discount_curve = pd.read_excel(file_path, sheet_name='Yield Curve')
payment_pattern_df = pd.read_excel(file_path, sheet_name='Payment Pattern')

#Process CAT Parameters
CAT_params = dict(zip(CAT_df['parameter_name'], CAT_df['value']))


lobs = underwriting_params['lob'].tolist()
Poisson_Parameters = underwriting_params['Poisson_Parameter'].tolist()
GDP_Shape_Parameters = underwriting_params['GDP_Shape_Parameter'].tolist()
GDP_Scale_Parameters = underwriting_params['GDP_Scale_Parameter'].tolist()
GDP_Loc_Parameters = underwriting_params['GDP_Loc_Parameter'].tolist()
Gamma_Alpha_Parameters = underwriting_params['Gamma_Alpha_Parameter'].tolist()
Gamma_Theta_Parameters = underwriting_params['Gamma_Theta_Parameter'].tolist()
reserve_params_mean = reserve_params['Reserve_Mean'].tolist()
reserve_params_sigma = reserve_params['Reserve_Sigma'].tolist()
reserve_risk_min = reserve_params['Reserve_Risk_Minimum'].tolist()
n_events = int(CAT_params['n_events'])
CAT_frequency_ll = CAT_params['CAT_frequency_ll']
CAT_frequency_ul = CAT_params['CAT_frequency_ul']
CAT_severity_ll = CAT_params['CAT_severity_ll']
CAT_severity_ul = CAT_params['CAT_severity_ul']
CAT_Coeff_Var= CAT_params['CAT_CV']
CAT_Class_Impact = cat_class_impact['CAT_Impact'].tolist()
Credit_Impact_CAT_Threshold = credit_operational_cat.loc[credit_operational_cat['parameter_name'] == 'Credit_Impact_CAT_Threshold', 'value'].values[0]
Operational_Impact_CAT_Threshold = credit_operational_cat.loc[credit_operational_cat['parameter_name'] == 'Operational_Impact_CAT_Threshold', 'value'].values[0]
Credit_Impact_CAT = credit_operational_cat['Credit_Impact_CAT'].tolist()
Operational_Impact_CAT = credit_operational_cat['Operational_Impact_CAT'].tolist()
Economic_Cycle_Class_Impact = market_class_impact['Economic_Cycle_Impact'].tolist()
Currency_Exchange_Class_Impact = market_class_impact['Currency_Exchange_Impact'].tolist()
Credit_Impact_EC_Threshold = credit_operational_market_ec.loc[credit_operational_market_ec['parameter_name'] == 'Credit_Impact_EC_Threshold', 'value'].values[0]
Operational_Impact_EC_Threshold = credit_operational_market_ec.loc[credit_operational_market_ec['parameter_name'] == 'Operational_Impact_EC_Threshold', 'value'].values[0]
Market_Impact_EC_Threshold = credit_operational_market_ec.loc[credit_operational_market_ec['parameter_name'] == 'Market_Impact_EC_Threshold', 'value'].values[0]
Credit_Impact_EC = credit_operational_market_ec['Credit_Impact_EC'].tolist()
Operational_Impact_EC = credit_operational_market_ec['Operational_Impact_EC'].tolist()
Market_Impact_EC = credit_operational_market_ec['Market_Impact_EC'].tolist()
Underwriting_Cycle_mu = underwriting_cycle_params['Underwriting_Cycle_mu']
Underwriting_Cycle_sigma = underwriting_cycle_params['Underwriting_Cycle_sigma']
Underwriting_Cycle_class_impact = underwriting_cycle_class_impact['Underwriting_Cycle_Impact'].tolist()
Credit_Impact_UC_Threshold = credit_operational_uc.loc[credit_operational_uc['parameter_name'] == 'Credit_Impact_UC_Threshold', 'value'].values[0]
Operational_Impact_UC_Threshold = credit_operational_uc.loc[credit_operational_uc['parameter_name'] == 'Operational_Impact_UC_Threshold', 'value'].values[0]
Credit_Impact_UC = credit_operational_uc['Credit_Impact_UC'].tolist()
Operational_Impact_UC = credit_operational_uc['Operational_Impact_UC'].tolist()
Inflation_class_impact = inflation_class_impact['Inflation_Impact'].tolist()
Quota_Share = reinsurance_params['Quota_Share'].tolist()
Large_XoL_Retention = reinsurance_params['Large_XoL_Retention'].tolist()
Large_XoL_Limit = reinsurance_params['Large_XoL_Limit'].tolist()
CAT_XoL_Retention = reinsurance_params['CAT_XoL_Retention'].tolist()
CAT_XoL_Limit = reinsurance_params['CAT_XoL_Limit'].tolist()
Inflation_mu = inflation_params['Inflation_mu']
Inflation_sigma = inflation_params['Inflation_sigma']
year = discount_curve['Year'].tolist()
Yield_Curve = discount_curve['Yield_Curve'].tolist()
Payment_Pattern = payment_pattern_df.set_index("Year")



#Process Expense Parameters
expense_params = dict(zip(expense_df['parameter_name'], expense_df['value']))
Expense_mu = expense_params['Expense_mu']
Expense_sigma = expense_params['Expense_sigma']


# Process Market Parameters
market_params = {}
for _, row in market_df.iterrows():
    category = row['main_category']
    param = row['parameter_name']
    value = row['value']
    if category not in market_params:
        market_params[category] = {}
    market_params[category][param] = value

# Process Credit Parameters
credit_params = {}
for _, row in credit_df.iterrows():
    category = row['main_category']
    param = row['parameter_name']
    value = row['value']
    if category not in credit_params:
        credit_params[category] = {}
    credit_params[category][param] = value

# Process Operational Parameters
operational_params = {}
for _, row in operational_df.iterrows():
    param = row['parameter_name']
    value = row['value']
    operational_params[param] = value    


# Create the frequency and severity models for each class

#Underwriting Risk

# Generate the individual large losses by class
individual_large_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        name: FrequencySeverityModel(
            distributions.Poisson(mean=Poisson_Parameters[idx]),
            distributions.GPD(shape=GDP_Shape_Parameters[idx], scale=GDP_Scale_Parameters[idx], loc=GDP_Loc_Parameters[idx]),
        ).generate()
        for idx, name in enumerate(lobs)
    },
)
# Generate the attritional losses by class
attritional_losses_by_lob_ind = ProteusVariable(
    "class",
    values={
        lob: distributions.Gamma(alpha=Gamma_Alpha_Parameters[idx], theta=Gamma_Theta_Parameters[idx]).generate()
        for idx, lob in enumerate(lobs)
    },
)

losses_with_LAE = individual_large_losses_by_lob * 1.05

# create the aggregate losses by class
aggregate_large_losses_by_class_ind = ProteusVariable(
    "class", {name: losses_with_LAE[name].aggregate() for name in lobs}
)

# Generate the CAT losses by class

# Event catalog
event_ids = [f"Event_{i+1}" for i in range(n_events)]
frequencies = np.random.uniform(CAT_frequency_ll, CAT_frequency_ul, size=n_events)  # Avg occurrences per year
mean_losses = np.random.uniform(CAT_severity_ll, CAT_severity_ul, size=n_events)  # Mean losses in $
n_years = config.n_sims  # Number of years to simulate

#Generate YELT
yelt = pd.DataFrame(0.0, index=np.arange(n_years), columns=event_ids)

for i, (event_id, freq, mean_loss) in enumerate(zip(event_ids, frequencies, mean_losses)):
    # Lognormal parameters
    sigma = np.sqrt(np.log(1 + CAT_Coeff_Var**2))
    mu = np.log(mean_loss) - 0.5 * sigma**2
    
    for year in range(n_years):
        n_occurrences = np.random.poisson(freq)
        if n_occurrences > 0:
            losses = np.random.lognormal(mean=mu, sigma=sigma, size=n_occurrences)
            yelt.loc[year, event_id] = np.sum(losses)

yelt["CAT_losses"] = yelt.sum(axis=1)
CAT_losses = yelt["CAT_losses"]


#Allocating YELT to LoBs

CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: StochasticScalar(CAT_losses * CAT_Class_Impact[idx])
        for idx, lob in enumerate(lobs)
    },
)

   
# Generate the underwriting cycle by class
# As James Toller mentioned, I have used lognormal distribution instead of normal distribution for the underwriting cycle

underwriting_cycle = distributions.LogNormal(mu=Underwriting_Cycle_mu, sigma=Underwriting_Cycle_sigma).generate()

#Market Drivers   
economic_cycle_changes = distributions.LogNormal(market_params["economic_cycle"]["mu"], market_params["economic_cycle"]["sigma"]).generate()
currency_exchange_rate_changes = distributions.Normal(market_params["currency_exchange_rate"]["omega"], market_params["currency_exchange_rate"]["beta"]).generate()


# Apply the underwriting cycle to the attritional and large losses

attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: attritional_losses_by_lob_ind[lob] * (1 + (underwriting_cycle * Underwriting_Cycle_class_impact[idx]))
        for idx, lob in enumerate(lobs)
    },
)

aggregate_large_losses_by_class = ProteusVariable(
    "class",
    values={
        lob: aggregate_large_losses_by_class_ind[lob] * (1 + (underwriting_cycle * Underwriting_Cycle_class_impact[idx]))
        for idx, lob in enumerate(lobs)
    },
)

# apply stochastic inflation

stochastic_inflation = distributions.Normal(Inflation_mu, Inflation_sigma).generate()

#attritional
inflated_attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: attritional_losses_by_lob[lob] * (1 + (stochastic_inflation * Inflation_class_impact[idx]))
        for idx, lob in enumerate(lobs)
    },
)

#large
inflated_large_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: aggregate_large_losses_by_class[lob] * (1 + (stochastic_inflation * Inflation_class_impact[idx]))
        for idx, lob in enumerate(lobs)
    },
)

#CAT
inflated_CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: CAT_losses_by_lob[lob] * (1 + (stochastic_inflation * Inflation_class_impact[idx]))
        for idx, lob in enumerate(lobs)
    },
)


# Initialize updated_inflated_total_losses_by_lob by applying currency exchange rate changes and economic cycle changes

#attritional
updated_inflated_attritional_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        lob: inflated_attritional_losses_by_lob[lob] * Currency_Exchange_Class_Impact[idx] * Economic_Cycle_Class_Impact[idx]
        for idx, lob in enumerate(lobs) 
    }
)

#Large
updated_inflated_large_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        lob: inflated_large_losses_by_lob[lob] * Currency_Exchange_Class_Impact[idx] * Economic_Cycle_Class_Impact[idx]
        for idx, lob in enumerate(lobs)
    }
)

#CAT
updated_inflated_CAT_losses_by_lob = ProteusVariable(
    dim_name="class",
    values={
        lob: inflated_CAT_losses_by_lob[lob] * Currency_Exchange_Class_Impact[idx] * Economic_Cycle_Class_Impact[idx]
        for idx, lob in enumerate(lobs)
    }
)


#Netting down the losses
#Quota Share

Net_attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: updated_inflated_attritional_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx, lob in enumerate(lobs)
    },
)

Net_QS_aggregate_large_losses_by_class = ProteusVariable(
    "class",
    values={
        lob: updated_inflated_large_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx, lob in enumerate(lobs)
    },
)

Net_QS_CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: updated_inflated_CAT_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx, lob in enumerate(lobs)
    },
)

# Apply the XoL to the aggregate large and CAT losses

Net_aggregate_large_losses_by_class = ProteusVariable(
    "class",
    values={
        lob:StochasticScalar(np.minimum(Net_QS_aggregate_large_losses_by_class[lob].values, Large_XoL_Retention[idx]) + np.maximum(Net_QS_aggregate_large_losses_by_class[lob].values-Large_XoL_Limit[idx], 0))       
        for idx, lob in enumerate(lobs)
    },
)

Net_CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob:StochasticScalar(np.minimum(Net_QS_CAT_losses_by_lob[lob].values, CAT_XoL_Retention[idx]) + np.maximum(Net_QS_CAT_losses_by_lob[lob].values-CAT_XoL_Limit[idx], 0))
        for idx, lob in enumerate(lobs)
    },
)


# calculate the total net losses
total_losses_by_lob = Net_aggregate_large_losses_by_class + Net_attritional_losses_by_lob + Net_CAT_losses_by_lob

# calculate the total gross losses
total_gross_losses_by_lob = updated_inflated_attritional_losses_by_lob + updated_inflated_large_losses_by_lob + updated_inflated_CAT_losses_by_lob


# create the total losses
   
total_gross_losses = total_gross_losses_by_lob.sum()
total_inflated_losses = Net_attritional_losses_by_lob.sum() + Net_aggregate_large_losses_by_class.sum() + Net_CAT_losses_by_lob.sum()
total_ceded_losses = total_gross_losses - total_inflated_losses

underwriting_risk = StochasticScalar(
    values=total_inflated_losses.values  
)


#reserve risk

reserve_risk_values = {lob: [] for lob in lobs}  # accumulate reserve risk per LoB

for sim in range(config.n_sims):
    # Generate noise for all LoBs
    noise = np.column_stack([
        np.random.lognormal(0, reserve_params_sigma[idx], len(Payment_Pattern)) 
        for idx in range(len(lobs))
    ])

    # Compute expected cashflows
    incremental_payment_pattern = Payment_Pattern.diff().fillna(Payment_Pattern.iloc[0])
    expected_cashflows = reserve_params_mean * incremental_payment_pattern

    # Actual cashflows per simulation
    actual_cashflows = expected_cashflows.values * noise
    actual_cashflows_df = pd.DataFrame(actual_cashflows, columns=lobs, index=Payment_Pattern.index)

    ber_t0 = actual_cashflows_df.sum()
    payment_year_1 = actual_cashflows_df.iloc[0]
    ber_t1 = actual_cashflows_df.iloc[1:].sum()
    reserve_risk_lob = np.minimum(np.maximum(ber_t1 - (ber_t0 - payment_year_1),0),reserve_risk_min)

    for lob in lobs:
        reserve_risk_values[lob].append(reserve_risk_lob[lob])

# Convert results into ProteusVariable
reserve_risk_by_lob = ProteusVariable(
    dim_name="class",
    values={
        lob: StochasticScalar(values=np.array(reserve_risk_values[lob]))
        for lob in lobs
    }
)

#Apply Stochastic Inflation to Reserve Risk
for idx, lob in enumerate(lobs):
    reserve_risk_by_lob = reserve_risk_by_lob * (1 + (stochastic_inflation * Inflation_class_impact[idx]))

# apply Market Drivers to Reserve Risk
for idx, lob in enumerate(lobs):
    reserve_risk_by_lob = reserve_risk_by_lob * Currency_Exchange_Class_Impact[idx] * (1+(underwriting_cycle * Underwriting_Cycle_class_impact[idx]))

reserve_risk=reserve_risk_by_lob.sum()

#Expense Risk
expense_factor = distributions.LogNormal(mu=Expense_mu, sigma=Expense_sigma).generate()
expense_risk = (expense_factor-1) * (total_gross_losses + reserve_risk.values)


#Market Risk   
initial_assets = distributions.Normal(market_params["initial_assets"]["mu"], market_params["initial_assets"]["sigma"]).generate()
adjusted_assets = np.maximum(initial_assets.values, market_params["initial_assets"]["lower_limit"])
interest_rate_changes = distributions.Normal(market_params["interest_rate"]["long_term_mean_level"], market_params["interest_rate"]["volatility"]).generate()
equity_price_changes = distributions.Normal(market_params["equity_price"]["drift"], market_params["equity_price"]["volatility"]).generate()
currency_exchange_rate_changes = distributions.Normal(market_params["currency_exchange_rate"]["omega"], market_params["currency_exchange_rate"]["beta"]).generate()
commodity_price_changes = distributions.Normal(market_params["commodity_price"]["long_term_mean_level"], market_params["commodity_price"]["volatility"]).generate()
credit_spread = distributions.Normal(market_params["credit_spread"]["mu"], market_params["credit_spread"]["sigma"]).generate()

total_market_risks = np.maximum(
    interest_rate_changes.values + equity_price_changes.values + currency_exchange_rate_changes.values + commodity_price_changes.values, 0
)


Market_risk_init = StochasticScalar(
    values=total_market_risks*adjusted_assets
)


#Applying Economic Cycle and Currency Exchange to Market Risk

Market_risk = Market_risk_init * np.where(
            economic_cycle_changes.values > Market_Impact_EC_Threshold,
            Market_Impact_EC[1],
            Market_Impact_EC[0]
        )



# Credit Risk   
primary_exposure = total_ceded_losses
high_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["high_quality"]).generate()
medium_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["medium_quality"]).generate()
low_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["low_quality"]).generate()
lgd = distributions.Beta(credit_params["loss_given_default"]["alpha"], credit_params["loss_given_default"]["beta"]).generate()
credit_risk_formula = primary_exposure * (high_quality_pd * lgd + medium_quality_pd * lgd + low_quality_pd * lgd)

#Applying Underwriting Cycle to Credit Risk
credit_risk_UC = credit_risk_formula * np.where(
            underwriting_cycle.values > Credit_Impact_UC_Threshold,
            Credit_Impact_UC[1],
            Credit_Impact_UC[0]
        )

#Applying Economic Cycle to Credit Risk
credit_risk_EC = credit_risk_UC * np.where(
            economic_cycle_changes.values > Credit_Impact_EC_Threshold,
            Credit_Impact_EC[1],    
            Credit_Impact_EC[0]
        )

#Applying CAT Losses to Credit Risk
credit_risk = credit_risk_EC * np.where(
        CAT_losses.values > Credit_Impact_CAT_Threshold,
        Credit_Impact_CAT[1],
        Credit_Impact_CAT[0]
        )

    
#Operational Risk
operational_freq = distributions.Poisson(mean=operational_params["frequency_lambda"]).generate()
operational_sev = distributions.LogNormal(operational_params["severity_mean"], operational_params["severity_stddev"]).generate()
operational_risk_formula = operational_sev * operational_freq

#Applying Underwriting Cycle to Operational Risk
operational_risk_UC = operational_risk_formula * np.where(
            underwriting_cycle.values > Operational_Impact_UC_Threshold,
            Operational_Impact_UC[1],
            Operational_Impact_UC[0]
        )

#Applying Economic Cycle to Operational Risk
operational_risk_EC = operational_risk_UC * np.where(
            economic_cycle_changes.values > Operational_Impact_EC_Threshold,
            Operational_Impact_EC[1],
            Operational_Impact_EC[0]
        )

#Applying CAT Losses to Operational Risk
operational_risk = operational_risk_EC * np.where(
            CAT_losses.values > Operational_Impact_CAT_Threshold,
            Operational_Impact_CAT[1],
            Operational_Impact_CAT[0]
        )

       
#combining risks

combined_risk_values = underwriting_risk.values + reserve_risk.values + expense_risk.values + Market_risk.values + credit_risk.values + operational_risk.values
combined_risk = StochasticScalar(values=combined_risk_values)

#Show CDFs
underwriting_risk.show_cdf(title="Underwriting Risk")
reserve_risk.show_cdf(title="Reserve Risk")
expense_risk.show_cdf(title="Expense Risk")
Market_risk.show_cdf(title="Market Risk")
credit_risk.show_cdf(title="Credit Risk")
operational_risk.show_cdf(title="Operational Risk")
combined_risk.show_cdf(title="Combined Risk")    

#Percentiles
selected_percentile = 99.5
Percentiles = [5,10,25,50,75,90,95,99.5]

Capital_Requirement = combined_risk.percentile(selected_percentile)
#print(f"Capital Requirement at {selected_percentile} percentile: {Capital_Requirement}")

#Table of VaR and TVaR at different percentiles
Combined_Risk_VaR = combined_risk.percentile(Percentiles)
Combined_Risk_TVaR = combined_risk.tvar(Percentiles)
underwriting_risk_VaR = underwriting_risk.percentile(Percentiles)
underwriting_risk_TVaR = underwriting_risk.tvar(Percentiles)
reserve_risk_VaR = reserve_risk.percentile(Percentiles)
reserve_risk_TVaR = reserve_risk.tvar(Percentiles)
expense_risk_VaR = expense_risk.percentile(Percentiles)
expense_risk_TVaR = expense_risk.tvar(Percentiles)
Market_risk_VaR = Market_risk.percentile(Percentiles)
Market_risk_TVaR = Market_risk.tvar(Percentiles)
credit_risk_VaR = credit_risk.percentile(Percentiles)
credit_risk_TVaR = credit_risk.tvar(Percentiles)
operational_risk_VaR = operational_risk.percentile(Percentiles)
operational_risk_TVaR = operational_risk.tvar(Percentiles)
Net_attritional_losses_VaR = Net_attritional_losses_by_lob.sum().percentile(Percentiles)
Net_attritional_losses_TVaR = Net_attritional_losses_by_lob.sum().tvar(Percentiles)
Net_aggregate_large_losses_VaR = Net_aggregate_large_losses_by_class.sum().percentile(Percentiles)
Net_aggregate_large_losses_TVaR = Net_aggregate_large_losses_by_class.sum().tvar(Percentiles)
Net_CAT_losses_VaR = Net_CAT_losses_by_lob.sum().percentile(Percentiles)
Net_CAT_losses_TVaR = Net_CAT_losses_by_lob.sum().tvar(Percentiles)
percentile_data = pd.DataFrame({
    "Percentile": Percentiles,
    "Combined Risk VaR": Combined_Risk_VaR,
    "Combined Risk TVaR": Combined_Risk_TVaR,
    "Underwriting Risk VaR": underwriting_risk_VaR,
    "Underwriting Risk TVaR": underwriting_risk_TVaR,
    "Reserve Risk VaR": reserve_risk_VaR,
    "Reserve Risk TVaR": reserve_risk_TVaR,
    "Expense Risk VaR": expense_risk_VaR,
    "Expense Risk TVaR": expense_risk_TVaR,
    "Market Risk VaR": Market_risk_VaR,
    "Market Risk TVaR": Market_risk_TVaR,
    "Credit Risk VaR": credit_risk_VaR,
    "Credit Risk TVaR": credit_risk_TVaR,
    "Operational Risk VaR": operational_risk_VaR,
    "Operational Risk TVaR": operational_risk_TVaR,
    "Net Attritional Losses VaR": Net_attritional_losses_VaR,
    "Net Attritional Losses TVaR": Net_attritional_losses_TVaR,
    "Net Aggregate Large Losses VaR": Net_aggregate_large_losses_VaR,
    "Net Aggregate Large Losses TVaR": Net_aggregate_large_losses_TVaR,
    "Net CAT Losses VaR": Net_CAT_losses_VaR,
    "Net CAT Losses TVaR": Net_CAT_losses_TVaR   
})

#Scatter Plot of Various Risks
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(underwriting_risk), y=np.array(reserve_risk), mode='markers', name='Reserve Risk'))
fig.add_trace(go.Scatter(x=np.array(underwriting_risk), y=np.array(expense_risk), mode='markers', name='Expense Risk'))
fig.add_trace(go.Scatter(x=np.array(underwriting_risk), y=np.array(Market_risk), mode='markers', name='Market Risk'))
fig.add_trace(go.Scatter(x=np.array(underwriting_risk), y=np.array(credit_risk), mode='markers', name='Credit Risk'))
fig.add_trace(go.Scatter(x=np.array(underwriting_risk), y=np.array(operational_risk), mode='markers', name='Operational Risk'))
fig.update_layout(title='Scatter Plot of Various Risks',
                  xaxis_title='Total Inflated Losses',
                  yaxis_title='Risk Values')
fig.show()


#Calculate Spearman's Rank Correlation Coefficient Matrix
def calculate_spearman_correlation(data):
    correlation_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            correlation_matrix.iloc[i, j] = spearmanr(data.iloc[:, i], data.iloc[:, j])[0]
    return correlation_matrix

losses_data = pd.DataFrame({
    "Underwriting Risk": underwriting_risk.values,
    "Reserve Risk": reserve_risk.values,
    "Expense Risk": expense_risk.values,
    "Market Risk": Market_risk.values,
    "Credit Risk": credit_risk.values,
    "Operational Risk": operational_risk.values
})

Spearman_Correlation_Matrix = calculate_spearman_correlation(losses_data)


#Plotting Pearson Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(losses_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", xticklabels=losses_data.columns, yticklabels=losses_data.columns)
plt.title("Correlation Heatmap of Risk Factors")
plt.show()


#Contriution of each risk to the VaR

contribution_data_VAR = pd.DataFrame({
    "Underwriting Risk": [underwriting_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))],
    "Reserve Risk": [reserve_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))],
    "Expense Risk": [expense_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))],
    "Market Risk": [Market_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))],
    "Credit Risk": [credit_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))],
    "Operational Risk": [operational_risk.percentile(selected_percentile)/(underwriting_risk.percentile(selected_percentile)+ reserve_risk.percentile(selected_percentile)+
    expense_risk.percentile(selected_percentile)+ Market_risk.percentile(selected_percentile)+ credit_risk.percentile(selected_percentile)+ operational_risk.percentile(selected_percentile))]
})

#print("Contribution of Each Risk to VaR:")
#print(contribution_data_VAR)

#Contriution of each risk to the tVaR

contribution_data_tVAR = pd.DataFrame({
    "Underwriting Risk": [underwriting_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))],
    "Reserve Risk": [reserve_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))],
    "Expense Risk": [expense_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))],
    "Market Risk": [Market_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))],
    "Credit Risk": [credit_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))],
    "Operational Risk": [operational_risk.tvar(selected_percentile)/(underwriting_risk.tvar(selected_percentile)+ reserve_risk.tvar(selected_percentile)+
    expense_risk.tvar(selected_percentile)+ Market_risk.tvar(selected_percentile)+ credit_risk.tvar(selected_percentile)+ operational_risk.tvar(selected_percentile))]
})

#Plotting joint exceedance probabilities

thresholds=np.percentile(losses_data.values, selected_percentile, axis=0)

joint_exceedance_matrix = np.zeros((len(thresholds), len(thresholds)))

for i in range(len(thresholds)):
    for j in range(len(thresholds)):
        exceed_i = losses_data.iloc[:, i] > thresholds[i]
        exceed_j = losses_data.iloc[:, j] > thresholds[j]
        joint_exceedance_matrix[i, j] = np.mean(exceed_i.to_numpy() & exceed_j.to_numpy())

joint_exceedance_df = pd.DataFrame(joint_exceedance_matrix, 
                                   index=losses_data.columns, 
                                   columns=losses_data.columns)

plt.figure(figsize=(8, 6))
sns.heatmap(joint_exceedance_df, annot=True, cmap="coolwarm", fmt=".4f")
plt.title("Pairwise Joint Exceedance Probabilities")
plt.show()

#Applying Discounting to the losses
incremental_payment_pattern = Payment_Pattern.diff()
incremental_payment_pattern = incremental_payment_pattern.fillna(Payment_Pattern.iloc[0])

year1= np.array(incremental_payment_pattern.index)

mean_term = np.sum(incremental_payment_pattern.values * year1[:,np.newaxis],axis=0) / np.sum(incremental_payment_pattern.values,axis=0)
mean_term_df = pd.Series(mean_term, index=incremental_payment_pattern.columns)

interpolated_yield_curve = np.interp(mean_term_df, year1, Yield_Curve)
interpolated_yield_curve_df = pd.Series(interpolated_yield_curve, index=incremental_payment_pattern.columns)

discount_factor = np.exp(-interpolated_yield_curve_df * mean_term_df)
discount_factor_df = pd.Series(discount_factor, index=incremental_payment_pattern.columns)  

discounted_losses = ProteusVariable(
    dim_name="class",
    values={
        name: (total_losses_by_lob[name].values + reserve_risk_by_lob[name].values) * discount_factor_df[name]
        for name in lobs
    },
)

discounted_net_losses = discounted_losses.sum()

discounted_combined_risk = discounted_net_losses + expense_risk.values + credit_risk.values + operational_risk.values

Discounting_Impact = discounted_combined_risk / combined_risk.values

#print(f"Average Discounting Impact: {np.mean(Discounting_Impact)}")

#Outputting the results to an excel file

total_risks = pd.DataFrame({
    "Sim_Index" : np.arange(1, config.n_sims + 1),
    "Underwriting Risk": underwriting_risk.values,
    "Reserve Risk": reserve_risk.values,
    "Expense Risk": expense_risk.values,
    "Market Risk": Market_risk.values,
    "Credit Risk": credit_risk.values,
    "Operational Risk": operational_risk.values,
    "Combined Risk": combined_risk.values
})

with pd.ExcelWriter('Driver_Based_Model_Output.xlsx') as writer:
    pd.DataFrame({"Selected_Percentile" : [selected_percentile],"Capital" : [Capital_Requirement]}).to_excel(writer, sheet_name = "Capital_Reqirement", index=False)
    total_risks.to_excel(writer, sheet_name='Total Risks', index=False)
    percentile_data.to_excel(writer, sheet_name='VaR and TVaR', index=False)
    Spearman_Correlation_Matrix.to_excel(writer, sheet_name = "Spearman_Corr_Matrix", index=True)
    contribution_data_VAR.to_excel(writer, sheet_name = "Contribution_data_VAR", index=False)
    contribution_data_tVAR.to_excel(writer, sheet_name = "Contribution_data_tVAR", index=False)
    joint_exceedance_df.to_excel(writer, sheet_name = "Joint_Exceedance_Probabilities", index=True)
    pd.DataFrame({"Average_Discounting_Impact" : [np.mean(Discounting_Impact)]}).to_excel(writer, sheet_name = "Discounting_Impact", index=False)
    yelt.to_excel(writer, sheet_name='YELT', index=False)
