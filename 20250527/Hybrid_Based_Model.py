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
from types import MethodType

config.n_sims = 10000

file_path = 'Hybrid_Based_Model.xlsx'

underwriting_df = pd.read_excel(file_path, sheet_name='Underwriting')
CAT_df = pd.read_excel(file_path, sheet_name='Catastrophe')
reserve_df = pd.read_excel(file_path, sheet_name='Reserve')
market_df = pd.read_excel(file_path, sheet_name='Market')
credit_df = pd.read_excel(file_path, sheet_name='Credit')
operational_df = pd.read_excel(file_path, sheet_name='Operational')
inflation_df = pd.read_excel(file_path, sheet_name='Inflation')
inflation_class_df = pd.read_excel(file_path, sheet_name='Inflation_Class')
reinsurance_df = pd.read_excel(file_path, sheet_name='Reinsurance')
corr_df = pd.read_excel(file_path,sheet_name="Correlation Matrix", index_col=0)
corr_input_df = pd.read_excel(file_path,sheet_name="Correlation Inputs")
expense_df = pd.read_excel(file_path, sheet_name='Expense Inputs')
discount_curve = pd.read_excel(file_path, sheet_name='Yield Curve')
payment_pattern_df = pd.read_excel(file_path, sheet_name='Payment Pattern')

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

# Process Inflation Parameters
inflation_params = dict(zip(inflation_df['parameter_name'], inflation_df['value']))

# Process Inflation Class Impact
inflation_class_impact = dict(zip(inflation_class_df['lobs'], inflation_class_df['Inflation_Impact']))


# Process Correlation Inputs Parameters
corr_inputs_params = dict(zip(corr_input_df['parameter_name'], corr_input_df['value']))

#Process Expense Parameters
expense_params = dict(zip(expense_df['parameter_name'], expense_df['value']))

#Process CAT Parameters
CAT_params = dict(zip(CAT_df['parameter_name'], CAT_df['value']))


# Underwriting and Reserve parameters already loaded as DataFrames
# Reinsurance parameters already loaded as DataFrames

#Making individual lists for each parameter
lobs = underwriting_df['lobs'].tolist()
Poisson_Parameters = underwriting_df['Poisson_Parameter'].tolist()
GDP_Shape_Parameters = underwriting_df['GDP_Shape_Parameter'].tolist()
GDP_Scale_Parameters = underwriting_df['GDP_Scale_Parameter'].tolist()
GDP_Loc_Parameters = underwriting_df['GDP_Loc_Parameter'].tolist()
Gamma_Alpha_Parameters = underwriting_df['Gamma_Alpha_Parameter'].tolist()
Gamma_Theta_Parameters = underwriting_df['Gamma_Theta_Parameter'].tolist()
CAT_Class_Impact = underwriting_df['CAT_Class_Impact'].tolist()
n_events = int(CAT_params['n_events'])
CAT_frequency_ll = CAT_params['CAT_frequency_ll']
CAT_frequency_ul = CAT_params['CAT_frequency_ul']
CAT_severity_ll = CAT_params['CAT_severity_ll']
CAT_severity_ul = CAT_params['CAT_severity_ul']
CAT_Coeff_Var= CAT_params['CAT_CV']
reserve_params_mean = reserve_df['Reserve_Mean'].tolist()
reserve_params_sigma = reserve_df['Reserve_Sigma'].tolist()
Inflation_class_impact = inflation_class_df['Inflation_Impact'].tolist()
Quota_Share = reinsurance_df['Quota_Share'].tolist()
Large_XoL_Retention = reinsurance_df['Large_XoL_Retention'].tolist()
Large_XoL_Limit = reinsurance_df['Large_XoL_Limit'].tolist()
CAT_XoL_Retention = reinsurance_df['CAT_XoL_Retention'].tolist()
CAT_XoL_Limit = reinsurance_df['CAT_XoL_Limit'].tolist()
Inflation_mu = inflation_params['mu']
Inflation_sigma = inflation_params['sigma']
correlation_matrix = corr_df.values
Losses_Aggregation_Theta = corr_inputs_params['Losses_Aggregation_Theta']
Risk_Aggregation_Theta = corr_inputs_params['Risk_Aggregation_Theta']
Expense_mu = expense_params['Expense_mu']
Expense_sigma = expense_params['Expense_sigma']
year = discount_curve['Year'].tolist()
Yield_Curve = discount_curve['Yield_Curve'].tolist()
Payment_Pattern = payment_pattern_df.set_index("Year")

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
attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: distributions.Gamma(alpha=Gamma_Alpha_Parameters[idx], theta=Gamma_Theta_Parameters[idx]).generate()
        for idx, lob in enumerate(lobs)
    },
)

losses_with_LAE = individual_large_losses_by_lob * 1.05

# create the aggregate losses by class
aggregate_large_losses_by_class = ProteusVariable(
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


# correlate the attritionl, large and CAT losses. Use a pairwise copula to do this
for lob in lobs:
    copulas.GumbelCopula(theta=Losses_Aggregation_Theta, n=3).apply(
        [aggregate_large_losses_by_class[lob], attritional_losses_by_lob[lob], CAT_losses_by_lob[lob]]
    )

# correlate various LoBs. Use a pairwise copula to do this

Underwriting_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: aggregate_large_losses_by_class[lob] + attritional_losses_by_lob[lob] + CAT_losses_by_lob[lob]
        for lob in lobs 
    },
)

copulas.StudentsTCopula(correlation_matrix, len(lobs)).apply(Underwriting_losses_by_lob)

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


#Netting down the losses
#Quota Share

Net_attritional_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: inflated_attritional_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx,lob in enumerate(lobs)
    },
)

Net_QS_aggregate_large_losses_by_class = ProteusVariable(
    "class",
    values={
        lob: inflated_large_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx,lob in enumerate(lobs)
    },
)

Net_QS_CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob: inflated_CAT_losses_by_lob[lob] * (1 - Quota_Share[idx])
        for idx,lob in enumerate(lobs)
    },
)

# Apply the XoL to the aggregate large and CAT losses

Net_aggregate_large_losses_by_class = ProteusVariable(
    "class",
    values={
        lob:np.minimum(Net_QS_aggregate_large_losses_by_class[lob].values, Large_XoL_Retention[idx]) + np.maximum(Net_QS_aggregate_large_losses_by_class[lob].values-Large_XoL_Limit[idx], 0)       
        for idx, lob in enumerate(lobs)
    },
)

Net_CAT_losses_by_lob = ProteusVariable(
    "class",
    values={
        lob:np.minimum(Net_QS_CAT_losses_by_lob[lob].values, CAT_XoL_Retention[idx]) + np.maximum(Net_QS_CAT_losses_by_lob[lob].values-CAT_XoL_Limit[idx], 0)
        for idx, lob in enumerate(lobs)
    },
)


# calculate the total losses
total_gross_losses_by_lob = inflated_large_losses_by_lob + inflated_attritional_losses_by_lob + inflated_CAT_losses_by_lob
total_net_losses_by_lob = Net_aggregate_large_losses_by_class + Net_attritional_losses_by_lob + Net_CAT_losses_by_lob


total_gross_losses = total_gross_losses_by_lob.sum()
total_net_losses = Net_aggregate_large_losses_by_class.sum() + Net_attritional_losses_by_lob.sum() + Net_CAT_losses_by_lob.sum()
total_ceded_losses = total_gross_losses - total_net_losses

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
    reserve_risk_lob = ber_t1 - (ber_t0 - payment_year_1)

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

copulas.StudentsTCopula(correlation_matrix, len(lobs)).apply(reserve_risk_by_lob)
reserve_risk=reserve_risk_by_lob.sum()

#Expense Risk
expense_factor = distributions.LogNormal(Expense_mu, Expense_sigma).generate()
expense_risk = (expense_factor -1) * (total_gross_losses + reserve_risk.values)


#Market Risk   
initial_assets = distributions.Normal(market_params["initial_assets"]["mu"], market_params["initial_assets"]["sigma"]).generate()
adjusted_assets = np.maximum(initial_assets.values, market_params["initial_assets"]["lower_limit"])
interest_rate_changes = distributions.Normal(market_params["interest_rate"]["long_term_mean_level"], market_params["interest_rate"]["volatility"]).generate()
equity_price_changes = distributions.Normal(market_params["equity_price"]["drift"], market_params["equity_price"]["volatility"]).generate()
currency_exchange_rate_changes = distributions.Normal(market_params["currency_exchange_rate"]["omega"], market_params["currency_exchange_rate"]["beta"]).generate()
commodity_price_changes = distributions.Normal(market_params["commodity_price"]["long_term_mean_level"], market_params["commodity_price"]["volatility"]).generate()

total_market_risks = np.maximum(
    interest_rate_changes.values + equity_price_changes.values + currency_exchange_rate_changes.values + commodity_price_changes.values, 0
)


Market_risk = StochasticScalar(
    values=total_market_risks*adjusted_assets
)

# Credit Risk   
primary_exposure = total_ceded_losses
high_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["high_quality"]).generate()
medium_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["medium_quality"]).generate()
low_quality_pd = distributions.Binomial(1, credit_params["probability_of_default"]["low_quality"]).generate()
lgd = distributions.Beta(credit_params["loss_given_default"]["alpha"], credit_params["loss_given_default"]["beta"]).generate()
credit_risk = primary_exposure * (high_quality_pd * lgd + medium_quality_pd * lgd + low_quality_pd * lgd)
    
#Operational Risk
operational_freq = distributions.Poisson(mean=operational_params["frequency_lambda"]).generate()
operational_sev = distributions.LogNormal(operational_params["severity_mean"], operational_params["severity_stddev"]).generate()
operational_risk = operational_sev * operational_freq

#combining risk
#did not include credit and expense risks as they are dependented on the total net losses and reserve risk

copulas.GumbelCopula(Risk_Aggregation_Theta, 4).apply([total_net_losses,reserve_risk,Market_risk,operational_risk])
combined_risk_values = total_net_losses.values + reserve_risk.values + expense_risk.values + Market_risk.values + credit_risk.values + operational_risk.values 
combined_risk = StochasticScalar(values=combined_risk_values)


#Show CDFs
total_net_losses.show_cdf(title="Underwriting Risk")
reserve_risk.show_cdf(title="Reserve Risk")
expense_risk.show_cdf(title="Expense Risk")
Market_risk.show_cdf(title="Market Risk")
credit_risk.show_cdf(title="Credit Risk")
operational_risk.show_cdf(title="Operational Risk")
combined_risk.show_cdf(title="Combined Risk") 



#Percentiles
Selected_Percentile = 99.5
Percentiles = [5,10,25,50,75,90,95,99.5]

Capital_Requirement = combined_risk.percentile(Selected_Percentile)

#Table of VaR and TVaR at different percentiles
VaR = combined_risk.percentile(Percentiles)
TVaR = combined_risk.tvar(Percentiles)
percentile_data = pd.DataFrame({
    "Percentile": Percentiles,
    "VaR": VaR,
    "TVaR": TVaR
})

    
#Scatter Plot of Various Risks
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(total_net_losses), y=np.array(reserve_risk), mode='markers', name='Reserve Risk'))
fig.add_trace(go.Scatter(x=np.array(total_net_losses), y=np.array(expense_risk), mode='markers', name='Expense Risk'))
fig.add_trace(go.Scatter(x=np.array(total_net_losses), y=np.array(Market_risk), mode='markers', name='Market Risk'))
fig.add_trace(go.Scatter(x=np.array(total_net_losses), y=np.array(credit_risk), mode='markers', name='Credit Risk'))
fig.add_trace(go.Scatter(x=np.array(total_net_losses), y=np.array(operational_risk), mode='markers', name='Operational Risk'))
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
    "Total Inflated Losses": total_net_losses.values,
    "Reserve Risk": reserve_risk.values,
    "Expense Risk": expense_risk.values,
    "Market Risk": Market_risk.values,
    "Credit Risk": credit_risk.values,
    "Operational Risk": operational_risk.values
})

Spearman_Correlation_Matrix = calculate_spearman_correlation(losses_data)

    

#Plotting Pearson Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(np.corrcoef(losses_data.T), annot=True, cmap="coolwarm", fmt=".2f", xticklabels=losses_data.columns, yticklabels=losses_data.columns)
plt.title("Correlation Heatmap of Risk Factors")
plt.show()


#Contriution of each risk to the VaR

contribution_data_VAR = pd.DataFrame({
    "Total Inflated Losses": [total_net_losses.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)],
    "Reserve Risk": [reserve_risk.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)],
    "Expense Risk": [expense_risk.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)],
    "Market Risk": [Market_risk.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)],
    "Credit Risk": [credit_risk.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)],
    "Operational Risk": [operational_risk.percentile(Selected_Percentile)/combined_risk.percentile(Selected_Percentile)]
})

   

#Contriution of each risk to the tVaR

contribution_data_tVAR = pd.DataFrame({
    "Total Inflated Losses": [total_net_losses.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)],
    "Reserve Risk": [reserve_risk.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)],
    "Expense Risk": [expense_risk.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)],
    "Market Risk": [Market_risk.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)],
    "Credit Risk": [credit_risk.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)],
    "Operational Risk": [operational_risk.tvar(Selected_Percentile)/combined_risk.tvar(Selected_Percentile)]
})


#Plotting joint exceedance probabilities

thresholds=np.percentile(losses_data,Selected_Percentile,axis=0)

joint_exceedance_matrix = np.zeros((len(thresholds), len(thresholds)))

for i in range(len(thresholds)):
    for j in range(len(thresholds)):
        exceed_i = losses_data.iloc[:, i] > thresholds[i]
        exceed_j = losses_data.iloc[:, j] > thresholds[j]
        joint_exceedance_matrix[i, j] = np.mean(exceed_i & exceed_j)

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
        name: (total_net_losses_by_lob[name].values + reserve_risk_by_lob[name].values) * discount_factor_df[name]
        for name in lobs
    },
)

discounted_net_losses = discounted_losses.sum()

discounted_combined_risk = discounted_net_losses + expense_risk.values + interest_rate_changes.values + equity_price_changes.values + currency_exchange_rate_changes.values + commodity_price_changes.values + credit_risk.values + operational_risk.values

Discounting_Impact = discounted_combined_risk / combined_risk.values

#Outputting the results to an excel file

total_risks = pd.DataFrame({
    "Sim_Index" : np.arange(1, config.n_sims + 1),
    "Underwriting Risk": total_net_losses.values,
    "Reserve Risk": reserve_risk.values,
    "Expense Risk": expense_risk.values,
    "Market Risk": Market_risk.values,
    "Credit Risk": credit_risk.values,
    "Operational Risk": operational_risk.values,
    "Combined Risk": combined_risk.values
})

with pd.ExcelWriter('Hybrid_Based_Model_Output.xlsx') as writer:
    pd.DataFrame({"Selected_Percentile" : [Selected_Percentile],"Capital" : [Capital_Requirement]}).to_excel(writer, sheet_name = "Capital_Reqirement", index=False)
    total_risks.to_excel(writer, sheet_name='Total Risks', index=False)
    percentile_data.to_excel(writer, sheet_name='VaR and TVaR', index=False)
    Spearman_Correlation_Matrix.to_excel(writer, sheet_name = "Spearman_Corr_Matrix", index=True)
    contribution_data_VAR.to_excel(writer, sheet_name = "Contribution_data_VAR", index=False)
    contribution_data_tVAR.to_excel(writer, sheet_name = "Contribution_data_tVAR", index=False)
    joint_exceedance_df.to_excel(writer, sheet_name = "Joint_Exceedance_Probabilities", index=True)
    pd.DataFrame({"Average_Discounting_Impact" : [np.mean(Discounting_Impact)]}).to_excel(writer, sheet_name = "Discounting_Impact", index=False)
    yelt.to_excel(writer, sheet_name='YELT', index=False)
