from pal import FrequencySeverityModel, distributions, StochasticScalar
import pandas as pd


def generate_operational_risk(operational_params: pd.Series) -> StochasticScalar:
    """Generates the Operational Risk distributions"""
    operational_freq = operational_params["frequency_lambda"]
    operational_risk = (
        FrequencySeverityModel(
            distributions.Poisson(mean=operational_freq),
            distributions.LogNormal(operational_params["severity_mean"], operational_params["severity_stddev"]),
        )
        .generate()
        .aggregate()
    )
    return operational_risk
