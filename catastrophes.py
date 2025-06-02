import pandas as pd
import math
import pal
from pal.variables import ProteusVariable, FreqSevSims


def load_yelt(filename: str, yelt_n_sims: int) -> ProteusVariable:
    """
    Load the (Year Event Loss Table) YELT data from a CSV file.

    The csv file should contain the Simulation number or "Year" in the column "Sim", and the individual line of business (LOB) losses in the other columns.

    The Simulation number is expected to be 1-based, meaning the first simulation is labeled as 1.
    The function up-samples the YELT data to match the number of simulations specified in the `pal.config.n_sims` configuration.

    Args:
        filename (str): The path to the CSV file containing YELT data.
        yelt_n_sims (int): The number of simulations of the data in the YELT.

    Returns:
        ProteusVariable: A variable containing the catastrophe loss event simulations data structured by line of business (LOB).
    """
    df = pd.read_csv("data/cat_yelt.csv")
    # Ensure the 'Sim' column is present and contains the simulation numbers
    if "Sim" not in df.columns:
        raise ValueError("The input file must contain a 'Sim' column with simulation numbers.")
    # upsample the cat ylts to the correct number of simulations
    ylt_sims = 10000
    up_sample_factor = math.ceil(pal.config.n_sims / ylt_sims)
    sim_index = (df["Sim"].values - 1).repeat(up_sample_factor)  # Adjusting for 0-based index
    lobs = [col for col in df.columns if (col != "Sim" and col != "EventId")]
    individual_cat_losses_by_lob = ProteusVariable(
        "lob",
        {
            lob: FreqSevSims(
                sim_index,
                df[lob].values.repeat(up_sample_factor),
                n_sims=pal.config.n_sims,
            )
            for lob in lobs
        },
    )
    # Ensure the variables are coupled
    var: FreqSevSims
    for var in individual_cat_losses_by_lob:
        var.coupled_variable_group.merge(individual_cat_losses_by_lob[lobs[0]].coupled_variable_group)

    return individual_cat_losses_by_lob
