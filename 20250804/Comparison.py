# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File Paths
copula_path = "Copula_Based_Model_Output.xlsx"
driver_path = "Driver_Based_Model_Output.xlsx"
hybrid_path = "Hybrid_Based_Model_Output.xlsx"

# Load Excel Files
copula_file = pd.ExcelFile(copula_path)
driver_file = pd.ExcelFile(driver_path)
hybrid_file = pd.ExcelFile(hybrid_path)

# Load Capital Requirement Sheets
copula_capital = pd.read_excel(copula_file, sheet_name="Capital_Requirement")
driver_capital = pd.read_excel(driver_file, sheet_name="Capital_Requirement")
hybrid_capital = pd.read_excel(hybrid_file, sheet_name="Capital_Requirement")

# Load VaR and TVaR Sheets
copula_var = pd.read_excel(copula_file, sheet_name="VaR and TVaR")
driver_var = pd.read_excel(driver_file, sheet_name="VaR and TVaR")
hybrid_var = pd.read_excel(hybrid_file, sheet_name="VaR and TVaR")

# Combine Capital Requirement Data
capital_comparison = pd.DataFrame(
    {
        "Model": ["Copula-Based", "Driver-Based", "Hybrid-Based"],
        "Capital at 99.5%": [
            copula_capital["Capital"].iloc[0],
            driver_capital["Capital"].iloc[0],
            hybrid_capital["Capital"].iloc[0],
        ],
    }
)

# Tag and Combine VaR/TVaR Data
copula_var["Model"] = "Copula-Based"
driver_var["Model"] = "Driver-Based"
hybrid_var["Model"] = "Hybrid-Based"

var_combined = pd.concat([copula_var, driver_var, hybrid_var], ignore_index=True)

# Combined Correlation
copula_corr = pd.read_excel(copula_file, sheet_name="Spearman_Corr_Matrix", index_col=0)
driver_corr = pd.read_excel(driver_file, sheet_name="Spearman_Corr_Matrix", index_col=0)
hybrid_corr = pd.read_excel(hybrid_file, sheet_name="Spearman_Corr_Matrix", index_col=0)
combined_corr = pd.concat(
    [copula_corr, driver_corr, hybrid_corr], axis=1, keys=["Copula-Based", "Driver-Based", "Hybrid-Based"]
)

# Combined Joint Exceedance Probabilities
copula_jep = pd.read_excel(copula_file, sheet_name="Joint_Exceedance_Probabilities", index_col=0)
driver_jep = pd.read_excel(driver_file, sheet_name="Joint_Exceedance_Probabilities", index_col=0)
hybrid_jep = pd.read_excel(hybrid_file, sheet_name="Joint_Exceedance_Probabilities", index_col=0)
combined_jep = pd.concat(
    [copula_jep, driver_jep, hybrid_jep], axis=1, keys=["Copula-Based", "Driver-Based", "Hybrid-Based"]
)

# Plot Capital at 99.5%
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Capital at 99.5%", data=capital_comparison, palette="viridis")
plt.title("Capital at 99.5% Comparison Across Models")
plt.ylabel("Capital Requirement")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot VaR
plt.figure(figsize=(12, 6))
sns.lineplot(data=var_combined, x="Percentile", y="Combined Risk VaR", hue="Model", style="Model", markers=True)
plt.title("VaR Comparison Across Percentiles")
plt.ylabel("VaR")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot TVaR
plt.figure(figsize=(12, 6))
sns.lineplot(data=var_combined, x="Percentile", y="Combined Risk TVaR", hue="Model", style="Model", markers=True)
plt.title("TVaR Comparison Across Percentiles")
plt.ylabel("TVaR")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 1. Correlation Heatmaps ---
def plot_corr_heatmaps():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, file, title in zip(
        axes, [copula_file, driver_file, hybrid_file], ["Copula-Based", "Driver-Based", "Hybrid-Based"]
    ):
        corr_matrix = pd.read_excel(file, sheet_name="Spearman_Corr_Matrix", index_col=0)
        sns.heatmap(corr_matrix, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f")
        ax.set_title(title)
    plt.suptitle("Spearman Correlation Matrices")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


plot_corr_heatmaps()


# --- 4. Joint Exceedance Probabilities ---
def plot_joint_exceedance():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, file, title in zip(
        axes, [copula_file, driver_file, hybrid_file], ["Copula-Based", "Driver-Based", "Hybrid-Based"]
    ):
        df = pd.read_excel(file, sheet_name="Joint_Exceedance_Probabilities", index_col=0)
        sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".3f", ax=ax)
        ax.set_title(title)
    plt.suptitle("Joint Exceedance Probabilities")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


plot_joint_exceedance()


with pd.ExcelWriter("comparison_Output.xlsx") as writer:
    capital_comparison.to_excel(writer, sheet_name="Capital", index=False)
    var_combined.to_excel(writer, sheet_name="VaR and TVaR", index=False)
    combined_corr.to_excel(writer, sheet_name="Combined_Correlation", index=True)
    combined_jep.to_excel(writer, sheet_name="Combined_JEP", index=True)
