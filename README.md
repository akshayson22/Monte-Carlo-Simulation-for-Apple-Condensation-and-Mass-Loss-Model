# Monte Carlo Simulation for Apple Condensation and Mass Loss Model

This repository contains the Python code and data files for the Monte Carlo simulation of an apple condensation and mass loss model, as described in the publication:

**Sonawane, A. D., Hoffmann, T. G., Jedermann, R., Linke, M., & Mahajan, P. V. (2025).** Investigating apple surface condensation and mass loss with IoT and predictive modelling. *Postharvest Biology and Technology, 225*, 113520.
DOI: [https://doi.org/10.1016/j.postharvbio.2025.113520](https://doi.org/10.1016/j.postharvbio.2025.113520)

The full paper detailing the uncertainty analysis and validation has also been submitted to *Acta Horticulturae* (ISHS).

---

## Project Overview

This study focuses on the uncertainty analysis of an IoT-based condensation model using Monte Carlo simulations. The model assesses the impact of various environmental and physical parameters on condensation and mass loss in apples. The primary goal is to understand how the inherent uncertainty of key input parameters affects the model's output, and to identify the most influential factors.

### Key Input Parameters

* Air temperature
* Apple surface temperature
* Relative humidity
* Mass transfer coefficient for condensation
* Air speed
* Apple diameter
* Apple surface area

### Analysis Performed

* **Sensitivity Analysis:** Evaluates the impact of each parameter on condensation amount and retention time.
* **Model Validation:** Results are compared against experimental data to assess predictive accuracy.

---

## Key Findings

* The model demonstrated moderate accuracy in predicting outputs.
* Differences between predictions and experimental data are mainly due to condensation on bin surfaces and simplified assumptions.
* Inputs with greater uncertainty had a stronger influence on the output.
* Relative humidity was the most influential parameter affecting both condensation mass and retention time.
* Apple surface area was equally important for condensation mass.
* Air and surface temperatures had moderate but opposite effects, with their difference critical in condensation.
* Relative humidity and surface area together accounted for 48% of variability in condensation mass.
* Overall uncertainty of model outputs: ±0.05% (condensation mass) and ±50 seconds (retention time).

---

## Plots and Analysis

The analysis generates plots to visualize data, parameter relationships, and model sensitivity for both condensation mass and retention time.

### Condensation Mass Analysis

1. Histograms of Input Parameters
2. Output Condensation Mass Distribution
3. Scatter Plot of Output Condensation Mass vs. Input Parameters
4. Pareto Front
5. Parameter Correlation Matrix
6. Partial Dependence Plot (PDP)
7. Individual Conditional Expectation (ICE) Plot
8. Uncertainty Analysis
9. Sensitivity Analysis
10. Global Sensitivity Analysis
11. Validation Plot
12. Permutation Importance

### Retention Time Analysis

13. Retention Time - Parameter Correlation Matrix
14. Pareto Chart for Retention Time

---

## Repository Contents

* **MonteCarloSimulaionForAppleCondensation.py:** Python script for performing Monte Carlo simulations, sensitivity analysis, and generating plots.
* **Raw_data.xlsx:** Raw experimental data used for model validation.

---

## License

The code and data in this repository are intended to support the research publications mentioned above. For any use, please refer to the original articles for licensing and citation guidelines.
