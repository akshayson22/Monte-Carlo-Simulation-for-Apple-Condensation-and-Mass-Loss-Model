Monte Carlo Simulation for Apple Condensation and Mass Loss Model
This repository contains the Python code and data files for the Monte Carlo simulation of an apple condensation and mass loss model, as described in the publication:

Sonawane, A. D., Hoffmann, T. G., Jedermann, R., Linke, M., & Mahajan, P. V. (2025). Investigating apple surface condensation and mass loss with IoT and predictive modelling. Postharvest Biology and Technology, 225, 113520. DOI: https://doi.org/10.1016/j.postharvbio.2025.113520

The full paper, which details the uncertainty analysis and validation of the IoT-based condensation model using Monte Carlo simulations, has also been submitted to the journal Acta Horticulturae (ISHS).

Project Overview
This study focuses on the uncertainty analysis of an IoT-based condensation model using Monte Carlo simulations. The model assesses the impact of various environmental and physical parameters on condensation and mass loss in apples. The primary goal is to understand how the inherent uncertainty of key input parameters affects the model's output, and to identify the most influential factors.

The model explores the following:

Parameter Variation: Seven key input parameters were varied within their absolute uncertainties:

Air temperature

Apple surface temperature

Relative humidity

Mass transfer coefficient for condensation

Air speed

Apple diameter

Apple surface area

Sensitivity Analysis: The study evaluates the impact of each parameter on two primary outputs: condensation amount and retention time.

Model Validation: The results are validated against experimental data to assess the model's predictive accuracy.

Key Findings
Based on the uncertainty analysis and sensitivity simulations, the study revealed several important insights:

The model demonstrated moderate accuracy in predicting model outputs.

Differences between predictions and experimental data are mainly attributed to condensation on bin surfaces and simplified model assumptions.

Model inputs with greater uncertainty tended to have a stronger influence on the output.

Relative humidity was found to be the most influential parameter, significantly affecting both condensation mass and retention time.

Apple surface area was equally important for condensation mass.

Air and surface temperatures had moderate but opposite effects, with their difference playing a key role in the condensation process.

Together, relative humidity and surface area accounted for 48% of the variability in condensation mass.

The overall uncertainty of the model outputs (condensation mass and retention time) was ±0.05% and ±50 seconds, respectively.

Plots and Analysis
The analysis generates a series of plots to visualize the data, parameter relationships, and model sensitivity for both condensation mass and retention time. The provided Python script MonteCarloSimulaionForAppleCondensation.py generates these plots.

Condensation Mass Analysis
These plots focus on the model's output for apple condensation mass.

1. Histograms of Input Parameters: Displays the frequency distribution of each input parameter.

2. Output Condensation Mass Distribution: Shows the frequency distribution of the simulated condensation mass.

3. Scatter Plot of Output Condensation Mass vs. Input Parameters: Illustrates the relationship between each input parameter and the final condensation mass output.

4. Pareto Front: This chart identifies the parameters that contribute most significantly to the total effect on the model output.

5. Parameter Correlation Matrix: A heatmap that visualizes the correlation between all the input parameters.

6. Partial Dependence Plot (PDP): Shows the marginal effect of one or two features on the predicted outcome.

7. Individual Conditional Expectation (ICE) Plot: Shows the effect of a single parameter on the model's output for each individual simulation run.

8. Uncertainty Analysis: Visualizes the model's output variability across the entire range of input parameter uncertainties.

9. Sensitivity Analysis: Quantifies the relative importance of each input parameter in driving the model's output uncertainty.

10. Global Sensitivity Analysis: Considers the effects of parameters and their interactions across the entire input space.

11. Validation Plot: Compares the model's predicted condensation mass against experimental data.

12. Permutation Importance: Measures the importance of a parameter by shuffling its values and observing the decrease in model performance.

Retention Time Analysis
These plots are generated for the retention time output, providing insight into the model's behavior for this specific metric.

13. Retention Time - Parameter Correlation Matrix: A heatmap that visualizes the correlation between the input parameters and the retention time output.

14. Pareto Chart for Retention Time: A Pareto chart that specifically analyzes which parameters have the largest effect on retention time.

Repository Contents
MonteCarloSimulaionForAppleCondensation.py: The Python script for performing the Monte Carlo simulations, sensitivity analysis, and generating plots.

Raw_data.xlsx: The raw experimental data used for model validation.

License
The code and data in this repository are intended to support the research publications mentioned above. For any use, please refer to the original articles for licensing and citation guidelines.