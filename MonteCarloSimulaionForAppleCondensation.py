# Import required libraries
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import os
from scipy.stats import gaussian_kde
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.patches as mpatches
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import statsmodels.api as statsmodels_api
import warnings
import matplotlib.ticker as mticker
import statsmodels.api as sm
from matplotlib.ticker import FormatStrFormatter
warnings.filterwarnings('ignore')

# Set global style parameters for all plots
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
    'axes.linewidth': 1.5,
    'grid.color': '0.8',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

# Load the experimental data from Excel file
file_path = r'L:\........................................\Raw_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Pre-process data for retention time calculations
df_values = df.iloc[:, 0:7].apply(pd.to_numeric, errors='coerce').ffill().values
time_diffs = np.diff(df['A'].values, prepend=0)

# Physical constants used in both models
R_d = 287.05  # Gas constant for dry air (J/(kg·K))
R_v = 461.5  # Gas constant for water vapor (J/(kg·K))
mu0 = 1.716e-5  # Dynamic viscosity at reference temperature (kg/(m·s))
T0 = 273.15  # Reference temperature (K)
p_bar = 1013.25  # Reference pressure (hPa)
C = 120  # Sutherland's constant (K)
alpha0 = 2.24e-5  # Thermal diffusivity at reference temperature (m²/s)
fruit_mass = 252235   # Initial fruit mass (g) one cycle:252235;   Average2: 252243.9
n_apples = 1067  # Number of apples
surface_factor = 0.98  # Surface area adjustment factor

# Monte Carlo simulation parameters with their absolute uncertainties
num_simulations = 10000  # Number of Monte Carlo simulations
t_air_uncertainty = 0.06  # Uncertainty in air temperature measurement (°C)
t_surf_uncertainty = 0.06  # Uncertainty in surface temperature measurement (°C)
delta_t_uncertainty = 0.0  # Uncertainty in temperature difference measurement (°C)
humi_uncertainty = 1  # Uncertainty in relative humidity measurement (%)
air_speed_uncertainty = 3.0e-3  # Uncertainty in air speed measurement (m/s)
surface_area_apple_uncertainty = n_apples * 0.0074  # Uncertainty in surface area calculation (m²)
mtc_for_mass_uncertainty = 0#5.00e-11  # Uncertainty in mass transfer coefficient for transpiration (s/m)
rco2_uncertainty = 0#5.00e-7  # Uncertainty in CO2 respiration rate (kg/kg·s)
h_m_uncertainty = 3.64e-4  # Uncertainty in mass transfer coefficient for condensation (m/s)
d_uncertainty = 0.0167  # Uncertainty in diameter (m)

# Define consistent color scheme for all plots
colors = {
    'simulation': '#7f7f7f',  # Gray
    'mean': '#1f77b4',        # Blue
    'experimental': '#d62728',# Red
    'ci_68': '#1f77b4',
    'ci_95': '#2ca02c',
    'ci_99': '#ff7f0e',
    'density': '#1f77b4',
    'histogram': '#1f77b4',
    'min': '#d62728',
    'max': '#2ca02c',
    'median': '#1f77b4',
    'positive': '#006400',
    'negative': '#8B0000',
    'cumulative': '#1f77b4',
    'lightblue': '#ADD8E6',
    'dew_point': '#8A2BE2',  # Purple for dew point
    'not_applicable': '#A9A9A9'  # Gray for not applicable parameters
}


# =============================================
# PHYSICAL MODEL FOR MASS CHANGE
# =============================================

def run_physics_model(sa_adjust, h_m_adjust, rco2_adjust, mtc_adjust,
                     t_air_adjust, t_surf_adjust, delta_t_adjust, humi_adjust, air_speed_adjust, d_adjust):
    
    # Initialize lists to store results
    new_fruit_masses = [fruit_mass]
    cumulative_condensation = 0
    cumulative_condensation_list = [0]
    previous_time = None

    # Ensure numeric data types and forward fill missing values
    df.iloc[:, 0:7] = df.iloc[:, 0:7].apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 0:7] = df.iloc[:, 0:7].ffill()

    # Iterate through each row in the dataframe
    for index, row in df.iterrows():
        # Extract current measurement values
        t_surf1 = row['C']
        t_air1 = row['B']
        humi1 = row['D']
        current_time = row['A']
        air_speed1 = row['E']
        wetness_sensor_signal = row['G']
        actual_fruit_mass = row['F']

        # Calculate time difference since last measurement
        if previous_time is None:
            time_difference = 0
        else:
            time_difference = current_time - previous_time
        previous_time = current_time

        # Apply adjustments to input parameters
        t_surf = t_surf1 + t_surf_adjust if t_surf_uncertainty > 0 else t_surf1 # t_air1+ delta_t_adjust- delta_t   # t_surf1 + t_surf_adjust if t_surf_uncertainty > 0 else t_surf1
        t_air = t_air1 + t_air_adjust if t_air_uncertainty > 0 else t_air1
        humi = humi1 + humi_adjust if humi_uncertainty > 0 else humi1
        air_speed = max(0, air_speed1 + air_speed_adjust) if air_speed_uncertainty > 0 else air_speed1
        
        delta_t_calculated = t_air - t_surf
        
        # Thermodynamic calculations
        T_K = t_air + 273.15
        ptotal = 101325

        # Calculate saturation vapor pressures
        psl_cuv_air = 610.94 * math.exp(17.625 * t_air / (t_air + 243.04))
        psl_cuv_surf = 610.94 * math.exp(17.625 * t_surf / (t_surf + 243.04))
        pair = psl_cuv_air * humi / 100
        p_d = ptotal - pair

        # Calculate air properties
        rho_humid = (p_d / (R_d * T_K)) + (pair / (R_v * T_K))

        # Viscosity calculations
        mu_dry = mu0 * (T_K / T0) ** 1.5 * (T0 + C) / (T_K + C)
        mu_vapor = 9.8e-6
        Y = pair / ptotal
        mu_humid = mu_dry * (1 - Y) + mu_vapor * Y

        # Thermal conductivity of moist air
        k_moist_air = (((0.002646 + 0.0000737 * (t_air + 273.15)) * (1 - (0.622 * pair / (101325 - pair)))) + \
                     ((0.01468 + 0.0001536 * (t_air + 273.15)) * (0.622 * pair / (101325 - pair))))

        # Specific heat capacity of moist air
        cp_moist_air = 1005 + 1.82 * t_air + (0.61 * (0.622 * pair / (101325 - pair)) * 1860)

        # Kinematic viscosity and thermal diffusivity
        nu_humid = mu_humid / rho_humid
        alpha_humid = alpha0 * (T_K / T0) ** 1.5

        # Mass diffusivity and characteristic diameter
        D_m = 2.26e-5 * (T_K / 273.15) ** 1.81
        d = 0.0822 + (d_adjust if d_uncertainty > 0 else 0)

        # Dimensionless numbers
        Re = (rho_humid * air_speed * d) / mu_humid
        Sc = mu_humid / (rho_humid * D_m)
        Pr = (cp_moist_air * mu_humid) / k_moist_air

        # Sherwood number (for mass transfer)
        Sh = 2 + 1.3 * (Sc ** 0.15) + 0.66 * (Sc ** 0.31) * (Re ** 0.5)

        # Mass transfer coefficient with adjustment
        h_m = ((Sh * D_m) / d) + (h_m_adjust if h_m_uncertainty > 0 else 0)
    
        # Calculate current mass per apple (convert to kg)
        current_mass = new_fruit_masses[-1] / (n_apples*1000)
        if current_mass <= 0:
            current_mass = 1e-6
        # Calculate surface area with adjustment
        surface_area_apple = (n_apples * surface_factor * (0.0581 * current_mass ** 0.685)) + (sa_adjust if surface_area_apple_uncertainty > 0 else 0)
        
        v1 = surface_area_apple * h_m

        # Humidity ratio calculations
        xL_cuv_air = 1000 * 0.622 * pair / (100 * p_bar - pair)
        xL_cuv_surf = 1000 * 0.622 * psl_cuv_surf / (100 * p_bar - psl_cuv_surf)

        # Density calculations
        rhoL_cuv = ((1 + xL_cuv_air / 1000) * 100) / ((0.622 + xL_cuv_air / 1000) * 0.46152 * (273.16 + t_air))
        rhoL_cuv_surf = ((1 + xL_cuv_surf / 1000) * 100) / ((0.622 + xL_cuv_surf / 1000) * 0.46152 * (273.16 + t_surf))

        # Absolute humidity calculations
        xL_cuvv_air = xL_cuv_air * rhoL_cuv
        xL_cuvv_surf = xL_cuv_surf * rhoL_cuv_surf

        # Dew point temperature calculation
        Tdp = (243.04 * (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air))) / \
              (17.625 - (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air)))

        # Condensation calculation
        if (xL_cuvv_air > xL_cuvv_surf) or (t_surf < Tdp):
            condensed_capacity = (xL_cuvv_air - xL_cuvv_surf) * v1 * time_difference
            if condensed_capacity < 0:
                condensed_capacity = 0
        else:
            condensed_capacity = (xL_cuvv_air - xL_cuvv_surf) * v1 * time_difference
            if condensed_capacity > 0:
                condensed_capacity = 0

        # Update cumulative condensation
        cumulative_condensation += condensed_capacity
        if cumulative_condensation < 0:
            cumulative_condensation = 0

        # Calculate transpiration mass loss if no condensation is occurring
        if cumulative_condensation == 0:
            mass_loss_due_to_traspiration = surface_area_apple * 1000 * (pair - psl_cuv_surf) * time_difference / ((1/(1.30e-10 + (mtc_adjust if mtc_for_mass_uncertainty > 0 else 0))) + (1 / h_m))
        else:
            mass_loss_due_to_traspiration = 0

        # Calculate oxidative mass loss (respiration)
        rco2 = (2.56e-6 * math.exp((-34560 / 8.314) * ((1 / (t_air + 273.15)) - (1 / 278.15)))) + (rco2_adjust if rco2_uncertainty > 0 else 0)
        oxidative_mass_loss = float((new_fruit_masses[-1]/1000) * rco2 * time_difference * (180-108)/264)
        
        # Calculate new fruit mass accounting for all processes
        new_fruit_mass_with_condensation = (new_fruit_masses[-1]) - oxidative_mass_loss + mass_loss_due_to_traspiration + (cumulative_condensation - cumulative_condensation_list[-1])

        # Update tracking lists
        cumulative_condensation_list.append(cumulative_condensation)
        new_fruit_masses.append(new_fruit_mass_with_condensation)

    return new_fruit_masses[:-1]

# =============================================
# RETENTION TIME MODEL
# =============================================

def calculate_retention_time(t_air_adjust, t_surf_adjust, delta_t_adjust, humi_adjust):
    retention_time = 0
    
    t_surf1_values = df['C'].values
    t_air1_values = df['B'].values
    humi1_values = df['D'].values
    
        
    for j in range(len(df)):
        t_air = t_air1_values[j] + t_air_adjust
        t_surf = t_surf1_values[j] + t_surf_adjust #t_air1+ delta_t_adjust- delta_t       #t_surf1_values[j] + t_surf_adjust
        humi = max(0, humi1_values[j] + humi_adjust)
        
        # Saturation vapor pressure calculations
        psl_cuv_air = 610.94 * math.exp(17.625 * t_air / (t_air + 243.04))
        psl_cuv_surf = 610.94 * math.exp(17.625 * t_surf / (t_surf + 243.04))
        pair = psl_cuv_air * humi / 100
        
        # Water content calculations
        xL_cuv_air = 1000 * 0.622 * pair / (100 * p_bar - pair)
        xL_cuv_surf = 1000 * 0.622 * psl_cuv_surf / (100 * p_bar - psl_cuv_surf)
        
        # Air density calculations
        rhoL_cuv = ((1 + xL_cuv_air / 1000) * 100) / ((0.622 + xL_cuv_air / 1000) * 0.46152 * (273.16 + t_air))
        rhoL_cuv_surf = ((1 + xL_cuv_surf / 1000) * 100) / ((0.622 + xL_cuv_surf / 1000) * 0.46152 * (273.16 + t_surf))
        
        # Absolute humidity
        xL_cuvv_air = xL_cuv_air * rhoL_cuv
        xL_cuvv_surf = xL_cuv_surf * rhoL_cuv_surf
        
        # Dew point temperature
        Tdp_numerator = 243.04 * (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air))
        Tdp_denominator = 17.625 - (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air))
        Tdp = Tdp_numerator / Tdp_denominator
        
        # Check for condensation
        if (xL_cuvv_air > xL_cuvv_surf) or (t_surf < Tdp):
            retention_time += time_diffs[j]
    
    return retention_time

# =============================================
# MONTE CARLO SIMULATION SETUP
# =============================================

# Initialize arrays for Monte Carlo results
time_points = len(df)
all_simulations = np.zeros((time_points, num_simulations))
all_retention_times = np.zeros(num_simulations)

# Create list of active parameters for mass model (only those with uncertainty > 0)
active_params_mass = []
param_uncertainties_mass = []

if surface_area_apple_uncertainty > 0:
    active_params_mass.append('Surface Area')
    param_uncertainties_mass.append(surface_area_apple_uncertainty)
if mtc_for_mass_uncertainty > 0:
    active_params_mass.append('MTC Transpiration')
    param_uncertainties_mass.append(mtc_for_mass_uncertainty)
if rco2_uncertainty > 0:
    active_params_mass.append('Respiration Rate')
    param_uncertainties_mass.append(rco2_uncertainty)
if h_m_uncertainty > 0:
    active_params_mass.append('MTC Condensation')
    param_uncertainties_mass.append(h_m_uncertainty)
if t_air_uncertainty > 0:
    active_params_mass.append('Air Temperature')
    param_uncertainties_mass.append(t_air_uncertainty)
if t_surf_uncertainty > 0:
    active_params_mass.append('Surface Temperature')
    param_uncertainties_mass.append(t_surf_uncertainty)
if delta_t_uncertainty > 0:
    active_params_mass.append('Temperature Difference')
    param_uncertainties_mass.append(delta_t_uncertainty)
if humi_uncertainty > 0:
    active_params_mass.append('Relative Humidity')
    param_uncertainties_mass.append(humi_uncertainty)
if air_speed_uncertainty > 0:
    active_params_mass.append('Air Speed')
    param_uncertainties_mass.append(air_speed_uncertainty)
if d_uncertainty > 0:
    active_params_mass.append('Diameter')
    param_uncertainties_mass.append(d_uncertainty)

num_active_params_mass = len(active_params_mass)
param_variations_mass = np.zeros((num_simulations, num_active_params_mass))

# Parameters for retention time model (only those with uncertainty > 0)
active_params_retention = []
param_uncertainties_retention = []

if t_air_uncertainty > 0:
    active_params_retention.append('Air Temperature')
    param_uncertainties_retention.append(t_air_uncertainty)
if t_surf_uncertainty > 0:
    active_params_retention.append('Surface Temperature')
    param_uncertainties_retention.append(t_surf_uncertainty)
if delta_t_uncertainty > 0:
    active_params_retention.append('Temperature Difference')
    param_uncertainties_retention.append(delta_t_uncertainty)
if humi_uncertainty > 0:
    active_params_retention.append('Relative Humidity')
    param_uncertainties_retention.append(humi_uncertainty)

num_active_params_retention = len(active_params_retention)
param_variations_retention = np.zeros((num_simulations, num_active_params_retention))

# Create parameter mapping dictionaries
param_mapping_mass = {
    'Surface Area': 0,
    'MTC Transpiration': 1,
    'Respiration Rate': 2,
    'MTC Condensation': 3,
    'Air Temperature': 4,
    'Surface Temperature': 5,
    'Temperature Difference': 6,
    'Relative Humidity': 7,
    'Air Speed': 8,
    'Diameter': 9
}

param_mapping_retention = {
    'Air Temperature': 0,
    'Surface Temperature': 1,
    'Temperature Difference': 2,
    'Relative Humidity': 3
}

# =============================================
# MONTE CARLO SIMULATION EXECUTION
# =============================================

def run_single_simulation(i):
    # Initialize parameter adjustments
    sa_adjust = 0.0
    mtc_adjust = 0.0
    rco2_adjust = 0.0
    h_m_adjust = 0.0
    t_air_adjust = 0.0
    t_surf_adjust = 0.0
    delta_t_adjust = 0.0
    humi_adjust = 0.0
    air_speed_adjust = 0.0
    d_adjust = 0.0
    
    params_mass = []
    params_retention = []
    
    # Generate random parameter adjustments only for active parameters
    if surface_area_apple_uncertainty > 0:
        sa_adjust = random.uniform(-surface_area_apple_uncertainty, surface_area_apple_uncertainty)
        params_mass.append(sa_adjust)
    if mtc_for_mass_uncertainty > 0:
        mtc_adjust = random.uniform(-mtc_for_mass_uncertainty, mtc_for_mass_uncertainty)
        params_mass.append(mtc_adjust)
    if rco2_uncertainty > 0:
        rco2_adjust = random.uniform(-rco2_uncertainty, rco2_uncertainty)
        params_mass.append(rco2_adjust)
    if h_m_uncertainty > 0:
        h_m_adjust = random.uniform(-h_m_uncertainty, h_m_uncertainty)
        params_mass.append(h_m_adjust)
    if t_air_uncertainty > 0:
        t_air_adjust = random.uniform(-t_air_uncertainty, t_air_uncertainty)
        params_mass.append(t_air_adjust)
        params_retention.append(t_air_adjust)
    if t_surf_uncertainty > 0:
        t_surf_adjust = random.uniform(-t_surf_uncertainty, t_surf_uncertainty)
        params_mass.append(t_surf_adjust)
        params_retention.append(t_surf_adjust)
    if delta_t_uncertainty > 0:
        delta_t_adjust = random.uniform(-delta_t_uncertainty, delta_t_uncertainty)
        params_mass.append(delta_t_adjust)
        params_retention.append(delta_t_adjust)
    if humi_uncertainty > 0:
        humi_adjust = random.uniform(-humi_uncertainty, humi_uncertainty)
        params_mass.append(humi_adjust)
        params_retention.append(humi_adjust)
    if air_speed_uncertainty > 0:
        air_speed_adjust = random.uniform(-air_speed_uncertainty, air_speed_uncertainty)
        params_mass.append(air_speed_adjust)
    if d_uncertainty > 0:
        d_adjust = random.uniform(-d_uncertainty, d_uncertainty)
        params_mass.append(d_adjust)

    # Run mass model
    mass_result = run_physics_model(
        sa_adjust=sa_adjust,
        mtc_adjust=mtc_adjust,
        rco2_adjust=rco2_adjust,
        h_m_adjust=h_m_adjust,
        t_air_adjust=t_air_adjust,
        t_surf_adjust=t_surf_adjust,
        delta_t_adjust=delta_t_adjust,
        humi_adjust=humi_adjust,
        air_speed_adjust=air_speed_adjust,
        d_adjust=d_adjust
    )
    
    # Run retention time model only if there are active parameters
    if num_active_params_retention > 0:
        retention_result = calculate_retention_time(
            t_air_adjust=t_air_adjust,
            t_surf_adjust=t_surf_adjust,
            delta_t_adjust=delta_t_adjust,
            humi_adjust=humi_adjust
        )
    else:
        retention_result = calculate_retention_time(0.0, 0.0, 0.0, 0.0)
    
    return mass_result, retention_result, params_mass, params_retention

# Run simulations in parallel
results = Parallel(n_jobs=-1)(delayed(run_single_simulation)(i) for i in range(num_simulations))

# Unpack results
for i, (mass_result, retention_result, params_mass, params_retention) in enumerate(results):
    all_simulations[:, i] = np.array(mass_result) - 252235
    all_retention_times[i] = retention_result
    if params_mass:
        param_variations_mass[i] = params_mass
    if params_retention:
        param_variations_retention[i] = params_retention

# =============================================
# MASS MODEL ANALYSIS AND PLOTTING
# =============================================

# Calculate statistics for mass model
mean_results = np.mean(all_simulations, axis=1)
std_results = np.std(all_simulations, axis=1)
percentile_5 = np.percentile(all_simulations, 5, axis=1)
percentile_95 = np.percentile(all_simulations, 95, axis=1)
time_days = df['A'] / 3600
time_days_array = time_days.to_numpy()

# Plot 1: Monte Carlo Simulation Results for Mass
plt.figure(figsize=(10, 6))
plt.title('1. Monte Carlo Simulation Results for Condensation', fontweight='bold', fontsize=14)
for i in range(all_simulations.shape[1]):
    plt.plot(time_days, all_simulations[:, i], color=colors['simulation'], alpha=0.1, linewidth=0.5)
plt.plot(time_days, mean_results, color=colors['mean'], linewidth=2, label='Mean')
plt.plot(time_days, df['F']-252235, color=colors['experimental'], linewidth=2, label='Experimental')
plt.xlabel('Time (hours)', fontweight='bold', fontsize=14)
plt.ylabel('Condensation (g)', fontweight='bold', fontsize=14)
plt.legend(loc='best', prop={'weight':'bold', 'size':14})
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(top=200)
plt.tight_layout()

# ========================
# Plot 2: Monte Carlo Simulation Results for Mass
# ========================
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)

# Calculate dew point temperature for each time point
t_air_values = df['B'].values
humi_values = df['D'].values
dew_points = []
for t_air, humi in zip(t_air_values, humi_values):
    Tdp_numerator = 243.04 * (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air))
    Tdp_denominator = 17.625 - (math.log(humi / 100) + (17.625 * t_air) / (243.04 + t_air))
    dew_points.append(Tdp_numerator / Tdp_denominator)

# Create a list to store all parameter lines for legend
legend_handles = []

# Define the parameters you want to exclude
excluded_params = ['Surface Area', 'MTC Condensation', 'Diameter']

# Plot each active parameter with its uncertainty range
for i, param_name in enumerate(active_params_mass):
    if param_name in excluded_params:
        continue

    if param_name == 'Air Temperature':
        param_values = df['B'].values
        uncertainty = t_air_uncertainty
        unit = '°C'
        scale_factor = 1
        legend_label = param_name
    elif param_name == 'Surface Temperature':
        param_values = df['C'].values
        uncertainty = t_surf_uncertainty
        unit = '°C'
        scale_factor = 1
        legend_label = param_name
    elif param_name == 'Temperature Difference':
        param_values = df['B'].values - df['C'].values
        uncertainty = delta_t_uncertainty
        unit = '°C'
        scale_factor = 1
        legend_label = param_name
    elif param_name == 'Relative Humidity':
        param_values = df['D'].values
        uncertainty = humi_uncertainty
        unit = '%'
        scale_factor = 1
        legend_label = param_name
    elif param_name == 'Air Speed':
        param_values = df['E'].values
        uncertainty = air_speed_uncertainty
        unit = 'm/s'
        scale_factor = 0.01
        legend_label = param_name
    elif param_name == 'MTC Transpiration':
        param_values = np.zeros_like(df['A'].values)
        uncertainty = mtc_for_mass_uncertainty
        unit = 'm/s'
        scale_factor = 0.001
        legend_label = param_name
    elif param_name == 'Respiration Rate':
        param_values = np.zeros_like(df['A'].values)
        uncertainty = rco2_uncertainty
        unit = 'kg/kg·s'
        scale_factor = 1e6
        legend_label = param_name

    scaled_values = param_values / scale_factor
    scaled_uncertainty = uncertainty / scale_factor
    color = plt.cm.tab10(i % 10)

    if param_name != 'Relative Humidity':
        line, = plt.plot(time_days, scaled_values, color=color, linewidth=2, label=legend_label)
        legend_handles.append(line)

        plt.fill_between(time_days,
                         scaled_values - scaled_uncertainty,
                         scaled_values + scaled_uncertainty,
                         color=color, alpha=0.2)

# Dew point line
line, = plt.plot(time_days, dew_points, color=colors['dew_point'], linewidth=2, linestyle=':', label='Dew Point Temperature')
legend_handles.append(line)

# Labels and grid
plt.ylabel('Temperature (°C) and\nAir Speed × 10$^{-3}$ (m s$^{-1}$)', fontweight='bold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-4, 8)  # Changed to go up to 8 with 2 difference


# Twin y-axis for RH
ax2 = plt.gca().twinx()
rh_param_name = 'Relative Humidity'
if rh_param_name in active_params_mass and rh_param_name not in excluded_params:
    i = active_params_mass.index(rh_param_name)
    param_values = df['D'].values
    uncertainty = humi_uncertainty
    unit = '%'
    scale_factor = 1
    color = plt.cm.tab10(i % 10)

    line, = ax2.plot(time_days, param_values, color=color, linewidth=2, linestyle='-',
                     label=f'{rh_param_name}')
    legend_handles.append(line)

    ax2.fill_between(time_days,
                     param_values - uncertainty,
                     param_values + uncertainty,
                     color=color, alpha=0.1)

    ax2.set_ylabel('Relative Humidity (%)', fontweight='bold', fontsize=14)
    ax2.set_ylim(70, 100)  # Adjusted to give space for legend

# Move legend inside the plot
plt.legend(handles=legend_handles, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.0),  # Positioned above the plot but inside the box
           ncol=3,
           framealpha=0.3, 
           prop={'weight': 'bold', 'size': 14})

# ========================
# Subplot 2: Confidence Intervals for Mass
# ========================
plt.subplot(2, 1, 2)

conversion_factor = 252.235

# Confidence intervals
percentile_68_low = np.percentile(all_simulations, 16, axis=1) / conversion_factor
percentile_68_high = np.percentile(all_simulations, 84, axis=1) / conversion_factor
percentile_95_low = np.percentile(all_simulations, 2.5, axis=1) / conversion_factor
percentile_95_high = np.percentile(all_simulations, 97.5, axis=1) / conversion_factor
percentile_99_low = np.percentile(all_simulations, 0.5, axis=1) / conversion_factor
percentile_99_high = np.percentile(all_simulations, 99.5, axis=1) / conversion_factor

# Fill confidence intervals
plt.fill_between(time_days_array, percentile_99_low, percentile_99_high, 
                 color=colors['ci_99'], alpha=0.3, label='99% CI')
plt.fill_between(time_days_array, percentile_95_low, percentile_95_high, 
                 color=colors['ci_95'], alpha=0.4, label='95% CI')
plt.fill_between(time_days_array, percentile_68_low, percentile_68_high, 
                 color=colors['ci_68'], alpha=0.5, label='68% CI')

# Plot mean and experimental data
plt.plot(time_days_array, mean_results / conversion_factor, color='black', linewidth=2, label='Mean Predicted Condensation')
plt.plot(time_days_array, (df['F'] - 252235) / conversion_factor, color='red', linewidth=2, label='Experimental Condensation')

# Axis labels and grid
plt.xlabel('Time (hr)', fontweight='bold', fontsize=14)
plt.ylabel('Condensation Amount\n(g kg$^{-1}$)', fontweight='bold', fontsize=14)
plt.ylim(0, 0.8)  # Set to 0.8 as requested
plt.yticks(np.arange(0, 0.9, 0.2))  # Set ticks every 0.2 units
plt.grid(True, linestyle='--', alpha=0.7)

# Move legend inside the plot
plt.legend(loc='upper center', 
           bbox_to_anchor=(0.5, 1),  # Positioned above the plot but inside the box
           ncol=3,
           framealpha=0.3, 
           prop={'weight': 'bold', 'size': 14})

# Adjust layout with more space at the top
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted to make space for legends


# Plot 3: Final Mass Distribution
plt.figure(figsize=(10, 6))
final_masses = all_simulations[-1, :]
sns.histplot(final_masses, kde=True, color=colors['histogram'], bins=30)
plt.xlabel('Condensation (g)', fontweight='bold', fontsize=14)
plt.ylabel('Density', fontweight='bold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Plot 4: Extreme Effect Analysis for Mass (only if there are active parameters)
if num_active_params_mass > 0:
    plt.figure(figsize=(15, 10))
    
    
    # Dictionary to store all results for sensitivity calculation
    results_dict = {'Time': time_days}
    time_sensitivity_mass = {}
    average_sensitivity_mass = {}
    max_sensitivity_mass = {}
    time_of_max_sensitivity = {}

    for i in range(num_active_params_mass):
        plt.subplot(math.ceil(num_active_params_mass / 3), 3, i + 1)

        min_params = np.zeros(10)
        max_params = np.zeros(10)

        param_name = active_params_mass[i]
        param_idx = param_mapping_mass[param_name]
        param_uncertainty = param_uncertainties_mass[i]

        min_params[param_idx] = -param_uncertainty
        max_params[param_idx] = param_uncertainty

        min_result = np.array(run_physics_model(
            sa_adjust=min_params[param_mapping_mass['Surface Area']],
            mtc_adjust=min_params[param_mapping_mass['MTC Transpiration']],
            rco2_adjust=min_params[param_mapping_mass['Respiration Rate']],
            h_m_adjust=min_params[param_mapping_mass['MTC Condensation']],
            t_air_adjust=min_params[param_mapping_mass['Air Temperature']],
            t_surf_adjust=min_params[param_mapping_mass['Surface Temperature']],
            delta_t_adjust=min_params[param_mapping_mass['Temperature Difference']],
            humi_adjust=min_params[param_mapping_mass['Relative Humidity']],
            air_speed_adjust=min_params[param_mapping_mass['Air Speed']],
            d_adjust=min_params[param_mapping_mass['Diameter']]
        )) #- 252235  # Subtract baseline

        max_result = np.array(run_physics_model(
            sa_adjust=max_params[param_mapping_mass['Surface Area']],
            mtc_adjust=max_params[param_mapping_mass['MTC Transpiration']],
            rco2_adjust=max_params[param_mapping_mass['Respiration Rate']],
            h_m_adjust=max_params[param_mapping_mass['MTC Condensation']],
            t_air_adjust=max_params[param_mapping_mass['Air Temperature']],
            t_surf_adjust=max_params[param_mapping_mass['Surface Temperature']],
            delta_t_adjust=max_params[param_mapping_mass['Temperature Difference']],
            humi_adjust=max_params[param_mapping_mass['Relative Humidity']],
            air_speed_adjust=max_params[param_mapping_mass['Air Speed']],
            d_adjust=max_params[param_mapping_mass['Diameter']]
        ))  # - 252235# Subtract baseline

        median_result = np.array(run_physics_model(*[0.0] * 10)) #- 252235  # Subtract baseline

        plt.plot(time_days, min_result - 252235, linestyle='--', color=colors['min'], linewidth=1.5, label='Minimum Parameter Uncertainty')
        plt.plot(time_days, max_result - 252235, linestyle='--', color=colors['max'], linewidth=1.5, label='Maximum Parameter Uncertainty')
        plt.plot(time_days, median_result - 252235, color=colors['median'], linewidth=1.5, label='Median or Zero Parameter Uncertainty')

        plt.xlabel('Time (hr)', fontweight='bold', fontsize=14)
        plt.ylabel('Condensation (g)' if i % 3 == 0 else '', fontweight='bold', fontsize=14)
        plt.title(param_name, x=0.02, y=0.825, ha='left', fontsize=14, fontweight='bold')
        plt.ylim(top=125)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Store results for sensitivity analysis
        results_dict[f'{param_name}_Min'] = min_result 
        results_dict[f'{param_name}_Max'] = max_result 
        results_dict[f'{param_name}_Median'] = median_result

        min_val = np.min([min_result.min(), max_result.min(), median_result.min()])
        max_val = np.max([min_result.max(), max_result.max(), median_result.max()]) 

        normalized_median = (median_result - min_val) / (max_val - min_val + 1e-10)
        normalized_min = (min_result - min_val) / (max_val - min_val + 1e-10)
        normalized_max = (max_result - min_val) / (max_val - min_val + 1e-10)

        # Calculate time-dependent sensitivity
        median_nonzero = np.where(median_result == 0, 1e-10, median_result)  # prevent division by zero
        sensitivity = ((normalized_max - normalized_min) / (2)) * 100
        
        time_sensitivity_mass[param_name] = sensitivity
        average_sensitivity_mass[param_name] = np.mean(np.abs(sensitivity))
        max_sensitivity_mass[param_name] = np.max(np.abs(sensitivity))
        time_of_max_sensitivity[param_name] = time_days[np.argmax(np.abs(sensitivity))]

    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0), fontsize=14, borderaxespad=0., prop={'weight':'bold'})
    plt.tight_layout()

    # Plot 4a: Time-Varying Sensitivity Analysis
    plt.figure(figsize=(12, 6))
    

    for param_name in active_params_mass:
        plt.plot(results_dict['Time'], time_sensitivity_mass[param_name], label=param_name, linewidth=2)

    plt.xlabel('Time (hr)', fontweight='bold', fontsize=14)
    plt.ylabel('Sensitivity (% Chnage in Condensation)', fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'weight':'bold', 'size':14})
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Convert results_dict to DataFrame
    results_df = pd.DataFrame(results_dict)

    # Define path for Excel file
    output_excel_path = r"L:\.....................................................\mass_extreme_effects.xlsx"

    # Save DataFrame to Excel
    results_df.to_excel(output_excel_path, index=False, engine='openpyxl')

    print(f"Mass effect analysis results saved to Excel: {output_excel_path}")

    # Plot 4b: Average Sensitivity
    plt.figure(figsize=(12, 6))
    

    # Create DataFrame for sensitivity results
    sensitivity_results = []
    for param_name in active_params_mass:
        effect_direction = 'Positive' if np.mean(time_sensitivity_mass[param_name]) > 0 else 'Negative'
        sensitivity_results.append({
            'Parameter': param_name,
            'Avg_Sensitivity': average_sensitivity_mass[param_name],
            'Max_Sensitivity': max_sensitivity_mass[param_name],
            'Time_of_Max': time_of_max_sensitivity[param_name],
            'Effect_Direction': effect_direction
        })

    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df = sensitivity_df.sort_values('Avg_Sensitivity', ascending=False)

    # Create bar plot with color coding
    colors_bar = [colors['positive'] if x == 'Positive' else colors['negative'] 
                 for x in sensitivity_df['Effect_Direction']]
    bars = plt.bar(sensitivity_df['Parameter'], sensitivity_df['Avg_Sensitivity'], color=colors_bar)

    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Average Absolute Sensitivity (%)', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Create legend
    pos_patch = mpatches.Patch(color=colors['positive'], label='Positive Effect')
    neg_patch = mpatches.Patch(color=colors['negative'], label='Negative Effect')
    plt.legend(handles=[pos_patch, neg_patch], prop={'weight':'bold', 'size':14})
    
    plt.tight_layout()

    # Plot 4c: Maximum Sensitivity and When It Occurs
    plt.figure(figsize=(12, 6))
    

    # Sort by maximum sensitivity
    sensitivity_df = sensitivity_df.sort_values('Max_Sensitivity', ascending=False)
    
    # Create colormap for time values
    norm = plt.Normalize(min(time_days), max(time_days))
    cmap = plt.cm.viridis
    
    # Create bars colored by time of max sensitivity
    bars = plt.bar(sensitivity_df['Parameter'], sensitivity_df['Max_Sensitivity'],
                  color=cmap(norm(sensitivity_df['Time_of_Max'])))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time of Maximum Sensitivity (hr)', fontweight='bold', fontsize=14)
    
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Maximum Sensitivity (% change per unit)', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()

    # Plot 4d: Normalized Sensitivity (as %) and Cumulative Sum
    plt.figure(figsize=(12, 6))

    # Calculate absolute sensitivities and normalize to percentage
    abs_sensitivities = sensitivity_df['Avg_Sensitivity'].abs()
    normalized_sensitivities = (abs_sensitivities / abs_sensitivities.sum()) * 100  # Convert to percentage

    # Sort by normalized sensitivity
    sensitivity_df = sensitivity_df.iloc[np.argsort(-normalized_sensitivities)]
    normalized_sensitivities = (sensitivity_df['Avg_Sensitivity'].abs() / sensitivity_df['Avg_Sensitivity'].abs().sum()) * 100

    # Create bar plot with color coding
    colors_bar = [colors['positive'] if x == 'Positive' else colors['negative'] 
                for x in sensitivity_df['Effect_Direction']]
    bars = plt.bar(sensitivity_df['Parameter'], normalized_sensitivities,
                color=colors_bar, alpha=0.7)

    # Add cumulative sum line (as percentage)
    cumulative = np.cumsum(normalized_sensitivities)
    plt.plot(sensitivity_df['Parameter'], cumulative,
            color=colors['cumulative'], marker='o',
            linewidth=2, markersize=6,
            label='Cumulative Sum')

    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Sensitivity (%)', fontweight='bold', fontsize=14)
    plt.ylim(0, 110)  # Adjusted for percentage scale
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add value labels to bars (as percentage)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{height:.0f}%',
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Add cumulative value labels (as percentage)
    for i, val in enumerate(cumulative):
        plt.text(i, val + 3, f'{val:.0f}%',
                ha='center', va='bottom',
                color=colors['cumulative'], fontsize=14, fontweight='bold')

    # Create legend and position it on right side inside the plot
    pos_patch = mpatches.Patch(color=colors['positive'], alpha=0.7, label='Positive Effect')
    neg_patch = mpatches.Patch(color=colors['negative'], alpha=0.7, label='Negative Effect')
    cum_line = plt.Line2D([], [], color=colors['cumulative'], marker='o', label='Cumulative Sum')
    plt.legend(handles=[pos_patch, neg_patch, cum_line], 
           loc='center right',             # Inside the plot, center right
           prop={'weight': 'bold', 'size': 14},
           framealpha=1)

    plt.tight_layout()

    # Print sensitivity results
    print("\n=== MASS MODEL SENSITIVITY RESULTS ===")
    print(sensitivity_df[['Parameter', 'Avg_Sensitivity', 'Max_Sensitivity', 'Time_of_Max', 'Effect_Direction']].to_string(index=False))

    # Plot 5: Time-Varying Sensitivity Analysis
    plt.figure(figsize=(12, 6))
    

    time_sensitivity_mass = {}
    average_sensitivity_mass = {}

    for param_name in active_params_mass:
        min_vals = results_dict[f'{param_name}_Min']
        max_vals = results_dict[f'{param_name}_Max']
        median_vals = results_dict[f'{param_name}_Median']

        # Calculate sensitivity (% change per unit parameter)
        median_nonzero = np.where(median_vals == 0, 1e-10, median_vals)  # prevent division by zero
        sensitivity = ((max_vals - min_vals) / (2 * param_uncertainties_mass[active_params_mass.index(param_name)])) / median_nonzero * 100
        
        time_sensitivity_mass[param_name] = sensitivity
        average_sensitivity_mass[param_name] = np.mean(np.abs(sensitivity))

        plt.plot(results_dict['Time'], sensitivity, label=param_name, linewidth=2)

    plt.xlabel('Time (Hours)', fontweight='bold', fontsize=14)
    plt.ylabel('Sensitivity (%)', fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'weight':'bold', 'size':14})
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Plot 6: Normalized Sensitivity Analysis
    plt.figure(figsize=(12, 6))
    

    sensitivity_results_mass = {
        'Parameter': [],
        'Avg_Sensitivity': [],
        'Effect_Direction': []
    }

    for param_name in active_params_mass:
        avg_sensitivity = average_sensitivity_mass[param_name]
        effect_direction = 'Positive' if np.mean(time_sensitivity_mass[param_name]) > 0 else 'Negative'
        
        sensitivity_results_mass['Parameter'].append(param_name)
        sensitivity_results_mass['Avg_Sensitivity'].append(avg_sensitivity)
        sensitivity_results_mass['Effect_Direction'].append(effect_direction)

    sensitivity_mass_df = pd.DataFrame(sensitivity_results_mass)
    sensitivity_mass_df = sensitivity_mass_df.sort_values('Avg_Sensitivity', ascending=False)

    # Create bar plot with color coding
    colors_bar = [colors['positive'] if x == 'Positive' else colors['negative'] for x in sensitivity_mass_df['Effect_Direction']]
    bars = plt.bar(sensitivity_mass_df['Parameter'], sensitivity_mass_df['Avg_Sensitivity']*100, color=colors_bar)

    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Normalized Sensitivity', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Create legend
    pos_patch = mpatches.Patch(color=colors['positive'], label='Positive Effect')
    neg_patch = mpatches.Patch(color=colors['negative'], label='Negative Effect')
    plt.legend(handles=[pos_patch, neg_patch], loc='center right', prop={'weight':'bold', 'size':14})
    
    plt.tight_layout()

    # Plot 7: Parameter Correlation Heatmap for Mass
    if num_active_params_mass > 1:
        plt.figure(figsize=(10, 8))
        plt.title('7. Mass Model - Parameter Correlation Matrix', fontweight='bold', fontsize=14)
        param_mass_df = pd.DataFrame(param_variations_mass, columns=active_params_mass)
        corr_matrix_mass = param_mass_df.corr()
        sns.heatmap(corr_matrix_mass, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', annot_kws={'size': 14, 'weight':'bold'})
        plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
        plt.yticks(fontweight='bold', fontsize=14)
        plt.tight_layout()

# Plot 5: Regression-based Sensitivity Analysis with Interactions
if num_active_params_mass > 0:
    print("\n=== REGRESSION-BASED SENSITIVITY ANALYSIS ===")
    
    # Prepare data for regression analysis
    X = param_variations_mass
    y = all_simulations[-1, :]  # Final mass changes
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create interaction terms for training data
    interaction_terms_train = np.column_stack(
        [X_train[:, i] * X_train[:, j] 
         for i in range(X_train.shape[1]) 
         for j in range(i+1, X_train.shape[1])]
    )
    
    import statsmodels.api as sm_api
    # Then in your regression code, use sm_api instead:
    X_train_with_interactions = sm_api.add_constant(np.column_stack([X_train, interaction_terms_train]))
    model = sm_api.OLS(y_train, X_train_with_interactions).fit()
    
    plt.figure(figsize=(12, 6))
    plt.title('5f. Mean Simulation vs Regression Prediction', fontweight='bold', fontsize=14)
    
    # Calculate mean simulation results
    mean_simulation = np.mean(all_simulations, axis=1)
    
    # Prepare regression prediction data
    interaction_terms = np.column_stack(
        [param_variations_mass[:, i] * param_variations_mass[:, j] 
         for i in range(num_active_params_mass) 
         for j in range(i+1, num_active_params_mass)]
    )
    X_all_scaled = scaler.transform(param_variations_mass)
    X_all_with_interactions = sm_api.add_constant(np.column_stack([X_all_scaled, interaction_terms]))
    final_preds = model.predict(X_all_with_interactions)
    
    regression_preds = np.zeros_like(all_simulations)
    for i in range(all_simulations.shape[1]):
        if all_simulations[-1, i] != 0:
            scaling_factor = all_simulations[:, i] / all_simulations[-1, i]
            regression_preds[:, i] = final_preds[i] * scaling_factor
    
    mean_regression = np.mean(regression_preds, axis=1)
    
    # Plotting
    plt.plot(time_days, mean_simulation, color=colors['mean'], linewidth=2, label='Mean Simulation')
    plt.plot(time_days, mean_regression, color='purple', linestyle='--', linewidth=2, label='Regression Prediction')
    plt.plot(time_days, df['F']-252235, color=colors['experimental'], linewidth=1.5, label='Experimental Data')
    
    # Add statistics
    rmse = np.sqrt(mean_squared_error(mean_simulation, mean_regression))
    r2 = r2_score(mean_simulation, mean_regression)
    plt.text(0.95, 0.05, f'RMSE: {rmse:.2f} g\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
             ha='right', va='bottom', fontweight='bold', fontsize=14)
    
    plt.xlabel('Time (hours)', fontweight='bold', fontsize=14)
    plt.ylabel('Mass Change (g)', fontweight='bold', fontsize=14)
    plt.legend(prop={'weight':'bold', 'size':14})
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # PLOT 5g: Parameter Sensitivity Ranking (Fixed Version)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    plt.title('5g. Parameter Sensitivity Ranking', fontweight='bold', pad=20, fontsize=14)

    # Calculate sensitivity metrics for each parameter
    sensitivity_scores = {}
    for i, param in enumerate(active_params_mass):
        try:
            # Get parameter values and regression coefficient
            param_values = param_variations_mass[:, i]
            coeff = model.params[1+i]  # Skip intercept
            
            # Calculate time-varying correlation
            correlations = np.array([np.corrcoef(param_values, all_simulations[t,:])[0,1] 
                            for t in range(len(time_days))])
            
            # Calculate scaled effect size
            scaled_effects = np.array([
                np.corrcoef(all_simulations[t,:], param_values * coeff)[0,1]
                for t in range(len(time_days))
            ])
            
            # Combine metrics (absolute average of both components)
            combined_score = np.nanmean(np.abs(correlations * scaled_effects))
            sensitivity_scores[param] = combined_score * 100  # Convert to percentage
            
        except Exception as e:
            print(f"Error calculating sensitivity for {param}: {str(e)}")
            sensitivity_scores[param] = 0

    # Sort parameters by sensitivity score
    sorted_params = sorted(active_params_mass,
                        key=lambda x: sensitivity_scores[x],
                        reverse=True)
    sorted_scores = [sensitivity_scores[p] for p in sorted_params]
    signs = [np.sign(model.params[1+active_params_mass.index(p)]) for p in sorted_params]  # Get sign from regression

    # Create colormap (red for negative, blue for positive)
    cmap = plt.cm.RdBu
    norm = plt.Normalize(-1, 1)
    bar_colors = [cmap(norm(s)) for s in signs]

    # Create the bar plot - ensure bars are visible
    bars = plt.bar(sorted_params, sorted_scores, 
                color=bar_colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=1)

    # Add value labels
    for bar, score, sign in zip(bars, sorted_scores, signs):
        height = bar.get_height()
        label_color = 'black' if height > 1 else 'gray'  # Make small values visible
        sign_symbol = '+' if sign > 0 else '-' if sign < 0 else '±'
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.5,  # Small offset above bar
                f'{sign_symbol}{height:.1f}%',
                ha='center', 
                va='bottom',
                color=label_color,
                fontsize=14,
                fontweight='bold')

    # Formatting
    plt.axhline(0, color='black', linewidth=0.8)  # Baseline
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Combined Sensitivity Score (%)', fontweight='bold', fontsize=14)
    plt.grid(True, axis='y', linestyle=':', alpha=0.3)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label('Effect Direction', fontweight='bold', fontsize=14)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Negative', 'Neutral', 'Positive'], fontweight='bold', fontsize=14)

    # Ensure y-axis starts at 0
    plt.ylim(0, max(sorted_scores)*1.2 if max(sorted_scores) > 0 else 10)

    # Add explanatory note
    plt.annotate('Scores combine:\n• Time-varying correlation\n• Scaled regression effects',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # PLOT 5h: Regression-Based Time-Varying Sensitivity Analysis
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.title('5h. Regression-Based Time-Varying Parameter Sensitivity', 
            fontweight='bold', pad=20, fontsize=14)

    # Get regression coefficients from model
    regression_coeffs = model.params[1:num_active_params_mass+1]
    time_regression = {}

    # Calculate time-scaled regression sensitivity
    for i, param in enumerate(active_params_mass):
        param_values = param_variations_mass[:, i]
        
        # Time-scaled regression sensitivity
        param_effect_scale = np.array([
            np.corrcoef(all_simulations[t,:], param_values * regression_coeffs[i])[0,1]
            for t in range(len(time_days))
        ])
        time_regression[param] = regression_coeffs[i] * param_effect_scale

    # Normalize and prepare regression sensitivity
    reg_sensitivity = {}
    valid_params = []

    for param in active_params_mass:
        try:
            # Robust normalization using IQR
            reg_vals = time_regression[param]
            q5, q50, q95 = np.nanpercentile(reg_vals, [5, 50, 95])
            iqr = max(q95 - q5, 1e-10)  # Avoid division by zero
            
            # Normalize and scale to percentage
            norm_reg = (reg_vals - q50) / iqr * 100
            norm_reg = np.nan_to_num(norm_reg, nan=0, posinf=0, neginf=0)
            
            if np.any(np.isfinite(norm_reg)):
                reg_sensitivity[param] = norm_reg
                valid_params.append(param)
        except:
            continue

    # Check for valid parameters
    if not valid_params:
        raise ValueError("No valid sensitivity values could be calculated")

    # Select top parameters
    top_params = sorted(valid_params,
                    key=lambda x: np.nanmax(np.abs(reg_sensitivity[x])),
                    reverse=True)[:min(7, len(valid_params))]

    # Create colormap and determine y-axis limits
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_params)))
    all_vals = np.concatenate([reg_sensitivity[p] for p in top_params])
    ymax = np.nanmax(np.abs(all_vals)) * 1.2 if len(all_vals) > 0 else 1.0
    ymax = max(ymax, 5.0)  # Ensure minimum visible range

    # Plot each parameter's sensitivity
    for idx, param in enumerate(top_params):
        sensitivity = reg_sensitivity[param]
        
        # Smooth the sensitivity curve
        window_size = max(3, len(time_days)//50)
        weights = np.hanning(window_size)
        smoothed = np.convolve(sensitivity, weights/weights.sum(), mode='same')
        
        plt.plot(time_days, smoothed, 
                color=colors[idx],
                linewidth=2.5,
                alpha=0.9,
                label=param)
        
        # Mark peak sensitivity point
        peak_idx = np.nanargmax(np.abs(smoothed))
        plt.scatter(time_days[peak_idx], smoothed[peak_idx],
                color=colors[idx], 
                s=100, 
                edgecolor='black',
                zorder=5)
        
        # Annotate peak value
        plt.text(time_days[peak_idx], smoothed[peak_idx] + 0.07*ymax,
                f'{smoothed[peak_idx]:.1f}%',
                ha='center', va='bottom', 
                fontsize=14,
                fontweight='bold')

    # Configure plot appearance
    plt.ylim(-ymax, ymax)
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Time (days)', fontweight='bold', fontsize=14)
    plt.ylabel('Regression Sensitivity Index (%)', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.3)

    # Add legend and information box
    if top_params:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), 
                frameon=True, shadow=False, edgecolor='#333333',
                prop={'weight':'bold', 'size':14})

    plt.annotate('Sensitivity based exclusively on:\nTime-scaled regression coefficients',
                xy=(0.98, 0.03), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()


    # -------------------------------------------------------------------------
    # PLOT 5i: Interaction Effects Heatmap
    # -------------------------------------------------------------------------
    if num_active_params_mass > 1:
        plt.figure(figsize=(10, 8))
        plt.title('5i. Significant Interaction Effects', fontweight='bold', fontsize=14)
        
        # Extract interaction coefficients
        interaction_coeffs = []
        interaction_pairs = []
        current_idx = num_active_params_mass + 1  # Skip intercept and main effects
        
        for i in range(num_active_params_mass):
            for j in range(i+1, num_active_params_mass):
                if model.pvalues[current_idx] < 0.1:  # Only significant interactions
                    interaction_coeffs.append(model.params[current_idx])
                    interaction_pairs.append(f"{active_params_mass[i]}×{active_params_mass[j]}")
                current_idx += 1
        
        if interaction_coeffs:
            # Create matrix for heatmap
            n = len(interaction_pairs)
            heatmap_data = np.zeros((n, 1))
            heatmap_data[:, 0] = interaction_coeffs
            
            # Plot
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                        cbar=False, yticklabels=interaction_pairs, xticklabels=["Effect Size"],
                        annot_kws={'weight':'bold', 'size':14})
            plt.yticks(rotation=0, fontweight='bold', fontsize=14)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "No significant interactions (p < 0.1)", 
                    ha='center', va='center', fontweight='bold', fontsize=14)
            plt.axis('off')

    # -------------------------------------------------------------------------
    # PLOT 5j: Residual Analysis
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.title('5j. Regression Residual Analysis', fontweight='bold', fontsize=14)
    
    # Calculate residuals
    y_pred = model.predict(X_all_with_interactions)
    residuals = y - y_pred
    
    # Q-Q plot
    plt.subplot(1, 2, 1)
    sm_api.qqplot(residuals, line='s', ax=plt.gca())
    plt.title('Q-Q Plot of Residuals', fontweight='bold', fontsize=14)
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values', fontweight='bold', fontsize=14)
    plt.ylabel('Residuals', fontweight='bold', fontsize=14)
    plt.title('Residuals vs Predicted', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # PLOT 5k: Partial Dependence Plots
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.suptitle('5k. Partial Dependence Plots', fontweight='bold', y=1.02, fontsize=14)
    
    # Fit Random Forest for more flexible PDPs
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Plot top 4 parameters
    top_4_idx = np.argsort(-np.abs(model.params[1:num_active_params_mass+1]))[:4]
    display = PartialDependenceDisplay.from_estimator(
        rf, X_train, 
        features=top_4_idx,
        feature_names=active_params_mass,
        n_cols=2, 
        n_jobs=-1,
        grid_resolution=20,
        line_kw={'linewidth': 2}
    )
    
    # Adjust subplot styling
    for ax in display.axes_.ravel():
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold', fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontweight='bold', fontsize=14)
    
    plt.tight_layout()

    # Print model summary
    print(model.summary())


# =============================================
# RETENTION TIME ANALYSIS AND PLOTTING
# =============================================

# Define color constants at the top (use UPPERCASE to avoid overwrites)
COLOR_PALETTE = {
    'simulation': '#7f7f7f',  # Gray
    'mean': '#1f77b4',        # Blue
    'experimental': '#d62728',# Red
    'ci_68': '#1f77b4',
    'ci_95': '#2ca02c',       # Green
    'ci_99': '#ff7f0e',       # Orange
    'density': '#1f77b4',
    'histogram': 'red',
    'min': '#d62728',
    'max': '#2ca02c',
    'median': '#1f77b4',
    'positive': '#006400',    # Dark Green
    'negative': '#8B0000',    # Dark Red
    'cumulative': '#1f77b4',
    'kde': 'blue',
    'not_applicable': '#A9A9A9'  # Gray for not applicable parameters
}

# Calculate statistics for retention time
all_retention_times_min = np.array(all_retention_times) / 60  # Convert to minutes
mean_retention = np.mean(all_retention_times_min)
std_retention = np.std(all_retention_times_min)
ci_68 = (mean_retention - std_retention, mean_retention + std_retention)
ci_95 = (mean_retention - 2*std_retention, mean_retention + 2*std_retention)

# Plot 8: Retention Time Distribution (in Hours)
plt.figure(figsize=(10, 6))

# Convert values from minutes to hours
all_retention_times_hr = np.array(all_retention_times_min) / 60.0
mean_retention_hr = mean_retention / 60.0
ci_68_hr = (ci_68[0] / 60.0, ci_68[1] / 60.0)
ci_95_hr = (ci_95[0] / 60.0, ci_95[1] / 60.0)
experimental_mean_hr = 87.00 / 60.0  # Convert 87 minutes to hours

# Create histogram with KDE using count
bin_count = 30
counts, bins, _ = plt.hist(
    all_retention_times_hr,
    bins=bin_count,
    color=COLOR_PALETTE['histogram'],
    edgecolor='black',
    linewidth=0.5,
    alpha=0.6
)

bin_width = bins[1] - bins[0]
total_count = len(all_retention_times_hr)

# Add KDE plot scaled to match histogram count
kde = gaussian_kde(all_retention_times_hr)
x_vals = np.linspace(min(all_retention_times_hr), max(all_retention_times_hr), 500)
kde_vals = kde(x_vals) * total_count * bin_width
plt.plot(x_vals, kde_vals, color=COLOR_PALETTE['kde'], linewidth=2)

# Add confidence intervals
plt.axvspan(ci_95_hr[0], ci_68_hr[0], color=COLOR_PALETTE['ci_95'],
            alpha=0.3, label='95% CI')
plt.axvspan(ci_68_hr[0], ci_68_hr[1], color=COLOR_PALETTE['ci_68'],
            alpha=0.3, label='68% CI')
plt.axvspan(ci_68_hr[1], ci_95_hr[1], color=COLOR_PALETTE['ci_95'],
            alpha=0.3)

# Add mean and experimental lines
plt.axvline(mean_retention_hr, color=COLOR_PALETTE['mean'],
            linestyle='--', linewidth=2,
            label=f'Predicted Mean = {mean_retention_hr:.2f} hr')
plt.axvline(experimental_mean_hr, color='black', linestyle='--',
            linewidth=2, label=f'Experimental Mean = {experimental_mean_hr:.2f} hr')

# Axis labels and styling
plt.xlabel('Retention Time (hr)', fontweight='bold', fontsize=14)
plt.ylabel('Count', fontweight='bold', fontsize=14)
plt.legend(prop={'weight': 'bold', 'size': 14}, bbox_to_anchor=(0.5, 1), 
           loc='upper center', ncol=2)
plt.ylim(0, 1600)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.tight_layout()

# Plot 9: Retention Time vs Parameter Variations
plt.figure(figsize=(15, 5))
plt.suptitle('9. Retention Time vs Parameter Variations', 
             fontweight='bold', y=1.05, fontsize=14)

for i in range(num_active_params_retention):
    plt.subplot(1, num_active_params_retention, i + 1)
    
    plt.scatter(param_variations_retention[:, i], all_retention_times_min, 
               alpha=0.3, color=COLOR_PALETTE['simulation'])
    plt.xlabel(f'{active_params_retention[i]} Variation', fontweight='bold', fontsize=14)
    plt.ylabel('Retention Time (min)', fontweight='bold', fontsize=14)
    plt.title(active_params_retention[i], fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1400)

plt.tight_layout()

# Plot 10: Extreme Effect Analysis for Retention Time
if num_active_params_retention > 0:
    plt.figure(figsize=(15, 5))
    plt.suptitle('10. Retention Time - Extreme Parameter Effects', 
                 fontweight='bold', y=1.05, fontsize=14)
    
    sensitivity_results_retention = {
        'Parameter': [], 'Min_Retention': [], 'Max_Retention': [],
        'Median_Retention': [], 'Sensitivity': [], 'Normalized_Sensitivity': []
    }
    
    for i in range(num_active_params_retention):
        plt.subplot(1, num_active_params_retention, i + 1)
        
        param_name = active_params_retention[i]
        uncertainty = param_uncertainties_retention[i]
        
        # Calculate min, max, median results
        min_result = calculate_retention_time(
            t_air_adjust=-uncertainty if param_name == 'Air Temperature' else 0,
            t_surf_adjust=-uncertainty if param_name == 'Surface Temperature' else 0,
            delta_t_adjust=-uncertainty if param_name == 'Temperature Difference' else 0,
            humi_adjust=-uncertainty if param_name == 'Relative Humidity' else 0
        ) / 60
        
        max_result = calculate_retention_time(
            t_air_adjust=uncertainty if param_name == 'Air Temperature' else 0,
            t_surf_adjust=uncertainty if param_name == 'Surface Temperature' else 0,
            delta_t_adjust=uncertainty if param_name == 'Temperature Difference' else 0,
            humi_adjust=uncertainty if param_name == 'Relative Humidity' else 0
        ) / 60
        
        median_result = calculate_retention_time(0, 0, 0, 0) / 60
        
        # Store results
        sensitivity_results_retention['Parameter'].append(param_name)
        sensitivity_results_retention['Min_Retention'].append(min_result)
        sensitivity_results_retention['Max_Retention'].append(max_result)
        sensitivity_results_retention['Median_Retention'].append(median_result)
        sensitivity_results_retention['Sensitivity'].append(max_result - min_result)
        sensitivity_results_retention['Normalized_Sensitivity'].append((max_result - min_result)/median_result)
        
        # Create bar plot with safe color access
        plt.bar(['Min', 'Median', 'Max'], [min_result, median_result, max_result],
               color=[COLOR_PALETTE['min'], COLOR_PALETTE['median'], COLOR_PALETTE['max']])
        
        plt.ylabel('Retention Time (min)', fontweight='bold', fontsize=14)
        plt.title(param_name, fontweight='bold', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Plot 11: Retention Time Sensitivity Analysis
    sensitivity_retention_df = pd.DataFrame(sensitivity_results_retention)
    sensitivity_retention_df = sensitivity_retention_df.sort_values('Sensitivity', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.title('11. Retention Time - Parameter Sensitivity', fontweight='bold', fontsize=14)
    
    colors_bar = [COLOR_PALETTE['positive'] if x > 0 else COLOR_PALETTE['negative'] 
                 for x in sensitivity_retention_df['Sensitivity']]
    
    bars = plt.bar(sensitivity_retention_df['Parameter'], 
                  sensitivity_retention_df['Sensitivity'],
                  color=colors_bar)
    
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Sensitivity (min)', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} min',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add legend
    pos_patch = mpatches.Patch(color=COLOR_PALETTE['positive'], 
                              label='Increases Retention')
    neg_patch = mpatches.Patch(color=COLOR_PALETTE['negative'], 
                              label='Decreases Retention')
    plt.legend(handles=[pos_patch, neg_patch], prop={'weight':'bold', 'size':14})
    
    plt.tight_layout()
    
   # Plot 12: Normalized Sensitivity and Cumulative Sum (All Params Shown in Order)
    plt.figure(figsize=(12, 6))
    

    # Desired bar order
    ordered_params = ['Relative Humidity', 'Air Temperature', 'Surface Temperature',
                    'MTC Condensation', 'Diameter', 'Surface Area', 'Air Speed']

    # Initialize DataFrame
    sensitivity_all_params = pd.DataFrame(index=ordered_params)
    sensitivity_all_params['Sensitivity'] = 0.0
    sensitivity_all_params['Normalized_Sensitivity'] = 0.0
    sensitivity_all_params['Effect_Direction'] = 'Not Applicable'

    # Top 3 parameters
    top_params = ['Air Temperature', 'Surface Temperature', 'Relative Humidity']

    # Fill in values for active parameters
    for param in top_params:
        sens_val = sensitivity_retention_df.loc[sensitivity_retention_df['Parameter'] == param, 'Sensitivity'].values[0]
        norm_val = sensitivity_retention_df.loc[sensitivity_retention_df['Parameter'] == param, 'Normalized_Sensitivity'].values[0]
        
        sensitivity_all_params.loc[param, 'Sensitivity'] = sens_val
        sensitivity_all_params.loc[param, 'Normalized_Sensitivity'] = norm_val
        sensitivity_all_params.loc[param, 'Effect_Direction'] = 'Positive' if sens_val > 0 else 'Negative'

    # Normalize only the active parameters
    active_mask = sensitivity_all_params['Effect_Direction'] != 'Not Applicable'
    abs_sens = sensitivity_all_params.loc[active_mask, 'Normalized_Sensitivity'].abs()
    normalized_sens = abs_sens / abs_sens.sum()

    # Create full-length normalized sensitivity column
    sensitivity_all_params['Normalized_Sensitivity_Final'] = 0.0
    sensitivity_all_params.loc[active_mask, 'Normalized_Sensitivity_Final'] = normalized_sens

    # Color assignment
    colors_bar = []
    for param in sensitivity_all_params.index:
        effect = sensitivity_all_params.loc[param, 'Effect_Direction']
        if effect == 'Positive':
            colors_bar.append(COLOR_PALETTE['positive'])
        elif effect == 'Negative':
            colors_bar.append(COLOR_PALETTE['negative'])
        else:
            colors_bar.append(COLOR_PALETTE['not_applicable'])

    # Plot bars
    bars = plt.bar(sensitivity_all_params.index, 
                sensitivity_all_params['Normalized_Sensitivity_Final'] * 100,  # Multiply by 100 for percentage
                color=colors_bar, alpha=0.7)

    # Full cumulative line (including zeros)
    cumulative_all = np.cumsum(sensitivity_all_params['Normalized_Sensitivity_Final'] * 100)  # Multiply by 100 for percentage

    # Plot cumulative line across all bars
    plt.plot(range(len(ordered_params)), cumulative_all,
            color=COLOR_PALETTE['cumulative'], marker='o',
            linewidth=2, markersize=6, label='Cumulative Sum')

    # Axes and layout
    plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
    plt.ylabel('Sensitivity (%)', fontweight='bold', fontsize=14)  # Changed y-axis label
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))  # Changed to show whole numbers for percentage
    plt.ylim(0, 110)  # Adjusted for percentage scale
    plt.grid(True, linestyle='--', alpha=0.3)

    # Bar labels (include zeros)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 3,  # Adjusted position for percentage scale
                f'{height:.0f}%', ha='center', va='bottom',  # Changed format to percentage
                fontsize=14, fontweight='bold')

    # Cumulative value labels (across all bars)
    for i, val in enumerate(cumulative_all):
        plt.text(i, val + 3, f'{val:.0f}%',  # Changed format to percentage
                ha='center', va='bottom', color=COLOR_PALETTE['cumulative'],
                fontsize=14, fontweight='bold')

    # Legend (only positive, negative, and cumulative)
    pos_patch = mpatches.Patch(color=COLOR_PALETTE['positive'], alpha=0.7, label='Positive Effect')
    neg_patch = mpatches.Patch(color=COLOR_PALETTE['negative'], alpha=0.7, label='Negative Effect')
    cum_line = plt.Line2D([], [], color=COLOR_PALETTE['cumulative'], marker='o', label='Cumulative Sum')

    plt.legend(handles=[pos_patch, neg_patch, cum_line], 
        loc='center right', prop={'weight': 'bold', 'size': 14})

    plt.tight_layout()



    # Plot 13: Parameter Correlation Heatmap
    if num_active_params_retention > 1:
        plt.figure(figsize=(8, 6))
        plt.title('13. Retention Time - Parameter Correlation Matrix', 
                 fontweight='bold', fontsize=14)
        param_retention_df = pd.DataFrame(param_variations_retention, 
                                        columns=active_params_retention)
        corr_matrix_retention = param_retention_df.corr()
        sns.heatmap(corr_matrix_retention, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', annot_kws={'size': 14, 'weight':'bold'})
        plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=14)
        plt.yticks(fontweight='bold', fontsize=14)
        plt.tight_layout()

plt.show()