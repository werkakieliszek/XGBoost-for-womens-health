#!/usr/bin/env python
# generate_synthetic_data.py - Generate synthetic IUD clinical trial data

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import bernoulli
from pathlib import Path
from datetime import datetime
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer



def generate_continuous_variables(df, n_samples=20000):
    """Generate synthetic continuous variables using Gaussian Copula."""
    # Extract the continuous variables from the original dataset
    data = df[continuous_vars]
    
    # Fit a Gaussian Copula to the data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    copula = GaussianCopulaSynthesizer(metadata)
    copula.fit(data)
    
    # Generate new samples
    samples = copula.sample(n_samples)
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(samples, columns=continuous_vars)
    
    # Apply inverse transform to match original distributions more closely
    for col in continuous_vars:
        original = df[col]
        synthetic = synthetic_df[col]
        
        # Fit a kernel density estimation to the original data
        kde = stats.gaussian_kde(original)
        
        # Generate samples from the KDE
        synthetic_df[col] = kde.resample(n_samples)[0]
        
        if col in ['since_last_treatment_ct', 'since_last_treatment_trich', 'since_last_treatment_bv', 'since_last_treatment_gc']:
            synthetic_df[col] = np.round(np.maximum(synthetic_df[col], -1)).astype(int)
        elif col in ['composite_risk_score']:
            synthetic_df[col] = np.maximum(synthetic_df[col], 0)
        else:
            synthetic_df[col] = np.round(np.maximum(synthetic_df[col], 0)).astype(int)
    
    return synthetic_df

def generate_binary_variables(df, synthetic_df, n_samples=10000):
    """Generate synthetic binary variables based on original probabilities."""
    # List of binary variables to generate
    binary_probs = {var: df[var].mean() for var in binary_vars if var in df.columns}
    
    # Generate binary variables
    for var in binary_vars:
        prob = binary_probs[var]
        synthetic_df[var] = bernoulli.rvs(prob, size=n_samples)
    
    return synthetic_df

def add_gvl_pvl_columns(synthetic_df):
    """Add calculated GVL and PVL columns based on quantitative values."""
    synthetic_df['gvl'] = (synthetic_df['gvlquant'] > 0).astype(int)
    synthetic_df['pvl'] = (synthetic_df['pvlquant'] > 20).astype(int)
    synthetic_df['lagged_gvl'] = (synthetic_df['lagged_gvlquant'] > 0).astype(int)
    synthetic_df['lagged_pvl'] = (synthetic_df['lagged_pvlquant'] > 20).astype(int)
    
    return synthetic_df

def generate_lagged_variables(df, synthetic_df, n_samples=10000):
    """Generate synthetic lagged variables preserving correlations."""
    for var in lagged_vars:
        original_var = var.replace('lagged_', '')
        if original_var in synthetic_df.columns:
            corr = df[var].corr(df[original_var]) if var in df.columns else 0.6
            temp = np.random.normal(0, 1, n_samples)
            synthetic_df[var] = corr * synthetic_df[original_var] + np.sqrt(1 - corr**2) * temp
            synthetic_df[var] = (synthetic_df[var] > 0).astype(int)  # Convert to binary
        else:
            # If original variable doesn't exist, generate based on its distribution in df
            prob = df[var].mean() if var in df.columns else 0.5
            synthetic_df[var] = bernoulli.rvs(prob, size=n_samples)
    
    # Handle variables separately if they weren't in original list
    if 'sy' not in synthetic_df.columns:
        prob_sy = df['lagged_sy'].mean() if 'lagged_sy' in df.columns else 0.2
        synthetic_df['lagged_sy'] = bernoulli.rvs(prob_sy, size=n_samples)

    if 'ct' not in synthetic_df.columns:
        prob_ct = df['lagged_ct'].mean() if 'lagged_ct' in df.columns else 0.2
        synthetic_df['lagged_ct'] = bernoulli.rvs(prob_ct, size=n_samples)

    if 'cttreat' not in synthetic_df.columns:
        prob_cttreat = df['lagged_cttreat'].mean() if 'lagged_cttreat' in df.columns else 0.2
        synthetic_df['lagged_cttreat'] = bernoulli.rvs(prob_cttreat, size=n_samples)
    
    if 'gctreat' not in synthetic_df.columns:
        prob_gctreat = df['lagged_gctreat'].mean() if 'lagged_gctreat' in df.columns else 0.2
        synthetic_df['lagged_gctreat'] = bernoulli.rvs(prob_gctreat, size=n_samples)
        
    if 'trich' not in synthetic_df.columns:
        prob_trich = df['lagged_trich'].mean() if 'lagged_trich' in df.columns else 0.2
        synthetic_df['lagged_trich'] = bernoulli.rvs(prob_trich, size=n_samples)
        
    if 'bv' not in synthetic_df.columns:
        prob_bv = df['lagged_bv'].mean() if 'lagged_bv' in df.columns else 0.2
        synthetic_df['lagged_bv'] = bernoulli.rvs(prob_bv, size=n_samples)
        
    if 'gc' not in synthetic_df.columns:
        prob_gc = df['lagged_gc'].mean() if 'lagged_gc' in df.columns else 0.2
        synthetic_df['lagged_gc'] = bernoulli.rvs(prob_gc, size=n_samples)
    
    return synthetic_df

def generate_synthetic_data(df, n_samples=20000):
    """Main function to generate complete synthetic dataset."""
    # Generate continuous variables
    synthetic_continuous = generate_continuous_variables(df, n_samples)
    
    # Add 'gvl' and 'pvl' columns based on 'gvlquant' and 'pvlquant'
    synthetic_continuous = add_gvl_pvl_columns(synthetic_continuous)
    
    # Generate binary variables
    synthetic_continuous = generate_binary_variables(df, synthetic_continuous, n_samples)
    
    # Generate lagged binary variables
    synthetic_df = generate_lagged_variables(df, synthetic_continuous, n_samples)
    
    return synthetic_df

# Variable definitions
continuous_vars = ['hbg', 'cd4', 'age', 'pvlquant', 'gravid', 'sexfreq', 'gvlquant', 'lagged_gvlquant', 
                   'lagged_pvlquant', 'hiv_diagnosis', 'sexparts', 'since_last_treatment_trich', 
                   'since_last_treatment_bv', 'since_last_treatment_ct', 'since_last_treatment_gc', 
                   'previous_uti_count', 'previous_trich_count', 'in_the_trial', 'previous_bv_count', 
                   'lagged_hbg', 'age_sexparts_interaction', 'previous_ct_count', 'previous_gc_count', 
                   'composite_risk_score', 'lagged_cd4']

binary_vars = ['artgroup', 'lagged_PID', 'PID', 'arm', 'education', 'employed', 'iud_reason_Bleeding', 
               'iud_reason_Wants pregnancy', 'iud_reason_Pain', 'iud_reason_PID', 
               'iud_reason_Pregnancy with IUD', 'iud_reason_Ectopic preg', 'iud_reason_Risk of infection', 
               'iud_reason_Colposcopy', 'iud_reason_Relocating', 'art', 'everpreg', 'lagged_art']

lagged_vars = ['lagged_gc', 'lagged_sy', 'lagged_ct', 'lagged_cttreat', 
               'lagged_trich', 'lagged_bv', 'lagged_trichtreat', 'lagged_bvtreat']

def main():
    """Main execution function."""
    # Set up paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    
    # Create synthetic data directory if it doesn't exist
    synthetic_dir = data_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    original_data_path = data_dir / "processed" / "imputed_data.pkl"
    df = pd.read_pickle(original_data_path)
    
    # Generate synthetic data
    print(f"Generating synthetic data based on {original_data_path}...")
    n_samples = 20000
    synthetic_df = generate_synthetic_data(df, n_samples)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = synthetic_dir / f"synthetic_data_{timestamp}.pkl"
    synthetic_df.to_pickle(output_path)
    
    # Also save CSV for easier viewing
    csv_path = synthetic_dir / f"synthetic_data_{timestamp}.csv"
    synthetic_df.to_csv(csv_path, index=False)
    
    # Generate condition-specific datasets
    for target in ['trich', 'bv', 'ct', 'gc']:
        target_dir = synthetic_dir / target
        target_dir.mkdir(exist_ok=True)
        synthetic_df.to_pickle(target_dir / f"synthetic_{target}_{timestamp}.pkl")
    
    print(f"Synthetic data generated successfully!")
    print(f"Main file saved to: {output_path}")
    print(f"CSV version saved to: {csv_path}")
    print(f"Target-specific files saved in: {synthetic_dir}")
    print("\nSynthetic data summary statistics:")
    print(synthetic_df.describe())

if __name__ == "__main__":
    main()
