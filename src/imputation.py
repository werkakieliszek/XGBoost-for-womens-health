import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

def load_processed_data():
    """Load the preprocessed data from pickle file."""
    project_dir = Path(__file__).parent.parent
    data_path = project_dir / "data/processed/processed_data.pkl"
    return pd.read_pickle(data_path)

def impute_data(df):
    """
    Perform imputation on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed DataFrame to impute
        
    Returns:
    --------
    pandas.DataFrame
        The imputed DataFrame
    """
    # Basic imputation for binary/count variables
    df['art'] = df['art'].fillna(0)
    
    # Logical imputations based on domain knowledge
    df.loc[(df['gvl'] == 0) & (df['gvlquant'].isna()), 'gvlquant'] = 20
    
    # Fill missing values for frequency variables
    df['sexfreq'] = df['sexfreq'].fillna(0)
    
    # Plasma viral load imputation
    df['pvl'] = df['pvl'].fillna(0)
    df['pvlquant'] = df['pvlquant'].fillna(20)
    df.loc[df['pvlquant'] <= 20, 'pvl'] = 0
    
    # STI binary variable imputation
    df['ct'] = df['ct'].fillna(0)
    df['gc'] = df['gc'].fillna(0)
    df['bv'] = df['bv'].fillna(0)
    df['sy'] = df['sy'].fillna(0)
    df['trich'] = df['trich'].fillna(0)
    
    # Genital viral load imputation
    df['gvl'] = df['gvl'].fillna(0)
    df['gvlquant'] = df['gvlquant'].fillna(20)
    
    # Advanced imputation for continuous variables
    columns_with_missing = ['cd4', 'hbg']
    imputer = IterativeImputer(max_iter=10, tol=0.001)
    imputer.fit(df[columns_with_missing])
    imputed_data = imputer.transform(df[columns_with_missing])
    imputed_df = pd.DataFrame(imputed_data, columns=columns_with_missing)
    df.loc[:, columns_with_missing] = imputed_df.values
    
    # Create composite risk score
    composite_df = df.copy()
    scaler = MinMaxScaler()
    composite_df[['sexfreq', 'sexparts', 'previous_uti_count']] = scaler.fit_transform(
        composite_df[['sexfreq', 'sexparts', 'previous_uti_count']]
    )
    
    df['composite_risk_score'] = (
        0.4 * composite_df['sexfreq'] +
        0.3 * composite_df['sexparts'] +
        0.3 * composite_df['previous_uti_count']
    )
    
    return df

def check_missing_values(df):
    """Check for any remaining missing values in the DataFrame."""
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    
    if len(missing_values) > 0:
        print("\nRemaining missing values:")
        print(missing_values)
    else:
        print("\nNo missing values remaining in the dataset.")
    
    return missing_values

def main():
    """Main execution function."""
    # Load data
    print("Loading preprocessed data...")
    df = load_processed_data()
    
    # Perform imputation
    print("Performing imputation...")
    imputed_df = impute_data(df)
    
    # Check for missing values
    print("Checking for missing values...")
    check_missing_values(imputed_df)
    
    # Save imputed data
    project_dir = Path(__file__).parent.parent
    output_path = project_dir / "data/processed/imputed_data.pkl"
    imputed_df.to_pickle(output_path)
    print(f"Imputed data saved to {output_path}")
    
    return imputed_df

if __name__ == "__main__":
    main()