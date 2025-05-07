import pandas as pd
from pathlib import Path

def load_data(data_path: str) -> pd.DataFrame:
    """Load raw data from Excel file."""
    return pd.read_excel(data_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    # Sort and convert dates
    df = df.sort_values(by=['fakeid', 'month'])
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col])
    
    # Encode categorical variables
    df['arm'] = pd.Categorical(df['arm'], categories=df['arm'].unique()).codes
    df['artgroup'] -= 1  # Adjust artgroup encoding
    
    # Create binary features from iud_reason
    unique_reasons = df['iud_reason'].dropna().unique()
    for reason in unique_reasons:
        df[f'iud_reason_{reason}'] = df['iud_reason'].apply(lambda x: 1 if x == reason else 0)
    
    # Feature engineering
    df = create_time_based_features(df)
    df = create_treatment_features(df)
    df = create_interaction_features(df)
    
    # Remove unnecessary columns
    cols_to_remove = [
        'expl', 'IUDDC', 'iud_remove', 'iud_reason', 'iud_expulsion',
        'iud_replaced', 'iud_nonelect', 'anyrti', 'ECTOP', 'preg', 'ATPOP'
    ]
    df.drop(columns=cols_to_remove, inplace=True)
    
    # Date imputation and filtering
    df = impute_missing_dates(df, 'gvldate', 'month')
    df = df[df['gvldate'] < df['enddate']]
    
    # Create historical counts
    df = create_historical_counts(df)
    
    # Create time since treatment features
    df = create_time_since_treatment(df)

    df = drop_columns(df)

    df = create_lagged_variables(df)
    
    # Set index
    df.set_index(['fakeid', 'month'], inplace=True)
    
    return df

def create_lagged_variables(df, exclude_vars=None, treatment_vars=None, clinical_vars=None):
    """
    Create lagged versions of variables in the DataFrame, grouped by 'fakeid'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with multiindex where one level is 'fakeid'
    exclude_vars : list, optional
        List of variables to exclude from lagging
    treatment_vars : list, optional
        List of treatment variables to lag
    clinical_vars : list, optional
        List of clinical variables to lag
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional lagged columns
    """
    # Use default lists if not provided
    if exclude_vars is None:
        exclude_vars = ['hiv_diagnosis', 'in_the_trial', 'arm', 'since_last_treatment_ct', 
                         'since_last_treatment_trich', 'since_last_treatment_bv', 
                         'since_last_treatment_gc', 'previous_trich_count', 
                         'previous_bv_count', 'previous_gc_count', 'previous_ct_count', 
                         'previous_uti_count', 'artgroup']
    
    if treatment_vars is None:
        treatment_vars = ['arm', 'art', 'artgroup', 'trichtreat', 'bvtreat', 'cttreat', 
                          'gctreat', 'in_the_trial', 'since_last_treatment_trich', 
                          'since_last_treatment_ct', 'since_last_treatment_bv', 
                          'since_last_treatment_gc']
    
    if clinical_vars is None:
        clinical_vars = ['PID', 'gvl', 'gvlquant', 'pvl', 'pvlquant', 'trich', 'bv', 
                         'sy', 'ct', 'gc', 'hbg', 'cd4', 'hiv_diagnosis', 
                         'previous_uti_count', 'previous_ct_count', 'previous_trich_count', 
                         'previous_gc_count', 'previous_bv_count']
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create lagged variables
    for var in treatment_vars + clinical_vars:
        if var not in exclude_vars and var in df.columns:
            result_df[f'lagged_{var}'] = result_df.groupby('fakeid', as_index=False)[var].shift(1)
    
    # Create list of lagged column names
    lagged_columns = [f'lagged_{var}' for var in treatment_vars + clinical_vars 
                     if var not in exclude_vars and var in df.columns]
    
    # Fill NA values with 0
    result_df[lagged_columns] = result_df[lagged_columns].fillna(0)
    
    return result_df 

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based delta features."""
    df['hiv_diagnosis'] = (df['gvldate'] - df['hivdate']).dt.days
    df['in_the_trial'] = (df['enddate'] - df['enrolldate']).dt.days
    return df

def create_treatment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary treatment features."""
    treatments = ['trichtreat', 'bvtreat', 'cttreat', 'gctreat']
    for treatment in treatments:
        df[treatment] = df[f'{treatment}date'].notna().astype(int)
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    df['age_sexparts_interaction'] = df['age'] * df['sexparts']
    return df

def impute_missing_dates(df: pd.DataFrame, date_col: str, month_col: str) -> pd.DataFrame:
    """Impute missing dates based on month offsets."""
    def _impute_group(group):
        # Check if month 0 exists for this group
        if 0 not in group[month_col].values:
            print(f"Warning: No baseline (month 0) for fakeid {group.name}")
            return group  # Return unchanged
            
        base_date = group.loc[group[month_col] == 0, date_col].iloc[0]
        month_offsets = {
            0: pd.DateOffset(0), 3: pd.DateOffset(months=3),
            6: pd.DateOffset(months=6), 12: pd.DateOffset(months=12),
            18: pd.DateOffset(months=18), 24: pd.DateOffset(months=24)
        }
        
        def apply_offset(row):
            if pd.isnull(row[date_col]):
                if row[month_col] in month_offsets:
                    return base_date + month_offsets[row[month_col]]
                else:
                    print(f"Warning: Unknown month value {row[month_col]} for fakeid {group.name}")
                    return row[date_col]  # Keep as null
            return row[date_col]
            
        group[date_col] = group.apply(apply_offset, axis=1)
        return group
    
    result = df.groupby('fakeid', as_index=False).apply(_impute_group).reset_index(drop=True)
    # Check if still have nulls
    null_count = result[date_col].isnull().sum()
    if null_count > 0:
        print(f"Warning: {null_count} null values remain in {date_col}")
    
    return result

def create_historical_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative counts of previous infections."""
    utis = ['ct', 'trich', 'bv', 'gc']
    df['previous_uti_count'] = 0
    
    def _count_group(group):
        # Fill NaN values with 0 before performing operations
        uti_data = group[utis].fillna(0)
        
        # Calculate total previous UTI count
        group['previous_uti_count'] = uti_data.cumsum().shift(fill_value=0).sum(axis=1)
        
        # Calculate individual UTI type counts
        for uti in utis:
            group[f'previous_{uti}_count'] = group[uti].fillna(0).cumsum().shift(fill_value=0)
        
        return group
    
    return df.groupby('fakeid', as_index=False).apply(_count_group).reset_index(drop=True)

def create_time_since_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate days since last treatment for each infection type."""
    uti_types = ['trich', 'bv', 'ct', 'gc']
    
    def _calculate_group(group, uti):
        visit_col = f'{uti}date'
        treat_col = f'{uti}treatdate'
        
        group[f'since_last_treatment_{uti}'] = group.apply(
            lambda row: (row[visit_col] - group.loc[
                (group[treat_col] < row[visit_col]) & 
                group[treat_col].notna(), treat_col
            ].max()).days if not group.loc[
                (group[treat_col] < row[visit_col]) & 
                group[treat_col].notna()
            ].empty else -1, axis=1
        )
        return group
    
    for uti in uti_types:
        df = df.groupby('fakeid', as_index=False).apply(lambda x: _calculate_group(x, uti))
    
    return df.reset_index(drop=True)

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_columns = [col for col in df.columns if 'date' in col]
    df = df.drop(columns=date_columns)

    return df

if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent
    raw_data_path = project_dir / "data/raw/2IUD paper data.xlsx"
    
    df = load_data(raw_data_path)
    processed_df = preprocess_data(df)
    
    processed_path = project_dir / "data/processed/processed_data.pkl"
    processed_df.to_pickle(processed_path)
    print(f"Data preprocessing complete. Saved to {processed_path}")