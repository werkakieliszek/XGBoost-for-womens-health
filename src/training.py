import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTETomek, SMOTEENN
from xgboost import XGBClassifier
import joblib
import json

# Versioned output structure
output_dir = Path(__file__).parent.parent / "model_artifacts"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = output_dir / run_id

# Create the parent run directory
run_dir.mkdir(parents=True, exist_ok=True)

def load_processed_data():
    """Load the preprocessed data from pickle file."""
    project_dir = Path(__file__).parent.parent
    data_path = project_dir / "data/processed/imputed_data.pkl"
    return pd.read_pickle(data_path)

def prepare_data(df, target):
    columns_to_drop = ['trichtreat', 'gctreat', 'bvtreat', 'cttreat', 
                      'bv', 'sy', 'trich', 'gc', 'ct']
    columns_to_drop = [col for col in columns_to_drop if col != target]
    date_columns = [col for col in df.columns if 'date' in col]
    X = df.drop(columns=columns_to_drop + date_columns + [target])
    y = df[target]
    return X, y

def train_and_select_features(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smotetomek = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smotetomek.fit_resample(X_train, y_train)
    
    estimator = XGBClassifier(random_state=42, eval_metric='auc')
    selector =  RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(3),
        scoring='precision',
        min_features_to_select=3
    )
    selector.fit(X_train_res, y_train_res)
    X_train_sel = selector.transform(X_train_res)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, y_train_res, y_test, X.columns[selector.support_], selector

def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(random_state=42)),
                ('logreg', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42))
            ],
            voting='soft'
        ))
    ])

def get_param_grid():
    return {
        'classifier__xgb__n_estimators': [100, 200, 300, 400, 500, 700],
        'classifier__xgb__max_depth': [3, 5, 7, 9, 11],
        'classifier__xgb__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'classifier__xgb__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__xgb__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__xgb__scale_pos_weight': [1, 5, 10, 15, 20],
        'classifier__xgb__min_child_weight': [1, 3, 5, 7],
        'classifier__xgb__gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'classifier__xgb__reg_alpha': [0, 0.01, 0.1, 1, 10],
        'classifier__xgb__reg_lambda': [0.1, 1, 10, 20],
        'classifier__xgb__max_delta_step': [0, 1, 2, 5, 10],
        'classifier__logreg__C': [0.01, 0.1, 1, 10, 100, 1000],
        'classifier__logreg__solver': ['liblinear', 'lbfgs', 'sag', 'saga']
    }

def train_model(X_train, y_train):
    pipeline = build_pipeline()
    random_search = RandomizedSearchCV(
        pipeline,
        get_param_grid(),
        n_iter=200,
        scoring='precision',
        cv=StratifiedKFold(n_splits=5),
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search

def save_artifacts(target, model, params, features, score):
    target_dir = run_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, target_dir / "model.joblib")
    
    # Save parameters
    with open(target_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    # Save features
    with open(target_dir / "features.json", 'w') as f:
        json.dump(list(features), f, indent=2)
    
    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "target": target,
        "cv_precision": score,
        "n_features": len(features),
        "data_version": "doi:10.7910/DVN/NTN7KY"
    }
    with open(target_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    df = load_processed_data()
    targets = ['trich', 'bv', 'ct', 'gc']
    
    for target in targets:
        print(f"\n{'='*40}\nTraining {target.upper()} model\n{'='*40}")
        X, y = prepare_data(df, target)
        X_train, X_test, y_train, y_test, features, selector = train_and_select_features(X, y)
        random_search = train_model(X_train, y_train)
        
        save_artifacts(
            target=target,
            model=random_search.best_estimator_,
            params=random_search.best_params_,
            features=features,
            score=random_search.best_score_
        )

        target_dir = run_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        # Save test features and true labels
        test_df = pd.DataFrame(X_test, columns=features)
        test_df['true_label'] = y_test.values
        test_df.to_csv(target_dir / "test_data.csv", index=False)

    print(f"\nAll artifacts saved in versioned structure:\n{run_dir}")