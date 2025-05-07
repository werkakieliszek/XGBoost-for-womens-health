import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, matthews_corrcoef, f1_score, classification_report, auc
from sklearn.inspection import permutation_importance
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def load_processed_data():
    """Load the preprocessed data from pickle file."""
    project_dir = Path(__file__).parent.parent
    data_path = project_dir / "data/processed/processed_data.pkl"
    return pd.read_pickle(data_path)

def prepare_data(df):
    columns_to_drop_gc = ['trichtreat', 'gctreat', 'bvtreat', 'cttreat', 'bv', 'sy', 'trich', 'gc', 'ct']
    X_gc = df.drop(columns=columns_to_drop_gc)
    y_gc = df['gc']

    return X_gc, y_gc

def train_and_select_features(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Resample
    smotetomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smotetomek.fit_resample(X_train, y_train)
    # Feature selection
    estimator = XGBClassifier(random_state=42, eval_metric='auc')
    selector = RFE(estimator, n_features_to_select=25, step=1)
    selector.fit(X_train_res, y_train_res)
    X_train_sel = selector.transform(X_train_res)
    X_test_sel = selector.transform(X_test)
    selected_features = X.columns[selector.support_]
    return X_train_sel, X_test_sel, y_train_res, y_test, selected_features, selector

def build_pipeline(selected_features):
    xgb = XGBClassifier(random_state=42, eval_metric='auc')
    logreg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    voting_clf = VotingClassifier(estimators=[('xgb', xgb), ('logreg', logreg)], voting='soft')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', voting_clf)
    ])
    return pipeline

def get_param_grid():
    param_grid = {
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
    return param_grid

def model_training(X_train, y_train, pipeline, param_grid):
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=100,
        scoring='precision',
        cv=StratifiedKFold(n_splits=5),
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search

def main(df):
    X_gc, y_gc = prepare_data(df)
    X_train_gc, X_test_gc, y_train_gc, y_test_gc, selected_features_gc, selector_gc = train_and_select_features(X_gc, y_gc)
    print("Selected features:", selected_features_gc)
    pipeline_gc = build_pipeline(selected_features_gc)
    param_grid_gc = get_param_grid()
    random_search_gc = model_training(X_train_gc, y_train_gc, pipeline_gc, param_grid_gc)
    print("Best parameters found: ", random_search_gc.best_params_)
    print("Best cross-validation precision: ", random_search_gc.best_score_)
    best_model_gc = random_search_gc.best_estimator_
    return best_model_gc, X_test_gc, y_test_gc, selected_features_gc

# If running as a script:
if __name__ == "__main__":
    df = load_processed_data()
    best_model_gc, X_test_gc, y_test_gc, selected_features_gc = main(df)