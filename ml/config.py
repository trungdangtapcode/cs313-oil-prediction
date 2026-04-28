"""
Shared config & utils for ML pipeline.
"""
import os, random, warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.getenv(
    'CLASSIFICATION_DATA_PATH',
    os.path.join(ROOT, 'data', 'processed', 'dataset_final_noleak_step5c_scaler.csv'),
)
PRICE_SOURCE_PATH = os.getenv(
    'CLASSIFICATION_PRICE_SOURCE_PATH',
    os.path.join(ROOT, 'data', 'processed', 'dataset_step4_transformed.csv'),
)
OUT_DIR = os.getenv(
    'CLASSIFICATION_OUT_DIR',
    os.path.join(ROOT, 'ml', 'classification', 'results'),
)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'oil_return_fwd1'
TARGET_DATE_COL = 'oil_return_fwd1_date'
VAL_SPLIT_DATE = '2022-01-01'
SPLIT_DATE = '2023-01-01'
N_SPLITS = 5  # TimeSeriesSplit
RANDOM_STATE = 42

# Features to DROP (non-stationary raw prices, leakage, redundant, near-zero-var)
DROP_COLS = [
    'date',
    TARGET_DATE_COL,
    # raw prices (non-stationary + same-day leakage)
    'oil_close', 'usd_close', 'sp500_close', 'vix_close', 'wti_fred',
    # redundant (|rho|>0.95 or derived duplicate)
    'stress_tone',        # = -gdelt_tone (rho=-1.0)
    'stress_goldstein',   # = -gdelt_goldstein (rho=-1.0)
    'stress_volume',      # = gdelt_volume_log (rho=1.0)
    'gdelt_volume',       # = gdelt_volume_log (rho=1.0)
    # near-zero variance
    'gdelt_data_imputed', # 99.5% = 0
]


def get_train_test_masks(df):
    """Split by target date when a forward target exists."""
    split_col = TARGET_DATE_COL if TARGET_DATE_COL in df.columns else 'date'
    split_values = pd.to_datetime(df[split_col])
    train_mask = split_values < pd.Timestamp(SPLIT_DATE)
    test_mask = split_values >= pd.Timestamp(SPLIT_DATE)
    return train_mask, test_mask, split_col


def get_train_val_test_masks(df):
    """Split by target date into train / validation / final test."""
    split_col = TARGET_DATE_COL if TARGET_DATE_COL in df.columns else 'date'
    split_values = pd.to_datetime(df[split_col])
    train_mask = split_values < pd.Timestamp(VAL_SPLIT_DATE)
    val_mask = (split_values >= pd.Timestamp(VAL_SPLIT_DATE)) & (split_values < pd.Timestamp(SPLIT_DATE))
    test_mask = split_values >= pd.Timestamp(SPLIT_DATE)
    return train_mask, val_mask, test_mask, split_col


def load_data():
    """Load, split, scale. Returns dict with everything needed."""
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    if TARGET_DATE_COL in df.columns:
        df[TARGET_DATE_COL] = pd.to_datetime(df[TARGET_DATE_COL])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_mask, test_mask, split_col = get_train_test_masks(df)
    train = df[train_mask].copy()
    test  = df[test_mask].copy()

    features = [c for c in df.columns if c not in DROP_COLS and c != TARGET]

    X_train = train[features]
    X_test  = test[features]
    y_train = train[TARGET]
    y_test  = test[TARGET]
    dates_train = train['date']
    dates_test  = test['date']

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    print(f'  Features: {len(features)}')
    print(f'  {features}')
    print(f'  Train: {len(X_train)} ({dates_train.iloc[0].date()} -> {dates_train.iloc[-1].date()})')
    print(f'  Test:  {len(X_test)}  ({dates_test.iloc[0].date()} -> {dates_test.iloc[-1].date()})')
    if split_col != 'date':
        print(f'  Train targets: {train[split_col].iloc[0].date()} -> {train[split_col].iloc[-1].date()}')
        print(f'  Test targets:  {test[split_col].iloc[0].date()} -> {test[split_col].iloc[-1].date()}')

    return {
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'dates_train': dates_train, 'dates_test': dates_test,
        'features': features, 'scaler': scaler,
    }


def get_tscv(): 
    return TimeSeriesSplit(n_splits=N_SPLITS) # default is 5 splits


def set_global_seed(seed=RANDOM_STATE):
    """Best-effort reproducibility across numpy/sklearn/xgboost/lightgbm single-process runs."""
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    np.random.seed(seed)
    return seed
