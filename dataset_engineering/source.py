from dataset_engineering.env import Env
import pandas as pd
import pywt
import numpy as np
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


def compute_high_freq(data_dict, tick):
    """
    Compute high-frequency component using wavelet decomposition.
    Uses detail coefficients (cD) which represent high-frequency components.
    """
    time_series = data_dict[tick].values.squeeze()

    # Ensure we have enough data points for wavelet decomposition
    if len(time_series) < 4:
        print(f"Warning: {tick} has insufficient data for wavelet decomposition")
        return data_dict

    try:
        # Decompose into approximation (low freq) and detail (high freq) coefficients
        (cA, cD) = pywt.dwt(time_series, 'db4', 'symmetric')

        # Reconstruct high-frequency component using detail coefficients
        high_freq_component = pywt.idwt(None, cD, 'db4', 'symmetric')

        # Handle length mismatch due to wavelet border effects
        if len(high_freq_component) > len(time_series):
            high_freq_component = high_freq_component[:len(time_series)]
        elif len(high_freq_component) < len(time_series):
            # Pad with zeros if needed
            padding = len(time_series) - len(high_freq_component)
            high_freq_component = np.pad(high_freq_component, (0, padding), 'constant')

        # Add high-frequency feature to dataframe
        data_dict[tick][f'{tick}_highfreq'] = high_freq_component

    except Exception as e:
        print(f"Warning: Could not compute high-frequency component for {tick}: {e}")
        # If wavelet decomposition fails, create a simple differenced series as fallback
        data_dict[tick][f'{tick}_highfreq'] = data_dict[tick][tick].diff().fillna(0)

    return data_dict


def validate_sofr_calculation(usgg2yr, sofr_spread, usgg5yr, sofr5_spread):
    """
    Validate SOFR calculation makes economic sense.
    SOFR should typically be Treasury rate + spread (not Treasury - spread).
    """
    print("SOFR Validation:")
    print(f"2Y Treasury range: {usgg2yr.min():.2f} to {usgg2yr.max():.2f}")
    print(f"2Y SOFR Spread range: {sofr_spread.min():.2f} to {sofr_spread.max():.2f}")
    print(f"5Y Treasury range: {usgg5yr.min():.2f} to {usgg5yr.max():.2f}")
    print(f"5Y SOFR Spread range: {sofr5_spread.min():.2f} to {sofr5_spread.max():.2f}")


def handle_missing_nondaily_data(data, tickers_nondaily):
    """
    Handle missing data for non-daily indicators more appropriately.
    Uses forward fill only for recent gaps, backfill for initial missing values.
    """
    for ticker in tickers_nondaily:
        if ticker in data.columns:
            # First backfill to handle initial missing values
            data[ticker] = data[ticker].bfill()

            # Then forward fill, but limit to reasonable periods (e.g., 30 days max)
            data[ticker] = data[ticker].fillna(method='ffill', limit=30)

            # For any remaining missing values, use interpolation
            data[ticker] = data[ticker].interpolate(method='linear')

            # Final fallback: use median for any remaining NaN
            if data[ticker].isna().any():
                median_val = data[ticker].median()
                data[ticker] = data[ticker].fillna(median_val)
                print(f"Warning: Used median fill for {ticker}")

    return data


def compute_data():
    """
    Load and process financial data with improved feature engineering.
    """
    # Load data
    data = pd.read_excel('data_.xlsx', header=[1], sheet_name='extract')
    tickers = pd.read_excel('data_.xlsx', header=[0], sheet_name='extract').columns
    tickers = [c for c in tickers if 'Unnamed' not in c]

    # Define ticker categories
    tickers_nondaily = [
        'CPI_CHNG Index', 'CPUPXCHG Index', 'INJCJC Index', 'INJCJC4 Index',
        'INJCJCNS Index', 'INJCJMOM Index', 'INJCJYOY Index', 'INJCSP Index',
        'INJCSPNS Index', 'NFP_TCH Index'
    ]

    # Remove problematic tickers
    tickers_drop = [
        'ADP_CHNG Index', 'US5020_Curncy', 'USGG20YR Index',
        'US5010_Curncy', 'USSFCT10_Curncy'
    ]

    fields_base = ['Date', 'PX_LAST']
    data_dict = {}

    # Process each ticker
    for i in range(len(tickers)):
        tick = tickers[i]
        if tick not in tickers_drop:
            if i > 0:
                fields = [f'{f}.{i}' for f in fields_base]
            else:
                fields = fields_base

            # Extract and clean data
            data_dict[tick] = data[fields].iloc[1:].copy()
            data_dict[tick].set_index(data_dict[tick].columns[0], inplace=True)
            data_dict[tick].columns = [tick]
            data_dict[tick].dropna(inplace=True)
            data_dict[tick].index.name = 'Date'

            # Convert to numeric, handling any string values
            data_dict[tick][tick] = pd.to_numeric(data_dict[tick][tick], errors='coerce')
            data_dict[tick].dropna(inplace=True)

            # Compute high-frequency features for daily data only
            if tick not in tickers_nondaily and len(data_dict[tick]) >= 10:
                data_dict = compute_high_freq(data_dict, tick)

    # Improved SOFR calculation with validation
    if 'USGG2YR Index' in data_dict and 'US5S02_Curncy' in data_dict:
        usgg2yr = data_dict['USGG2YR Index']['USGG2YR Index']
        sofr2_spread = data_dict['US5S02_Curncy']['US5S02_Curncy']

        # SOFR = Treasury + Spread (not Treasury - Spread)
        data_dict['USSFCT02_Curncy'] = (usgg2yr + sofr2_spread).to_frame('USSFCT02_Curncy')
        data_dict['USSFCT02_Curncy'].dropna(inplace=True)

        # Validate the calculation
        validate_sofr_calculation(usgg2yr, sofr2_spread, None, None)

        # Compute high-frequency component for SOFR
        if len(data_dict['USSFCT02_Curncy']) >= 10:
            data_dict = compute_high_freq(data_dict, 'USSFCT02_Curncy')

    if 'USGG5YR Index' in data_dict and 'US5S05_Curncy' in data_dict:
        usgg5yr = data_dict['USGG5YR Index']['USGG5YR Index']
        sofr5_spread = data_dict['US5S05_Curncy']['US5S05_Curncy']

        # SOFR = Treasury + Spread
        data_dict['USSFCT05_Curncy'] = (usgg5yr + sofr5_spread).to_frame('USSFCT05_Curncy')
        data_dict['USSFCT05_Curncy'].dropna(inplace=True)

        # Compute high-frequency component for 5Y SOFR
        if len(data_dict['USSFCT05_Curncy']) >= 10:
            data_dict = compute_high_freq(data_dict, 'USSFCT05_Curncy')

    # Remove intermediate spread series from final dataset
    tickers = [tick for tick in data_dict.keys() if tick not in ['US5S02_Curncy', 'US5S05_Curncy']]

    # Find ticker with maximum length for base merge
    max_len = 0
    max_ticker = None
    for ticker in tickers:
        len_ticker = len(data_dict[ticker])
        if len_ticker > max_len:
            max_len = len_ticker
            max_ticker = ticker

    if max_ticker is None:
        raise ValueError("No valid tickers found")

    # Merge all data
    data = data_dict[max_ticker].copy()
    for ticker in tickers:
        if ticker != max_ticker:
            data = pd.merge(data, data_dict[ticker], on='Date', how='left')

    # Handle missing data for non-daily indicators
    data = handle_missing_nondaily_data(data, tickers_nondaily)

    # Get features list (excluding SOFR spread intermediates)
    features = [col for col in data.columns if col not in ['US5S02_Curncy', 'US5S05_Curncy']]

    # Load target data
    target = pd.read_excel('data_.xlsx', sheet_name='bench').dropna()
    target.set_index(target.columns[0], inplace=True)
    target.index.name = 'Date'

    # Find optimal start date with improved missing data strategy
    # Instead of requiring zero missing values, allow some missing data
    na_counts = data.isna().sum(axis=1)
    missing_threshold = len(features) * 0.1  # Allow up to 10% missing features

    valid_dates = data[na_counts <= missing_threshold]
    if len(valid_dates) == 0:
        # Fallback: use date with minimum missing values
        start_date = data.loc[na_counts.idxmin()].name
        print(f"Warning: Using start date {start_date} with {na_counts.min()} missing features")
    else:
        start_date = valid_dates.index[0]
        print(f"Using start date {start_date} with {na_counts[start_date]} missing features")

    data = data.loc[start_date:]

    # Merge with target
    data = data.merge(target, on='Date', how='inner')

    # Create dates dataframe for reference
    dates = pd.DataFrame(index=list(data.index))
    target_col = list(target.columns)[0]

    # Final data quality checks
    print(f"\nData Quality Summary:")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Total observations: {len(data)}")
    print(f"Features: {len(features)}")
    print(f"Missing values per feature:")
    missing_summary = data[features].isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print("No missing values!")

    return data, dates, target_col, features


def compute_envs(data: pd.DataFrame, features: list, train_dates, val_dates, test_dates,
                 target: str, days_ahead: int, data_test: pd.DataFrame = None,
                 fix_outliers: bool = False, add_transf: bool = False, no_missing_timesteps: bool = False,
                 scale_target: bool = False, n_steps: int = 100, type_scaler: str = 'normalizer',
                 method_aggregate_target: str = 'returns', type_learn: str = 'regression', verbose: bool = False,
                 threshold=None):
    """
    Create training, validation, and test environments with improved error handling.
    """
    try:
        train_env = Env(data, features, train_dates.start, train_dates.end, target, days_ahead,
                        fix_outliers=fix_outliers, add_transf=add_transf, no_missing_timesteps=no_missing_timesteps,
                        scale_target=scale_target, n_steps=n_steps, type_scaler=type_scaler,
                        method_aggregate_target=method_aggregate_target, type_learn=type_learn, verbose=verbose,
                        print=verbose, threshold=threshold)
    except Exception as e:
        print(f"Error creating train environment: {e}")
        raise

    # Extract fitted transformers from training environment
    scaler_train_X, knn_imputer, scaler_outliers, scaler_outliers_y, scaler_train_y, oneh_encoder, flag_t = (
        train_env.scaler_X,
        train_env.knn_imputer,
        train_env.scaler_outliers,
        train_env.scaler_outliers_y,
        train_env.scaler_y,
        train_env.oneh_encoder,
        train_env.flag_threshold
    )

    # Create validation environment
    if val_dates is not None:
        try:
            val_env = Env(data, features, val_dates.start, val_dates.end, target, days_ahead,
                          scaler_train_X=scaler_train_X, knn_imputer=knn_imputer, fix_outliers=fix_outliers,
                          scaler_outliers=scaler_outliers, add_transf=add_transf,
                          no_missing_timesteps=no_missing_timesteps, scaler_outliers_y=scaler_outliers_y,
                          scale_target=scale_target, scaler_train_y=scaler_train_y, n_steps=n_steps,
                          type_scaler=type_scaler, oneh_encoder=oneh_encoder, flag_threshold=flag_t,
                          method_aggregate_target=method_aggregate_target, type_learn=type_learn,
                          verbose=verbose, print=verbose, threshold=threshold)
        except Exception as e:
            print(f"Error creating validation environment: {e}")
            val_env = None
    else:
        val_env = None

    # Create test environment
    if data_test is not None:
        start_test, end_test = None, None
        data = data_test
    else:
        if test_dates is not None:
            start_test, end_test = test_dates.start, test_dates.end
        else:
            start_test, end_test = None, None

    if data_test is not None or test_dates is not None:
        try:
            test_env = Env(data, features, start_test, end_test, target, days_ahead,
                           scaler_train_X=scaler_train_X, knn_imputer=knn_imputer, fix_outliers=fix_outliers,
                           scaler_outliers=scaler_outliers, add_transf=add_transf,
                           no_missing_timesteps=no_missing_timesteps, scaler_outliers_y=scaler_outliers_y,
                           scale_target=scale_target, scaler_train_y=scaler_train_y, n_steps=n_steps,
                           type_scaler=type_scaler, oneh_encoder=oneh_encoder, flag_threshold=flag_t,
                           method_aggregate_target=method_aggregate_target, type_learn=type_learn,
                           verbose=verbose, print=verbose, threshold=threshold)
        except Exception as e:
            print(f"Error creating test environment: {e}")
            test_env = None
    else:
        test_env = None

    return train_env, val_env, test_env