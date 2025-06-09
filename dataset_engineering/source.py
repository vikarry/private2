from dataset_engineering.env import Env
import pandas as pd
import pywt


def compute_high_freq(data_dict, tick):
    time_series = data_dict[tick].values.squeeze()
    (cA, cD) = pywt.dwt(time_series, 'db4', 'smooth')
    try:
        data_dict[tick][f'{tick}_highfreq'] = pywt.idwt(None, cD, 'db4', 'smooth')[1:]
    except:
        data_dict[tick][f'{tick}_highfreq'] = pywt.idwt(None, cD, 'db4', 'smooth')
    return data_dict


def compute_data():
    data = pd.read_excel('data_.xlsx', header=[1], sheet_name='extract')
    tickers = pd.read_excel('data_.xlsx', header=[0], sheet_name='extract').columns
    tickers = [c for c in tickers if 'Unnamed' not in c]

    tickers_nondaily = ['CPI_CHNG Index', 'CPUPXCHG Index', 'INJCJC Index', 'INJCJC4 Index', 'INJCJCNS Index',
                        'INJCJMOM Index', 'INJCJYOY Index', 'INJCSP Index', 'INJCSPNS Index', 'NFP_TCH Index']

    tickers_drop = ['ADP_CHNG Index', 'US5020_Curncy', 'USGG20YR Index', 'US5010_Curncy', 'USSFCT10_Curncy']

    fields_base = ['Date', 'PX_LAST']
    data_dict = {}
    for i in range(len(tickers)):
        tick = tickers[i]
        if tick not in tickers_drop:
            if i > 0:
                fields = [f'{f}.{i}' for f in fields_base]
            else:
                fields = fields_base
            data_dict[tick] = data[fields].iloc[1:]
            data_dict[tick].set_index(data_dict[tick].columns[0], inplace=True)
            data_dict[tick].columns = [tick]
            data_dict[tick].dropna(inplace=True)
            data_dict[tick].index.name = 'Date'
            if tick not in tickers_nondaily:
                data_dict = compute_high_freq(data_dict, tick)

    # Re create SOFR
    data_dict['USSFCT02_Curncy'] = data_dict['USGG2YR Index']['USGG2YR_Index'] - data_dict['US5S02_Curncy'][
        'US5S02_Curncy']
    data_dict['USSFCT02_Curncy'] = data_dict['USSFCT02_Curncy'].to_frame('USSFCT02_Curncy')
    data_dict['USSFCT02_Curncy'].dropna(inplace=True)
    data_dict['USSFCT05_Curncy'] = data_dict['USGG5YR Index']['USGG5YR_Index'] - data_dict['US5S05_Curncy'][
        'US5S05_Curncy']
    data_dict['USSFCT05_Curncy'] = data_dict['USSFCT05_Curncy'].to_frame('USSFCT05_Curncy')
    data_dict['USSFCT05_Curncy'].dropna(inplace=True)

    data_dict = compute_high_freq(data_dict, 'USSFCT02_Curncy')
    data_dict = compute_high_freq(data_dict, 'USSFCT05_Curncy')

    tickers = list(data_dict.keys())

    max_len = 0
    max_ticker = None
    for ticker in tickers:
        len_ticker = len(data_dict[ticker])
        if len_ticker > max_len:
            max_len = len_ticker
            max_ticker = ticker

    data = data_dict[max_ticker]
    for ticker in tickers:
        if ticker != max_ticker and ticker not in ['US5S02_Curncy', 'US5S05_Curncy']:
            data = pd.merge(data, data_dict[ticker], on='Date', how='left')

    tickers.remove('US5S02_Curncy')
    tickers.remove('US5S05_Curncy')
    data[tickers_nondaily] = data[tickers_nondaily].ffill()
    features = list(data.columns)

    target = pd.read_excel('data_.xlsx', sheet_name='bench').dropna()
    target.set_index(target.columns[0], inplace=True)
    target.index.name = 'Date'

    na = data.isna().sum(axis=1)
    start_date = data[na == 0].index[0]
    data = data.loc[start_date:]

    data = data.merge(target, on='Date')
    dates = pd.DataFrame(index=list(data.index))
    target = list(target.columns)[0]
    return data, dates, target, features


def compute_envs(data: pd.DataFrame, features: list, train_dates, val_dates, test_dates,
                 target: str, days_ahead: int, data_test: pd.DataFrame = None,
                 fix_outliers: bool = False, add_transf: bool = False, no_missing_timesteps: bool = False,
                 scale_target: bool = False, n_steps: int = 100, type_scaler: str = 'normalizer',
                 method_aggregate_target: str = 'returns', type_learn: str = 'regression', verbose: bool = False,
                 threshold=None):
    train_env = Env(data, features, train_dates.start, train_dates.end, target, days_ahead,
                    fix_outliers=fix_outliers, add_transf=add_transf, no_missing_timesteps=no_missing_timesteps,
                    scale_target=scale_target, n_steps=n_steps, type_scaler=type_scaler,
                    method_aggregate_target=method_aggregate_target, type_learn=type_learn, verbose=verbose,
                    print=verbose, threshold=threshold)
    scaler_train_X, knn_imputer, scaler_outliers, scaler_outliers_y, scaler_train_y, oneh_encoder, flag_t = (
    train_env.scaler_X,
    train_env.knn_imputer,
    train_env.scaler_outliers,
    train_env.scaler_outliers_y,
    train_env.scaler_y,
    train_env.oneh_encoder,
    train_env.flag_threshold)
    if val_dates is not None:
        val_env = Env(data, features, val_dates.start, val_dates.end, target, days_ahead, scaler_train_X=scaler_train_X,
                      knn_imputer=knn_imputer, fix_outliers=fix_outliers, scaler_outliers=scaler_outliers,
                      add_transf=add_transf, no_missing_timesteps=no_missing_timesteps,
                      scaler_outliers_y=scaler_outliers_y,
                      scale_target=scale_target, scaler_train_y=scaler_train_y, n_steps=n_steps,
                      type_scaler=type_scaler, oneh_encoder=oneh_encoder, flag_threshold=flag_t,
                      method_aggregate_target=method_aggregate_target, type_learn=type_learn, verbose=verbose,
                      print=verbose, threshold=threshold)
    else:
        val_env = None

    if data_test is not None:
        start_test, end_test = None, None
        data = data_test
    else:
        if test_dates is not None:
            start_test, end_test = test_dates.start, test_dates.end
    if data_test is not None or test_dates is not None:
        test_env = Env(data, features, start_test, end_test, target, days_ahead, scaler_train_X=scaler_train_X,
                       knn_imputer=knn_imputer, fix_outliers=fix_outliers, scaler_outliers=scaler_outliers,
                       add_transf=add_transf, no_missing_timesteps=no_missing_timesteps,
                       scaler_outliers_y=scaler_outliers_y,
                       scale_target=scale_target, scaler_train_y=scaler_train_y, n_steps=n_steps,
                       type_scaler=type_scaler, oneh_encoder=oneh_encoder, flag_threshold=flag_t,
                       method_aggregate_target=method_aggregate_target, type_learn=type_learn, verbose=verbose,
                       print=verbose, threshold=threshold)
    else:
        test_env = None

    return train_env, val_env, test_env