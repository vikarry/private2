import pandas as pd
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import os


class FinancialDataCollector:
    """
    A comprehensive collector for financial market data from multiple sources
    Supports custom date ranges and file paths for saving output
    """

    def __init__(self, start_date=None, end_date=None, output_dir=None):
        """
        Initialize the collector with custom date range and output directory

        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Default is None (uses yfinance default periods)
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Default is None (uses current date)
        output_dir : str, optional
            Directory to save output files. Default is current directory.
        """
        self.data = {}
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir or os.getcwd()

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def _get_history(self, ticker_obj, period=None):
        """
        Helper method to get ticker history based on date settings

        Parameters:
        -----------
        ticker_obj : yfinance.Ticker
            Ticker object
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        if self.start_date and self.end_date:
            return ticker_obj.history(start=self.start_date, end=self.end_date)
        elif self.start_date:
            return ticker_obj.history(start=self.start_date)
        else:
            return ticker_obj.history(period=period or '1mo')

    def get_government_bonds(self, period='1mo'):
        """
        Get government bond data

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        dict
            Dictionary of bond data
        """

        # Dictionary of bond tickers in Yahoo Finance
        bond_tickers = {
            'US Generic Govt 10 Yr': '^TNX',
            'US Generic Govt 5 Yr': '^FVX',
            'US Generic Govt 2 Yr': '^IRX',
            'US Generic Govt 30 Yr': '^TYX'
        }

        bond_data = {}

        for name, ticker in bond_tickers.items():
            try:
                bond = yf.Ticker(ticker)
                history = self._get_history(bond, period)

                if not history.empty:
                    bond_data[name] = {
                        'latest_yield': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")

        self.data['government_bonds'] = bond_data
        return bond_data

    def get_commodities(self, period='1mo'):
        """
        Get commodity data

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        dict
            Dictionary of commodity data
        """

        commodity_tickers = {
            'WTI Crude Oil': 'CL=F',
            'Brent Crude': 'BZ=F',
            'Copper': 'HG=F',
            'Gold': 'GC=F',
            'Soybeans': 'ZS=F',
            'Corn': 'ZC=F',
            'Sugar': 'SB=F'
        }

        commodity_data = {}

        for name, ticker in commodity_tickers.items():
            try:
                commodity = yf.Ticker(ticker)
                history = self._get_history(commodity, period)

                if not history.empty:
                    commodity_data[name] = {
                        'latest_price': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")

        self.data['commodities'] = commodity_data
        return commodity_data

    def get_currencies(self, period='1mo'):
        """
        Get currency data

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        dict
            Dictionary of currency data
        """

        currency_tickers = {
            'Euro': 'EURUSD=X',
            'Japanese Yen': 'JPYUSD=X',
            'British Pound': 'GBPUSD=X',
            'Australian Dollar': 'AUDUSD=X',
            'Chinese Yuan': 'CNYUSD=X',
            'Brazilian Real': 'BRLUSD=X',
            'Dollar Index': 'DX-Y.NYB'
        }

        currency_data = {}

        for name, ticker in currency_tickers.items():
            try:
                currency = yf.Ticker(ticker)
                history = self._get_history(currency, period)

                if not history.empty:
                    currency_data[name] = {
                        'latest_rate': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")

        self.data['currencies'] = currency_data
        return currency_data

    def get_world_indices(self, period='6mo'):
        """
        Get global market indices

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '6mo'

        Returns:
        --------
        dict
            Dictionary of index data with technical indicators
        """

        index_tickers = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ Composite': '^IXIC',
            'NASDAQ 100': '^NDX',
            'Russell 2000': '^RUT',
            'FTSE 100': '^FTSE',
            'DAX': '^GDAXI',
            'Nikkei 225': '^N225',
            'Hang Seng': '^HSI',
            'Shanghai Composite': '000001.SS'
        }

        index_data = {}

        for name, ticker in index_tickers.items():
            try:
                index = yf.Ticker(ticker)
                history = self._get_history(index, period)

                if not history.empty:
                    # Calculate daily returns
                    daily_returns = history['Close'].pct_change()

                    # Calculate technical indicators
                    sma_5 = history['Close'].rolling(window=5).mean()
                    sma_10 = history['Close'].rolling(window=10).mean()
                    sma_20 = history['Close'].rolling(window=20).mean()
                    sma_50 = history['Close'].rolling(window=50).mean()
                    sma_200 = history['Close'].rolling(window=200).mean()

                    # Volatility calculations
                    volatility_10d = daily_returns.rolling(window=10).std() * np.sqrt(252) * 100
                    volatility_30d = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
                    volatility_60d = daily_returns.rolling(window=60).std() * np.sqrt(252) * 100

                    # RSI Calculation
                    delta = history['Close'].diff()
                    up = delta.copy()
                    up[up < 0] = 0
                    down = -delta.copy()
                    down[down < 0] = 0
                    avg_gain_14 = up.rolling(window=14).mean()
                    avg_loss_14 = down.rolling(window=14).mean()
                    rs_14 = avg_gain_14 / avg_loss_14
                    rsi_14 = 100 - (100 / (1 + rs_14))

                    # Store all calculated metrics
                    index_data[name] = {
                        'latest_price': history['Close'].iloc[-1],
                        'daily_change_pct': daily_returns.iloc[-1] * 100 if not pd.isna(daily_returns.iloc[-1]) else 0,
                        'sma_5': sma_5.iloc[-1],
                        'sma_10': sma_10.iloc[-1],
                        'sma_20': sma_20.iloc[-1],
                        'sma_50': sma_50.iloc[-1],
                        'sma_200': sma_200.iloc[-1],
                        'volatility_10d': volatility_10d.iloc[-1],
                        'volatility_30d': volatility_30d.iloc[-1],
                        'volatility_60d': volatility_60d.iloc[-1],
                        'rsi_14': rsi_14.iloc[-1],
                        'return_5d': history['Close'].pct_change(periods=5).iloc[-1] * 100,
                        'return_10d': history['Close'].pct_change(periods=10).iloc[-1] * 100,
                        'return_30d': history['Close'].pct_change(periods=30).iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded and calculated metrics for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

        self.data['world_indices'] = index_data
        return index_data

    def get_vix_data(self, period='3mo'):
        """
        Get VIX volatility index data

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '3mo'

        Returns:
        --------
        dict
            Dictionary of VIX data with technical indicators
        """

        try:
            vix = yf.Ticker('^VIX')
            history = self._get_history(vix, period)

            if not history.empty:
                # Calculate daily returns and moving averages
                daily_returns = history['Close'].pct_change()
                sma_5 = history['Close'].rolling(window=5).mean()
                sma_10 = history['Close'].rolling(window=10).mean()
                sma_20 = history['Close'].rolling(window=20).mean()

                # Calculate RSI
                delta = history['Close'].diff()
                up = delta.copy()
                up[up < 0] = 0
                down = -delta.copy()
                down[down < 0] = 0
                avg_gain_14 = up.rolling(window=14).mean()
                avg_loss_14 = down.rolling(window=14).mean()
                rs_14 = avg_gain_14 / avg_loss_14
                rsi_14 = 100 - (100 / (1 + rs_14))

                # Calculate changes over various periods
                vix_change_3d = history['Close'].pct_change(periods=3).iloc[-1] * 100
                vix_change_5d = history['Close'].pct_change(periods=5).iloc[-1] * 100
                vix_change_10d = history['Close'].pct_change(periods=10).iloc[-1] * 100
                vix_change_30d = history['Close'].pct_change(periods=30).iloc[-1] * 100

                # Calculate standard deviations
                vix_std_5d = history['Close'].rolling(window=5).std().iloc[-1]
                vix_std_10d = history['Close'].rolling(window=10).std().iloc[-1]
                vix_std_30d = history['Close'].rolling(window=30).std().iloc[-1]

                # Max and min over 30 days
                max_30d = history['Close'].rolling(window=30).max().iloc[-1]
                min_30d = history['Close'].rolling(window=30).min().iloc[-1]

                self.data['vix'] = {
                    'latest_value': history['Close'].iloc[-1],
                    'daily_change_pct': daily_returns.iloc[-1] * 100 if not pd.isna(daily_returns.iloc[-1]) else 0,
                    'sma_5': sma_5.iloc[-1],
                    'sma_10': sma_10.iloc[-1],
                    'sma_20': sma_20.iloc[-1],
                    'rsi_14': rsi_14.iloc[-1],
                    'change_3d': vix_change_3d,
                    'change_5d': vix_change_5d,
                    'change_10d': vix_change_10d,
                    'change_30d': vix_change_30d,
                    'std_5d': vix_std_5d,
                    'std_10d': vix_std_10d,
                    'std_30d': vix_std_30d,
                    'max_30d': max_30d,
                    'min_30d': min_30d,
                    'history': history['Close']
                }
                print("Downloaded and calculated VIX metrics")
                return self.data['vix']
        except Exception as e:
            print(f"Error processing VIX data: {e}")
            return None

    def get_major_equities(self, period='1mo', custom_tickers=None):
        """
        Get data for major individual stocks

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'
        custom_tickers : dict, optional
            Dictionary of name:ticker pairs to override default tickers

        Returns:
        --------
        dict
            Dictionary of equity data
        """

        equity_tickers = custom_tickers or {
            'IBM': 'IBM',
            'Apple': 'AAPL',
            'Amazon': 'AMZN',
            'General Electric': 'GE',
            'Microsoft': 'MSFT',
            'Nvidia': 'NVDA'
        }

        equity_data = {}

        for name, ticker in equity_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                history = self._get_history(stock, period)

                if not history.empty:
                    equity_data[name] = {
                        'latest_price': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")

        self.data['major_equities'] = equity_data
        return equity_data

    def get_macroeconomic_data(self, use_fred=False, fred_api_key=None, period='1mo'):
        """
        Get macroeconomic indicators

        Parameters:
        -----------
        use_fred : bool, optional
            Whether to use FRED API. Default is False
        fred_api_key : str, optional
            FRED API key. Required if use_fred is True
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        dict
            Dictionary of macroeconomic data
        """

        macro_data = {}

        if use_fred and fred_api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=fred_api_key)

                # Dictionary of FRED series IDs
                indicators = {
                    'Unemployment Rate': 'UNRATE',
                    'CPI Urban Consumers MoM': 'CPIAUCSL',
                    'Initial Jobless Claims': 'ICSA',
                    'GDP': 'GDP',
                    'Personal Income': 'PI',
                    'Personal Consumption Expenditures': 'PCE',
                    'Industrial Production': 'INDPRO'
                }

                # Convert date strings to datetime objects for FRED API
                observation_start = None
                if self.start_date:
                    observation_start = datetime.strptime(self.start_date, '%Y-%m-%d')

                observation_end = None
                if self.end_date:
                    observation_end = datetime.strptime(self.end_date, '%Y-%m-%d')

                for name, series_id in indicators.items():
                    try:
                        # Pass date range to FRED API if specified
                        if observation_start and observation_end:
                            data = fred.get_series(series_id,
                                                   observation_start=observation_start,
                                                   observation_end=observation_end)
                        elif observation_start:
                            data = fred.get_series(series_id, observation_start=observation_start)
                        else:
                            data = fred.get_series(series_id)

                        macro_data[name] = {
                            'latest_value': data.iloc[-1],
                            'previous_value': data.iloc[-2],
                            'percent_change': ((data.iloc[-1] / data.iloc[-2]) - 1) * 100,
                            'history': data
                        }
                        print(f"Downloaded {name} from FRED")
                    except Exception as e:
                        print(f"Error downloading {name} from FRED: {e}")
            except ImportError:
                print("FRED API package not installed. Install with: pip install fredapi")
        else:
            # Use Yahoo Finance for the Fed Funds Rate
            try:
                ff_rate = yf.Ticker('^IRX')  # 13-week Treasury bill rate as proxy
                history = self._get_history(ff_rate, period)

                if not history.empty:
                    macro_data['Fed Funds Proxy'] = {
                        'latest_rate': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print("Downloaded Fed Funds proxy data")
            except Exception as e:
                print(f"Error downloading Fed Funds proxy: {e}")

        self.data['macroeconomic'] = macro_data
        return macro_data

    def get_spx_sector_performance(self, period='1mo'):
        """
        Get S&P 500 sector ETF performance as proxy for sector data

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1mo'

        Returns:
        --------
        dict
            Dictionary of sector data
        """

        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }

        sector_data = {}

        for sector, ticker in sector_etfs.items():
            try:
                etf = yf.Ticker(ticker)
                history = self._get_history(etf, period)

                if not history.empty:
                    sector_data[sector] = {
                        'latest_price': history['Close'].iloc[-1],
                        'daily_change_pct': history['Close'].pct_change().iloc[-1] * 100,
                        'history': history['Close']
                    }
                    print(f"Downloaded {sector} sector data")
            except Exception as e:
                print(f"Error downloading {sector} data: {e}")

        self.data['spx_sectors'] = sector_data
        return sector_data

    def get_spx_technicals(self, period='1y'):
        """
        Get S&P 500 technical indicators

        Parameters:
        -----------
        period : str, optional
            Period to use if dates are not specified. Default is '1y'

        Returns:
        --------
        dict
            Dictionary of S&P 500 technical indicators
        """

        try:
            spx = yf.Ticker('^GSPC')
            history = self._get_history(spx, period)

            if not history.empty:
                # Calculate daily returns
                daily_returns = history['Close'].pct_change()

                # Calculate volatilities
                volatility_10d = daily_returns.rolling(window=10).std() * np.sqrt(252) * 100
                volatility_30d = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
                volatility_60d = daily_returns.rolling(window=60).std() * np.sqrt(252) * 100
                volatility_90d = daily_returns.rolling(window=90).std() * np.sqrt(252) * 100
                volatility_180d = daily_returns.rolling(window=180).std() * np.sqrt(252) * 100
                volatility_360d = daily_returns.rolling(window=252).std() * np.sqrt(252) * 100

                # Calculate moving averages
                sma_5 = history['Close'].rolling(window=5).mean()
                sma_10 = history['Close'].rolling(window=10).mean()
                sma_20 = history['Close'].rolling(window=20).mean()
                sma_50 = history['Close'].rolling(window=50).mean()
                sma_100 = history['Close'].rolling(window=100).mean()
                sma_200 = history['Close'].rolling(window=200).mean()

                # Calculate RSI
                delta = history['Close'].diff()
                up = delta.copy()
                up[up < 0] = 0
                down = -delta.copy()
                down[down < 0] = 0
                avg_gain_3 = up.rolling(window=3).mean()
                avg_loss_3 = down.rolling(window=3).mean()
                rs_3 = avg_gain_3 / avg_loss_3
                rsi_3 = 100 - (100 / (1 + rs_3))

                avg_gain_9 = up.rolling(window=9).mean()
                avg_loss_9 = down.rolling(window=9).mean()
                rs_9 = avg_gain_9 / avg_loss_9
                rsi_9 = 100 - (100 / (1 + rs_9))

                avg_gain_14 = up.rolling(window=14).mean()
                avg_loss_14 = down.rolling(window=14).mean()
                rs_14 = avg_gain_14 / avg_loss_14
                rsi_14 = 100 - (100 / (1 + rs_14))

                # Returns over various periods
                return_5d = history['Close'].pct_change(periods=5).iloc[-1] * 100
                return_10d = history['Close'].pct_change(periods=10).iloc[-1] * 100
                return_30d = history['Close'].pct_change(periods=30).iloc[-1] * 100
                return_60d = history['Close'].pct_change(periods=60).iloc[-1] * 100

                # 30-day high/low
                max_30d = history['Close'].rolling(window=30).max().iloc[-1]
                min_30d = history['Close'].rolling(window=30).min().iloc[-1]
                days_from_max = 0
                days_from_min = 0

                for i in range(1, 31):
                    if i <= len(history):
                        if history['Close'].iloc[-i] == max_30d:
                            days_from_max = i - 1
                            break

                for i in range(1, 31):
                    if i <= len(history):
                        if history['Close'].iloc[-i] == min_30d:
                            days_from_min = i - 1
                            break

                # Calculate differences from moving averages
                latest_price = history['Close'].iloc[-1]
                diff_5d = ((latest_price / sma_5.iloc[-1]) - 1) * 100
                diff_10d = ((latest_price / sma_10.iloc[-1]) - 1) * 100
                diff_20d = ((latest_price / sma_20.iloc[-1]) - 1) * 100
                diff_50d = ((latest_price / sma_50.iloc[-1]) - 1) * 100
                diff_200d = ((latest_price / sma_200.iloc[-1]) - 1) * 100

                self.data['spx_technicals'] = {
                    'latest_price': latest_price,
                    'daily_change_pct': daily_returns.iloc[-1] * 100 if not pd.isna(daily_returns.iloc[-1]) else 0,
                    'volatility_10d': volatility_10d.iloc[-1],
                    'volatility_30d': volatility_30d.iloc[-1],
                    'volatility_60d': volatility_60d.iloc[-1],
                    'volatility_90d': volatility_90d.iloc[-1],
                    'volatility_180d': volatility_180d.iloc[-1],
                    'volatility_360d': volatility_360d.iloc[-1],
                    'sma_5': sma_5.iloc[-1],
                    'sma_10': sma_10.iloc[-1],
                    'sma_20': sma_20.iloc[-1],
                    'sma_50': sma_50.iloc[-1],
                    'sma_100': sma_100.iloc[-1],
                    'sma_200': sma_200.iloc[-1],
                    'rsi_3': rsi_3.iloc[-1],
                    'rsi_9': rsi_9.iloc[-1],
                    'rsi_14': rsi_14.iloc[-1],
                    'return_5d': return_5d,
                    'return_10d': return_10d,
                    'return_30d': return_30d,
                    'return_60d': return_60d,
                    'max_30d': max_30d,
                    'min_30d': min_30d,
                    'days_from_max': days_from_max,
                    'days_from_min': days_from_min,
                    'diff_5d_ma': diff_5d,
                    'diff_10d_ma': diff_10d,
                    'diff_20d_ma': diff_20d,
                    'diff_50d_ma': diff_50d,
                    'diff_200d_ma': diff_200d,
                    'history': history['Close']
                }

                print("Downloaded and calculated S&P 500 technical indicators")
                return self.data['spx_technicals']
        except Exception as e:
            print(f"Error processing S&P 500 technicals: {e}")
            return None

    def collect_all_data(self, use_fred=False, fred_api_key=None, custom_equities=None):
        """
        Collect all available data

        Parameters:
        -----------
        use_fred : bool, optional
            Whether to use FRED API. Default is False
        fred_api_key : str, optional
            FRED API key. Required if use_fred is True
        custom_equities : dict, optional
            Dictionary of name:ticker pairs for custom equity selection

        Returns:
        --------
        dict
            Dictionary of all collected data
        """

        print("Starting comprehensive data collection...")
        print(f"Date range: {self.start_date or 'default'} to {self.end_date or 'current'}")

        # Collect all data categories
        self.get_government_bonds()
        self.get_commodities()
        self.get_currencies()
        self.get_world_indices()
        self.get_vix_data()
        self.get_major_equities(custom_tickers=custom_equities)
        self.get_macroeconomic_data(use_fred, fred_api_key)
        self.get_spx_sector_performance()
        self.get_spx_technicals()

        print("Data collection complete!")
        return self.data

    def export_to_csv(self, filename=None):
        """
        Export collected data to CSV

        Parameters:
        -----------
        filename : str, optional
            Filename for CSV export. Default is 'financial_data_{current_date}.csv'

        Returns:
        --------
        pandas.DataFrame
            Dataframe of exported data
        """

        if not self.data:
            print("No data to export. Run collect_all_data() first.")
            return

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'financial_data_{current_date}.csv'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Create flat dataframe from nested dictionaries
        flat_data = []

        for category, category_data in self.data.items():
            if isinstance(category_data, dict):
                for asset, metrics in category_data.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if not isinstance(value, pd.Series) and not isinstance(value, pd.DataFrame):
                                flat_data.append({
                                    'Category': category,
                                    'Asset': asset,
                                    'Metric': metric_name,
                                    'Value': value
                                })

        df = pd.DataFrame(flat_data)
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        return df

    def export_raw_history_to_csv(self, directory=None):
        """
        Export raw historical price data for each asset to individual CSV files

        Parameters:
        -----------
        directory : str, optional
            Directory to save CSV files. Default is 'history' subfolder in output_dir

        Returns:
        --------
        list
            List of saved filenames
        """

        if not self.data:
            print("No data to export. Run collect_all_data() first.")
            return []

        # Create history directory if not provided
        if directory is None:
            directory = os.path.join(self.output_dir, 'history')

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        saved_files = []

        # Export historical data for each asset
        for category, category_data in self.data.items():
            if isinstance(category_data, dict):
                for asset, metrics in category_data.items():
                    if isinstance(metrics, dict) and 'history' in metrics and isinstance(metrics['history'], pd.Series):
                        # Create safe filename from asset name
                        safe_name = asset.replace(' ', '_').replace('&', 'and').replace('/', '_')
                        filename = os.path.join(directory, f"{category}_{safe_name}_history.csv")

                        # Export history to CSV
                        metrics['history'].to_csv(filename)
                        saved_files.append(filename)
                        print(f"Exported {asset} history to {filename}")

        return saved_files

    def save_historical_data_to_excel(self, filename=None):
        """
        Save all historical data to a multi-sheet Excel file

        Parameters:
        -----------
        filename : str, optional
            Filename for Excel file. Default is 'historical_data_{current_date}.xlsx'

        Returns:
        --------
        str
            Path to saved Excel file
        """

        if not self.data:
            print("No data to export. Run collect_all_data() first.")
            return None

        try:
            import openpyxl
        except ImportError:
            print("openpyxl not installed. Install with: pip install openpyxl")
            return None

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'historical_data_{current_date}.xlsx'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Create Excel writer
        writer = pd.ExcelWriter(filename, engine='openpyxl')

        # Export data to different sheets based on category
        for category, category_data in self.data.items():
            if isinstance(category_data, dict):
                # Create a list to store all historical series for this category
                all_histories = {}

                for asset, metrics in category_data.items():
                    if isinstance(metrics, dict) and 'history' in metrics and isinstance(metrics['history'], pd.Series):
                        all_histories[asset] = metrics['history']

                if all_histories:
                    # Convert to DataFrame and export to Excel
                    df = pd.DataFrame(all_histories)
                    df.to_excel(writer, sheet_name=category[:31])  # Excel sheet names limited to 31 chars

        # Save Excel file
        writer.close()
        print(f"Historical data exported to Excel: {filename}")
        return filename