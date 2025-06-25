import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class FinancialPlotter:
    """
    A comprehensive plotting class for financial market data visualization
    """

    def __init__(self, data_collector, output_dir=None):
        """
        Initialize the plotter with a data collector instance

        Parameters:
        -----------
        data_collector : FinancialDataCollector
            Instance of FinancialDataCollector with collected data
        output_dir : str, optional
            Directory to save plot files. Default uses data_collector's output_dir
        """
        self.data_collector = data_collector
        self.data = data_collector.data
        self.output_dir = output_dir or data_collector.output_dir

        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_dashboard(self, filename=None, figsize=(15, 10)):
        """
        Create a comprehensive dashboard of key metrics

        Parameters:
        -----------
        filename : str, optional
            Filename for dashboard image. Default is 'market_dashboard_{current_date}.png'
        figsize : tuple, optional
            Figure size. Default is (15, 10)

        Returns:
        --------
        str
            Path to saved dashboard image
        """

        if not self.data:
            print("No data to visualize. Run collect_all_data() first.")
            return None

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'market_dashboard_{current_date}.png'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Financial Markets Dashboard', fontsize=16, fontweight='bold')

        # Plot 1: Major indices performance
        if 'world_indices' in self.data:
            ax1 = axes[0, 0]
            performances = []
            names = []

            for name, data in self.data['world_indices'].items():
                if 'daily_change_pct' in data and not pd.isna(data['daily_change_pct']):
                    performances.append(data['daily_change_pct'])
                    names.append(name)

            if performances:
                colors = ['green' if x > 0 else 'red' for x in performances]
                bars = ax1.bar(names, performances, color=colors, alpha=0.7)
                ax1.set_title('Daily Change % - Major Indices', fontweight='bold')
                ax1.set_ylabel('Change (%)')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, val in zip(bars, performances):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height > 0 else -0.1),
                             f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 2: Government bond yields
        if 'government_bonds' in self.data:
            ax2 = axes[0, 1]
            yields = []
            names = []

            for name, data in self.data['government_bonds'].items():
                if 'latest_yield' in data and not pd.isna(data['latest_yield']):
                    yields.append(data['latest_yield'])
                    names.append(name.replace('US Generic Govt ', '').replace(' Yr', 'Y'))

            if yields:
                bars = ax2.bar(names, yields, color='steelblue', alpha=0.7)
                ax2.set_title('Current Government Bond Yields', fontweight='bold')
                ax2.set_ylabel('Yield (%)')
                ax2.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, val in zip(bars, yields):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                             f'{val:.2f}%', ha='center', va='bottom')

        # Plot 3: Commodity performance
        if 'commodities' in self.data:
            ax3 = axes[1, 0]
            performances = []
            names = []

            for name, data in self.data['commodities'].items():
                if 'daily_change_pct' in data and not pd.isna(data['daily_change_pct']):
                    performances.append(data['daily_change_pct'])
                    names.append(name)

            if performances:
                colors = ['green' if x > 0 else 'red' for x in performances]
                bars = ax3.bar(names, performances, color=colors, alpha=0.7)
                ax3.set_title('Daily Change % - Commodities', fontweight='bold')
                ax3.set_ylabel('Change (%)')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, val in zip(bars, performances):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height > 0 else -0.1),
                             f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 4: Currency performance vs USD
        if 'currencies' in self.data:
            ax4 = axes[1, 1]
            performances = []
            names = []

            for name, data in self.data['currencies'].items():
                if 'daily_change_pct' in data and not pd.isna(data['daily_change_pct']):
                    performances.append(data['daily_change_pct'])
                    names.append(name)

            if performances:
                colors = ['green' if x > 0 else 'red' for x in performances]
                bars = ax4.bar(names, performances, color=colors, alpha=0.7)
                ax4.set_title('Daily Change % - Currencies vs USD', fontweight='bold')
                ax4.set_ylabel('Change (%)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, val in zip(bars, performances):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2., height + (0.05 if height > 0 else -0.05),
                             f'{val:.2f}%', ha='center', va='bottom' if height > 0 else 'top')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dashboard created and saved as '{filename}'")
        return filename

    def create_time_series_plots(self, directory=None, figsize=(12, 6)):
        """
        Create individual time series plots for each asset's history

        Parameters:
        -----------
        directory : str, optional
            Directory to save plot images. Default is 'plots' subfolder in output_dir
        figsize : tuple, optional
            Figure size for each plot. Default is (12, 6)

        Returns:
        --------
        list
            List of saved filenames
        """

        if not self.data:
            print("No data to visualize. Run collect_all_data() first.")
            return []

        # Create plots directory if not provided
        if directory is None:
            directory = os.path.join(self.output_dir, 'plots')

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        saved_files = []

        # Generate time series plots for each asset
        for category, category_data in self.data.items():
            if isinstance(category_data, dict):
                for asset, metrics in category_data.items():
                    if isinstance(metrics, dict) and 'history' in metrics and isinstance(metrics['history'], pd.Series):
                        # Create safe filename from asset name
                        safe_name = asset.replace(' ', '_').replace('&', 'and').replace('/', '_')
                        filename = os.path.join(directory, f"{category}_{safe_name}_plot.png")

                        # Create and save time series plot
                        plt.figure(figsize=figsize)

                        # Plot the main time series
                        plt.plot(metrics['history'].index, metrics['history'].values,
                                 linewidth=2, label=asset)

                        # Add moving averages if available
                        if 'sma_20' in metrics and not pd.isna(metrics['sma_20']):
                            # Calculate 20-day moving average for the plot
                            ma_20 = metrics['history'].rolling(window=20).mean()
                            plt.plot(ma_20.index, ma_20.values, '--', alpha=0.7, label='20-day MA')

                        plt.title(f"{asset} - {category.replace('_', ' ').title()}",
                                  fontsize=14, fontweight='bold')
                        plt.ylabel("Value")
                        plt.xlabel("Date")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close()

                        saved_files.append(filename)
                        print(f"Created plot for {asset} at {filename}")

        return saved_files

    def create_correlation_heatmap(self, filename=None, figsize=(14, 12)):
        """
        Create a correlation heatmap of asset prices

        Parameters:
        -----------
        filename : str, optional
            Filename for heatmap image. Default is 'correlation_heatmap_{current_date}.png'
        figsize : tuple, optional
            Figure size. Default is (14, 12)

        Returns:
        --------
        str
            Path to saved heatmap image
        """

        if not self.data:
            print("No data to visualize. Run collect_all_data() first.")
            return None

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'correlation_heatmap_{current_date}.png'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Collect all historical price series
        price_series = {}
        for category, category_data in self.data.items():
            if isinstance(category_data, dict):
                for asset, metrics in category_data.items():
                    if isinstance(metrics, dict) and 'history' in metrics and isinstance(metrics['history'], pd.Series):
                        # Use shortened names for better display
                        display_name = f"{category[:3].upper()}_{asset[:15]}"
                        price_series[display_name] = metrics['history']

        if not price_series:
            print("No historical price data available for correlation analysis.")
            return None

        # Create a dataframe with all price series
        df = pd.DataFrame(price_series)

        # Calculate correlation matrix
        correlation = df.corr(method='pearson')

        # Create heatmap
        plt.figure(figsize=figsize)

        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(correlation, dtype=bool))

        # Generate heatmap
        sns.heatmap(correlation, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')

        plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Correlation heatmap saved as '{filename}'")
        return filename

    def create_sector_performance_chart(self, filename=None, figsize=(12, 8)):
        """
        Create a sector performance chart

        Parameters:
        -----------
        filename : str, optional
            Filename for chart image. Default is 'sector_performance_{current_date}.png'
        figsize : tuple, optional
            Figure size. Default is (12, 8)

        Returns:
        --------
        str
            Path to saved chart image
        """

        if 'spx_sectors' not in self.data:
            print("No sector data available. Run get_spx_sector_performance() first.")
            return None

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'sector_performance_{current_date}.png'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Extract sector performance data
        sectors = []
        performances = []

        for sector, data in self.data['spx_sectors'].items():
            if 'daily_change_pct' in data and not pd.isna(data['daily_change_pct']):
                sectors.append(sector)
                performances.append(data['daily_change_pct'])

        if not performances:
            print("No valid sector performance data available.")
            return None

        # Create horizontal bar chart
        plt.figure(figsize=figsize)
        colors = ['green' if x > 0 else 'red' for x in performances]
        bars = plt.barh(sectors, performances, color=colors, alpha=0.7)

        plt.title('S&P 500 Sector Performance (Daily Change %)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Daily Change (%)')
        plt.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, performances)):
            plt.text(val + (0.05 if val > 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}%', ha='left' if val > 0 else 'right', va='center')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sector performance chart saved as '{filename}'")
        return filename

    def create_volatility_chart(self, filename=None, figsize=(12, 8)):
        """
        Create a volatility comparison chart

        Parameters:
        -----------
        filename : str, optional
            Filename for chart image. Default is 'volatility_chart_{current_date}.png'
        figsize : tuple, optional
            Figure size. Default is (12, 8)

        Returns:
        --------
        str
            Path to saved chart image
        """

        # Create default filename with current date if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'volatility_chart_{current_date}.png'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Collect volatility data from various sources
        volatility_data = {}

        # Get VIX data
        if 'vix' in self.data and 'latest_value' in self.data['vix']:
            volatility_data['VIX'] = self.data['vix']['latest_value']

        # Get volatility from world indices
        if 'world_indices' in self.data:
            for name, data in self.data['world_indices'].items():
                if 'volatility_30d' in data and not pd.isna(data['volatility_30d']):
                    volatility_data[name] = data['volatility_30d']

        # Get S&P 500 specific volatilities if available
        if 'spx_technicals' in self.data:
            spx_data = self.data['spx_technicals']
            periods = ['10d', '30d', '60d', '90d']
            spx_vols = {}
            for period in periods:
                vol_key = f'volatility_{period}'
                if vol_key in spx_data and not pd.isna(spx_data[vol_key]):
                    spx_vols[f'SPX {period}'] = spx_data[vol_key]

        if not volatility_data and not spx_vols:
            print("No volatility data available.")
            return None

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Market Volatility Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Current volatility across indices
        if volatility_data:
            ax1 = axes[0]
            names = list(volatility_data.keys())
            values = list(volatility_data.values())

            bars = ax1.bar(names, values, color='orange', alpha=0.7)
            ax1.set_title('Current 30-Day Volatility', fontweight='bold')
            ax1.set_ylabel('Volatility (%)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{val:.1f}%', ha='center', va='bottom')

        # Plot 2: S&P 500 volatility across different time periods
        if spx_vols:
            ax2 = axes[1]
            periods = list(spx_vols.keys())
            values = list(spx_vols.values())

            bars = ax2.bar(periods, values, color='steelblue', alpha=0.7)
            ax2.set_title('S&P 500 Volatility by Period', fontweight='bold')
            ax2.set_ylabel('Volatility (%)')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                         f'{val:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Volatility chart saved as '{filename}'")
        return filename

    def create_technical_analysis_chart(self, asset_name=None, category=None, filename=None, figsize=(14, 10)):
        """
        Create a technical analysis chart with price, moving averages, and RSI

        Parameters:
        -----------
        asset_name : str, optional
            Name of the asset to plot. Default is S&P 500
        category : str, optional
            Category of the asset. Default is 'spx_technicals'
        filename : str, optional
            Filename for chart image. Default is 'technical_analysis_{asset}_{current_date}.png'
        figsize : tuple, optional
            Figure size. Default is (14, 10)

        Returns:
        --------
        str
            Path to saved chart image
        """

        # Default to S&P 500 if no asset specified
        if asset_name is None and category is None:
            if 'spx_technicals' in self.data:
                data_source = self.data['spx_technicals']
                asset_name = 'S&P 500'
                category = 'spx_technicals'
            else:
                print("No S&P 500 technical data available and no asset specified.")
                return None
        else:
            # Find the specified asset
            if category and category in self.data and asset_name in self.data[category]:
                data_source = self.data[category][asset_name]
            else:
                print(f"Asset {asset_name} not found in category {category}")
                return None

        if 'history' not in data_source:
            print(f"No historical data available for {asset_name}")
            return None

        # Create default filename if not provided
        if filename is None:
            current_date = datetime.now().strftime('%Y%m%d')
            safe_name = asset_name.replace(' ', '_').replace('&', 'and')
            filename = f'technical_analysis_{safe_name}_{current_date}.png'

        # If path is relative, prepend output directory
        if not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)

        # Get price history
        prices = data_source['history']

        # Calculate technical indicators
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()

        # Calculate RSI
        delta = prices.diff()
        up = delta.copy()
        up[up < 0] = 0
        down = -delta.copy()
        down[down < 0] = 0
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        fig.suptitle(f'Technical Analysis - {asset_name}', fontsize=16, fontweight='bold')

        # Plot 1: Price and moving averages
        ax1.plot(prices.index, prices.values, label='Price', linewidth=2)
        ax1.plot(sma_20.index, sma_20.values, label='20-day SMA', alpha=0.7)
        ax1.plot(sma_50.index, sma_50.values, label='50-day SMA', alpha=0.7)

        ax1.set_title('Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: RSI
        ax2.plot(rsi.index, rsi.values, color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(rsi.index, 30, 70, alpha=0.1, color='gray')

        ax2.set_title('RSI (14-day)')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Technical analysis chart saved as '{filename}'")
        return filename

    def create_comprehensive_report(self, filename=None):
        """
        Create a comprehensive visual report with all available charts

        Parameters:
        -----------
        filename : str, optional
            Base filename for report files. Default uses current date

        Returns:
        --------
        list
            List of all created chart filenames
        """

        if not self.data:
            print("No data to visualize. Run collect_all_data() first.")
            return []

        current_date = datetime.now().strftime('%Y%m%d')
        base_name = filename or f'financial_report_{current_date}'

        created_files = []

        print("Creating comprehensive financial report...")

        # Create main dashboard
        dashboard_file = self.create_dashboard(f'{base_name}_dashboard.png')
        if dashboard_file:
            created_files.append(dashboard_file)

        # Create correlation heatmap
        corr_file = self.create_correlation_heatmap(f'{base_name}_correlation.png')
        if corr_file:
            created_files.append(corr_file)

        # Create sector performance chart
        sector_file = self.create_sector_performance_chart(f'{base_name}_sectors.png')
        if sector_file:
            created_files.append(sector_file)

        # Create volatility chart
        vol_file = self.create_volatility_chart(f'{base_name}_volatility.png')
        if vol_file:
            created_files.append(vol_file)

        # Create technical analysis chart
        tech_file = self.create_technical_analysis_chart(filename=f'{base_name}_technical.png')
        if tech_file:
            created_files.append(tech_file)

        # Create time series plots directory
        ts_files = self.create_time_series_plots(
            directory=os.path.join(self.output_dir, f'{base_name}_timeseries')
        )
        created_files.extend(ts_files)

        print(f"Comprehensive report created! {len(created_files)} files generated.")
        return created_files