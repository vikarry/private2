#!/usr/bin/env python3
"""
Main execution script for the Financial Data Collection and Visualization System

This script demonstrates how to use the FinancialDataCollector and FinancialPlotter
classes to collect financial data and create comprehensive visualizations.
"""

from data_collector import FinancialDataCollector
from plotting import FinancialPlotter
from datetime import datetime, timedelta
import os


def main():
    """
    Main function to demonstrate the financial data collection and plotting system
    """

    print("=" * 60)
    print("Financial Data Collection and Visualization System")
    print("=" * 60)

    # Configuration
    # Set date range (example: last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create output directory
    output_dir = f'financial_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    print(f"Analysis period: {start_date_str} to {end_date_str}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Initialize the data collector
    print("Step 1: Initializing Financial Data Collector...")
    collector = FinancialDataCollector(
        start_date=start_date_str,
        end_date=end_date_str,
        output_dir=output_dir
    )

    # Step 2: Define custom equity selection (optional)
    custom_stocks = {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Tesla': 'TSLA',
        'Meta': 'META',
        'Netflix': 'NFLX',
        'Nvidia': 'NVDA'
    }

    # Step 3: Collect all financial data
    print("Step 2: Collecting comprehensive financial data...")
    print("This may take a few minutes...")

    try:
        all_data = collector.collect_all_data(
            use_fred=False,  # Set to True if you have a FRED API key
            fred_api_key=None,  # Add your FRED API key here if available
            custom_equities=custom_stocks
        )

        if not all_data:
            print("Error: No data was collected. Exiting...")
            return

        print(f"Successfully collected data for {len(all_data)} categories")
        print()

    except Exception as e:
        print(f"Error during data collection: {e}")
        return

    # Step 4: Export data to files
    print("Step 3: Exporting data to files...")

    try:
        # Export to CSV
        csv_file = collector.export_to_csv('comprehensive_financial_data.csv')
        print(f"✓ Exported summary data to CSV")

        # Export historical data to Excel
        excel_file = collector.save_historical_data_to_excel('historical_data.xlsx')
        if excel_file:
            print(f"✓ Exported historical data to Excel")

        # Export individual history files
        history_files = collector.export_raw_history_to_csv()
        print(f"✓ Exported {len(history_files)} individual history files")

    except Exception as e:
        print(f"Warning: Error during data export: {e}")

    print()

    # Step 5: Initialize the plotter and create visualizations
    print("Step 4: Creating comprehensive visualizations...")

    try:
        plotter = FinancialPlotter(collector)

        # Create main dashboard
        print("Creating main dashboard...")
        dashboard_file = plotter.create_dashboard()
        if dashboard_file:
            print(f"✓ Main dashboard: {os.path.basename(dashboard_file)}")

        # Create correlation heatmap
        print("Creating correlation analysis...")
        corr_file = plotter.create_correlation_heatmap()
        if corr_file:
            print(f"✓ Correlation heatmap: {os.path.basename(corr_file)}")

        # Create sector performance chart
        print("Creating sector analysis...")
        sector_file = plotter.create_sector_performance_chart()
        if sector_file:
            print(f"✓ Sector performance: {os.path.basename(sector_file)}")

        # Create volatility analysis
        print("Creating volatility analysis...")
        vol_file = plotter.create_volatility_chart()
        if vol_file:
            print(f"✓ Volatility analysis: {os.path.basename(vol_file)}")

        # Create technical analysis for S&P 500
        print("Creating technical analysis...")
        tech_file = plotter.create_technical_analysis_chart()
        if tech_file:
            print(f"✓ Technical analysis: {os.path.basename(tech_file)}")

        # Create individual time series plots
        print("Creating individual asset plots...")
        ts_files = plotter.create_time_series_plots()
        if ts_files:
            print(f"✓ Created {len(ts_files)} individual time series plots")

        print()

    except Exception as e:
        print(f"Error during visualization creation: {e}")
        return

    # Step 6: Create comprehensive report
    print("Step 5: Generating comprehensive report...")

    try:
        report_files = plotter.create_comprehensive_report('market_analysis_report')
        print(f"✓ Comprehensive report generated with {len(report_files)} files")

    except Exception as e:
        print(f"Warning: Error creating comprehensive report: {e}")

    # Step 7: Summary and recommendations
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"📁 All files saved to: {output_dir}")
    print()
    print("📊 Generated files include:")
    print("   • Main dashboard with key metrics")
    print("   • Correlation analysis heatmap")
    print("   • Sector performance charts")
    print("   • Volatility analysis")
    print("   • Technical analysis charts")
    print("   • Individual asset time series plots")
    print("   • CSV and Excel data exports")
    print()

    # Display some key insights if data is available
    display_key_insights(collector.data)

    print("🎉 Financial analysis complete!")
    print("Check the output directory for all generated files and visualizations.")


def display_key_insights(data):
    """
    Display key insights from the collected data

    Parameters:
    -----------
    data : dict
        Dictionary of collected financial data
    """

    print("KEY MARKET INSIGHTS:")
    print("-" * 40)

    # VIX insights
    if 'vix' in data and 'latest_value' in data['vix']:
        vix_value = data['vix']['latest_value']
        if vix_value < 20:
            vix_status = "Low (Market Complacency)"
        elif vix_value < 30:
            vix_status = "Moderate"
        else:
            vix_status = "High (Market Fear)"

        print(f"🔥 VIX (Fear Index): {vix_value:.1f} - {vix_status}")

    # Best and worst performing indices
    if 'world_indices' in data:
        performances = {}
        for name, metrics in data['world_indices'].items():
            if 'daily_change_pct' in metrics and not pd.isna(metrics['daily_change_pct']):
                performances[name] = metrics['daily_change_pct']

        if performances:
            best_performer = max(performances, key=performances.get)
            worst_performer = min(performances, key=performances.get)

            print(f"📈 Best Index: {best_performer} ({performances[best_performer]:+.2f}%)")
            print(f"📉 Worst Index: {worst_performer} ({performances[worst_performer]:+.2f}%)")

    # Bond yield insights
    if 'government_bonds' in data:
        for name, metrics in data['government_bonds'].items():
            if 'US Generic Govt 10 Yr' in name and 'latest_yield' in metrics:
                yield_10y = metrics['latest_yield']
                if yield_10y > 4.5:
                    yield_status = "High"
                elif yield_10y > 3.0:
                    yield_status = "Moderate"
                else:
                    yield_status = "Low"

                print(f"🏛️  10-Year Treasury: {yield_10y:.2f}% - {yield_status}")
                break

    # Top sector performance
    if 'spx_sectors' in data:
        sector_perf = {}
        for sector, metrics in data['spx_sectors'].items():
            if 'daily_change_pct' in metrics and not pd.isna(metrics['daily_change_pct']):
                sector_perf[sector] = metrics['daily_change_pct']

        if sector_perf:
            best_sector = max(sector_perf, key=sector_perf.get)
            worst_sector = min(sector_perf, key=sector_perf.get)

            print(f"🏆 Best Sector: {best_sector} ({sector_perf[best_sector]:+.2f}%)")
            print(f"💔 Worst Sector: {worst_sector} ({sector_perf[worst_sector]:+.2f}%)")

    print()


def quick_analysis():
    """
    Quick analysis function for faster execution with limited data
    """

    print("🚀 Quick Market Analysis Mode")
    print("=" * 40)

    # Initialize with shorter time period for faster execution
    collector = FinancialDataCollector(
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        output_dir='quick_analysis'
    )

    # Collect only essential data
    print("Collecting essential market data...")
    collector.get_world_indices(period='1mo')
    collector.get_vix_data(period='1mo')
    collector.get_government_bonds(period='1mo')

    # Create quick visualization
    plotter = FinancialPlotter(collector)
    dashboard_file = plotter.create_dashboard('quick_market_dashboard.png')

    if dashboard_file:
        print(f"✓ Quick dashboard created: {dashboard_file}")

    display_key_insights(collector.data)


if __name__ == "__main__":
    # Import pandas here to avoid issues if not available during import
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas")
        exit(1)

    print("Choose analysis mode:")
    print("1. Full comprehensive analysis (recommended)")
    print("2. Quick analysis (faster, limited data)")

    try:
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "2":
            quick_analysis()
        else:
            main()

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your internet connection and try again.")