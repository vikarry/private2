"""
Simple Yearly Seasonality Analysis for Prepayment CPR
Focus on yearly patterns and overall data analysis only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def simple_yearly_analysis(results_data):
    """
    Simple analysis of yearly prepayment patterns

    Parameters:
    results_data: DataFrame from AGGPREPAY function with CPR data

    Returns:
    Dictionary with yearly analysis results
    """

    # Prepare data
    if isinstance(results_data, dict) and 'agg' in results_data:
        data = results_data['agg'].copy()
    else:
        data = results_data.copy()

    # Reset index if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        if 'COB Date' in data.columns:
            data['date'] = pd.to_datetime(data['COB Date'])
        else:
            data['date'] = pd.to_datetime(data.index)
    else:
        data['date'] = data.index

    # Add year column
    data['year'] = data['date'].dt.year

    # Yearly statistics
    yearly_stats = data.groupby('year').agg({
        'annual CPR': ['mean', 'std', 'min', 'max', 'count'],
        'exposure': ['mean', 'sum'] if 'exposure' in data.columns else ['count'],
        'prepay': ['sum'] if 'prepay' in data.columns else ['count']
    }).round(4)

    # Flatten column names
    yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]

    # Calculate year-over-year changes
    yearly_stats['yoy_cpr_change'] = yearly_stats['annual CPR_mean'].pct_change()
    yearly_stats['yoy_exposure_change'] = yearly_stats.get('exposure_mean', yearly_stats.iloc[:, 0]).pct_change()

    # Overall statistics
    overall_stats = {
        'total_observations': len(data),
        'date_range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
        'overall_mean_cpr': data['annual CPR'].mean(),
        'overall_std_cpr': data['annual CPR'].std(),
        'overall_min_cpr': data['annual CPR'].min(),
        'overall_max_cpr': data['annual CPR'].max(),
        'years_covered': len(data['year'].unique()),
        'avg_observations_per_year': len(data) / len(data['year'].unique())
    }

    # Year-to-year volatility
    if len(yearly_stats) > 1:
        cpr_volatility = yearly_stats['annual CPR_mean'].std()
        overall_stats['yearly_cpr_volatility'] = cpr_volatility
        overall_stats['coefficient_of_variation'] = cpr_volatility / yearly_stats['annual CPR_mean'].mean()

    return {
        'yearly_stats': yearly_stats,
        'overall_stats': overall_stats,
        'raw_data': data
    }


def plot_yearly_trends(analysis_results):
    """Create simple plots for yearly trends"""

    yearly_stats = analysis_results['yearly_stats']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Yearly Prepayment Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Yearly CPR trends
    axes[0, 0].plot(yearly_stats.index, yearly_stats['annual CPR_mean'],
                    marker='o', linewidth=2, markersize=8, color='blue')
    axes[0, 0].fill_between(yearly_stats.index,
                            yearly_stats['annual CPR_mean'] - yearly_stats['annual CPR_std'],
                            yearly_stats['annual CPR_mean'] + yearly_stats['annual CPR_std'],
                            alpha=0.3, color='blue')
    axes[0, 0].set_title('Average CPR by Year (with ±1 Std Dev)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Annual CPR')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Year-over-Year changes
    yoy_changes = yearly_stats['yoy_cpr_change'].dropna() * 100
    colors = ['green' if x < 0 else 'red' for x in yoy_changes]
    axes[0, 1].bar(yoy_changes.index, yoy_changes.values, color=colors, alpha=0.7)
    axes[0, 1].set_title('Year-over-Year CPR Change (%)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('YoY Change (%)')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: CPR Range by year (min/max)
    axes[1, 0].fill_between(yearly_stats.index,
                            yearly_stats['annual CPR_min'],
                            yearly_stats['annual CPR_max'],
                            alpha=0.5, color='orange', label='Min-Max Range')
    axes[1, 0].plot(yearly_stats.index, yearly_stats['annual CPR_mean'],
                    marker='o', color='red', linewidth=2, label='Mean')
    axes[1, 0].set_title('CPR Range by Year')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Annual CPR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Data coverage
    axes[1, 1].bar(yearly_stats.index, yearly_stats['annual CPR_count'],
                   color='purple', alpha=0.7)
    axes[1, 1].set_title('Number of Observations by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Number of Observations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_all_segments(prepay_results):
    """
    Analyze all segments in your prepayment results

    Parameters:
    prepay_results: Your main prepay dictionary from the model

    Returns:
    Dictionary with analysis for each segment
    """

    all_analysis = {}

    # Extract all segments
    for facility in prepay_results.keys():
        if isinstance(prepay_results[facility], dict):
            for product in prepay_results[facility].keys():
                if 'result' in prepay_results[facility][product]:
                    segment_name = f"{facility}_{product}"
                    segment_data = prepay_results[facility][product]['result']

                    try:
                        analysis = simple_yearly_analysis(segment_data)
                        all_analysis[segment_name] = analysis

                        print(f"\n=== {segment_name} ===")
                        print(f"Period: {analysis['overall_stats']['date_range']}")
                        print(f"Overall CPR: {analysis['overall_stats']['overall_mean_cpr']:.2%}")
                        print(f"CPR Volatility: {analysis['overall_stats']['overall_std_cpr']:.2%}")
                        print(f"Years covered: {analysis['overall_stats']['years_covered']}")

                        # Show yearly trends
                        yearly = analysis['yearly_stats']
                        if len(yearly) > 1:
                            best_year = yearly['annual CPR_mean'].idxmin()
                            worst_year = yearly['annual CPR_mean'].idxmax()
                            print(f"Lowest CPR year: {best_year} ({yearly.loc[best_year, 'annual CPR_mean']:.2%})")
                            print(f"Highest CPR year: {worst_year} ({yearly.loc[worst_year, 'annual CPR_mean']:.2%})")

                    except Exception as e:
                        print(f"Error analyzing {segment_name}: {e}")

    return all_analysis


def create_summary_table(all_analysis):
    """Create a summary table of all segments"""

    summary_data = []

    for segment_name, analysis in all_analysis.items():
        stats = analysis['overall_stats']
        yearly_stats = analysis['yearly_stats']

        # Calculate trend if multiple years
        trend = "Stable"
        if len(yearly_stats) > 1:
            first_year_cpr = yearly_stats['annual CPR_mean'].iloc[0]
            last_year_cpr = yearly_stats['annual CPR_mean'].iloc[-1]
            change = (last_year_cpr - first_year_cpr) / first_year_cpr

            if change > 0.1:
                trend = "Increasing"
            elif change < -0.1:
                trend = "Decreasing"
            else:
                trend = "Stable"

        summary_data.append({
            'Segment': segment_name,
            'Overall_CPR': f"{stats['overall_mean_cpr']:.2%}",
            'CPR_Volatility': f"{stats['overall_std_cpr']:.2%}",
            'Years_Covered': stats['years_covered'],
            'Total_Observations': stats['total_observations'],
            'Trend': trend,
            'Min_CPR': f"{stats['overall_min_cpr']:.2%}",
            'Max_CPR': f"{stats['overall_max_cpr']:.2%}"
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def simple_seasonality_integration(prepay_results):
    """
    Simple integration with your existing prepayment model
    Just add this to your main script after AGGPREPAY calls
    """

    print("\n" + "=" * 60)
    print("YEARLY SEASONALITY ANALYSIS")
    print("=" * 60)

    # Analyze all segments
    all_analysis = analyze_all_segments(prepay_results)

    # Create summary
    summary_table = create_summary_table(all_analysis)
    print("\n=== SUMMARY TABLE ===")
    print(summary_table.to_string(index=False))

    # Create plots for each segment
    for segment_name, analysis in all_analysis.items():
        if len(analysis['yearly_stats']) > 1:  # Only plot if multiple years
            fig = plot_yearly_trends(analysis)
            fig.suptitle(f'Yearly Analysis - {segment_name}', fontsize=16)
            plt.show()

    # Overall portfolio analysis (combine all segments)
    print("\n=== PORTFOLIO LEVEL ANALYSIS ===")
    all_data = []
    for analysis in all_analysis.values():
        all_data.append(analysis['raw_data'])

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        portfolio_analysis = simple_yearly_analysis(combined_data)

        print(f"Portfolio CPR: {portfolio_analysis['overall_stats']['overall_mean_cpr']:.2%}")
        print(f"Portfolio Volatility: {portfolio_analysis['overall_stats']['overall_std_cpr']:.2%}")

        # Portfolio yearly plot
        if len(portfolio_analysis['yearly_stats']) > 1:
            fig = plot_yearly_trends(portfolio_analysis)
            fig.suptitle('Portfolio Level - Yearly Analysis', fontsize=16)
            plt.show()

    return all_analysis, summary_table


# Simple usage example
if __name__ == "__main__":
    # After your existing model runs, just add:
    # all_analysis, summary = simple_seasonality_integration(prepay)

    print("Simple Yearly Seasonality Analysis")
    print("=" * 50)
    print("\nTo use with your existing model:")
    print("1. Run your existing prepayment model")
    print("2. Call: simple_seasonality_integration(prepay)")
    print("3. Review yearly trends and summary table")
    print("\nThis will show:")
    print("- Year-over-year CPR changes")
    print("- Overall statistics by segment")
    print("- Simple trend analysis")
    print("- Portfolio-level summary")