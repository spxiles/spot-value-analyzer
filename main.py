import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def load_parquet_files(parquets_dir="parquets"):
    """Load all parquet files from the specified directory."""
    parquet_files = glob.glob(f"{parquets_dir}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquets_dir} directory")

    dataframes = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dataframes.append(df)

    # Combine all dataframes
    return pd.concat(dataframes, ignore_index=True)


def parse_spot_values(spot_value_string):
    """Parse semicolon-separated spot values, converting underscores to NaN."""
    if pd.isna(spot_value_string):
        return []

    values = spot_value_string.split(";")
    parsed_values = []

    for val in values:
        if val in ("_", ""):
            parsed_values.append(np.nan)
        else:
            try:
                parsed_values.append(float(val))
            except ValueError:
                parsed_values.append(np.nan)

    return parsed_values


def calculate_pnl_from_spot_values(spot_values, entry_value, actual_profit):
    """Calculate P/L for each spot value relative to entry value."""
    # First, handle trailing blanks by replacing them with actual profit
    spot_values_copy = spot_values.copy()

    # Find the last non-blank value index
    last_valid_index = next(
        (i for i in range(len(spot_values_copy) - 1, -1, -1) if not pd.isna(spot_values_copy[i])), -1
    )

    # If there are trailing blanks, fill them with the profit-derived spot value
    if last_valid_index >= 0 and last_valid_index < len(spot_values_copy) - 1:
        # Calculate the spot value that would give us the actual profit
        # profit = entry_value + (spot_value * 100)
        # spot_value = (profit - entry_value) / 100
        exit_spot_value = (actual_profit - entry_value) / 100

        # Fill trailing blanks with this exit spot value
        for i in range(last_valid_index + 1, len(spot_values_copy)):
            spot_values_copy[i] = exit_spot_value

    # Always append the actual final profit as the last value to ensure
    # High/Low/Close calculations capture the true exit value
    spot_values_copy.append(actual_profit)

    # Now calculate P/L values
    pnl_values = []
    for i, val in enumerate(spot_values_copy):
        if pd.isna(val):
            pnl_values.append(np.nan)
        elif i == len(spot_values_copy) - 1:
            # Last value is already the final P/L
            pnl_values.append(round(val, 2))
        else:
            # For credit spreads: spot_values are already signed correctly
            # Negative spot_value means we owe money to close
            # P/L = Entry_Value + (spot_value * 100)
            current_market_impact = val * 100
            pnl = entry_value + current_market_impact
            pnl_values.append(round(pnl, 2))
    return pnl_values


def aggregate_spot_values_by_date(df):
    """Group by Entry_Date and aggregate P/L values by position."""
    # First, calculate P/L for each trade
    trade_data = []

    for _, row in df.iterrows():
        entry_date = row["Entry_Date"]
        entry_value = row["Entry_Value"]
        profit = row["Profit"]
        spot_values = parse_spot_values(row["Spot_Values"])

        # Calculate P/L for each spot value
        pnl_values = calculate_pnl_from_spot_values(spot_values, entry_value, profit)

        trade_data.append({"Entry_Date": entry_date, "pnl_values": pnl_values, "profit": profit})

    # Create a dataframe from trade data
    trade_df = pd.DataFrame(trade_data)

    # Determine the maximum length of P/L values arrays
    max_length = max(len(values) for values in trade_df["pnl_values"])

    # Group by Entry_Date and aggregate
    grouped = trade_df.groupby("Entry_Date")

    result_data = []
    for date, group in grouped:
        # Initialize aggregated P/L array with zeros
        aggregated_pnl = np.zeros(max_length)

        # Sum daily profit for cross-check
        daily_profit = group["profit"].sum()

        # Sum P/L values at each position across all trades for this date
        for pnl_values in group["pnl_values"]:
            for i, val in enumerate(pnl_values):
                if not pd.isna(val):
                    aggregated_pnl[i] += val

        # Convert back to semicolon-separated string with underscores for zero values
        result_string = ";".join(["_" if val == 0 else str(round(val, 2)) for val in aggregated_pnl])

        result_data.append({"Entry_Date": date, "Aggregated_PnL_Values": result_string, "Daily_Profit": daily_profit})

    return pd.DataFrame(result_data)


def create_expanded_dataframe(aggregated_df):
    """Create a dataframe with separate columns for each P/L position."""
    expanded_data = []

    for _, row in aggregated_df.iterrows():
        entry_date = row["Entry_Date"]
        pnl_values = parse_spot_values(row["Aggregated_PnL_Values"])

        # Filter out NaN values for High, Low, Close calculations
        if valid_values := [val for val in pnl_values if not pd.isna(val)]:
            high_value = round(max(valid_values), 2)
            low_value = round(min(valid_values), 2)
            close_value = round(valid_values[-1], 2)  # Last valid P/L value
        else:
            high_value = None
            low_value = None
            close_value = None

        row_data = {
            "Entry_Date": entry_date,
            "High": high_value,
            "Low": low_value,
            "Close": close_value,
        }

        for i, val in enumerate(pnl_values):
            if i == len(pnl_values) - 1:
                # Last value is the final exit P/L
                row_data["Final_Exit_PnL"] = None if pd.isna(val) else round(val, 2)
            else:
                row_data[f"PnL_5min_{i}"] = None if pd.isna(val) else round(val, 2)

        expanded_data.append(row_data)

    return pd.DataFrame(expanded_data)


def create_percentage_return_dataframe(dollar_df, account_value=100000):
    """Create a dataframe with percentage returns based on account value."""
    percent_df = dollar_df.copy()

    # Convert dollar P/L columns to percentage returns
    for col in percent_df.columns:
        if (col in ['High', 'Low', 'Close'] or col.startswith('PnL_5min_') or col == 'Final_Exit_PnL') and percent_df[col].dtype in ['float64', 'int64']:
            percent_df[col] = round((percent_df[col] / account_value) * 100, 4)

    return percent_df


def create_pnl_progression_chart(percent_df, output_dir):
    """Create a single chart with P/L progression lines for each trading day."""
    plt.figure(figsize=(12, 8))

    # Get all P/L columns (5-minute intervals)
    pnl_columns = [col for col in percent_df.columns if col.startswith('PnL_5min_')]

    # Plot each trading day as a separate line
    for _, row in percent_df.iterrows():
        date = row['Entry_Date']

        # Extract P/L values for this day (collect all valid values)
        pnl_values = []

        for col in pnl_columns:
            val = row[col]
            if pd.notna(val):
                pnl_values.append(val)

        # Add final exit value if we have it and it's different from the last value
        if pd.notna(row.get('Final_Exit_PnL')):
            final_exit = row['Final_Exit_PnL']
            if not pnl_values or pnl_values[-1] != final_exit:
                pnl_values.append(final_exit)

        # Only plot if we have at least 2 data points to make a line
        if len(pnl_values) >= 2:
            time_points = list(range(len(pnl_values)))
            plt.plot(time_points, pnl_values, marker='o', markersize=1, linewidth=1.5, label=date, alpha=0.8)
        elif len(pnl_values) == 1:
            # If only one point, plot as a dot
            plt.plot([0], pnl_values, marker='o', markersize=4, label=date, alpha=0.8)

    # Customize the chart
    plt.title('Daily P/L Progression - Percentage Returns', fontsize=14, fontweight='bold')
    plt.xlabel('5-Minute Intervals from Market Open', fontsize=12)
    plt.ylabel('P/L Percentage Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

    # Add legend (may be crowded with many days, so make it small)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the chart
    chart_path = output_dir / "pnl_progression_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory

    return chart_path


def create_daily_hlc_chart(percent_df, output_dir):
    """Create a High-Low-Close bar chart for each trading day (Open is always 0%)."""
    _, ax = plt.subplots(figsize=(14, 8))

    dates = percent_df['Entry_Date'].tolist()
    highs = percent_df['High'].tolist()
    lows = percent_df['Low'].tolist()
    closes = percent_df['Close'].tolist()

    x_positions = range(len(dates))

    # Create High-Low bars
    for i, (high, low, close) in enumerate(zip(highs, lows, closes)):
        # High-Low line (vertical line from low to high)
        ax.plot([i, i], [low, high], 'k-', linewidth=2, alpha=0.7)

        # Close mark (horizontal tick on the right)
        ax.plot([i + 0.1, i + 0.1], [close, close], 'r-', linewidth=3)

        # High and Low marks (small horizontal ticks)
        ax.plot([i - 0.05, i + 0.05], [high, high], 'k-', linewidth=2)
        ax.plot([i - 0.05, i + 0.05], [low, low], 'k-', linewidth=2)

    # Add zero line (Open is always 0%)
    ax.axhline(y=0, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='Open (0%)')

    # Customize the chart
    ax.set_title('Daily High-Low-Close (HLC) - Percentage Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trading Date', fontsize=12)
    ax.set_ylabel('P/L Percentage Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dates, rotation=45, ha='right')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Open (0%)'),
        Line2D([0], [0], color='black', linewidth=2, label='High-Low Range'),
        Line2D([0], [0], color='red', linewidth=3, label='Close')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the chart
    chart_path = output_dir / "daily_hlc_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    return chart_path


def create_output_folder():
    """Create a timestamped output folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    try:
        # Load all parquet files
        print("Loading parquet files...")
        df = load_parquet_files()
        print(f"Loaded {len(df)} records from parquet files")

        # Check if required columns exist
        required_columns = ["Entry_Date", "Spot_Values", "Entry_Value", "Profit"]
        if missing_columns := [col for col in required_columns if col not in df.columns]:
            raise ValueError(f"Required columns {missing_columns} not found in data")

        # Extract only the required columns
        df = df[["Entry_Date", "Spot_Values", "Entry_Value", "Profit"]].copy()

        # Aggregate P/L values by date
        print("Calculating P/L and aggregating by date...")
        aggregated_df = aggregate_spot_values_by_date(df)
        print(f"Aggregated to {len(aggregated_df)} unique dates")

        # Create expanded dataframe with separate columns
        print("Creating expanded dataframe...")
        expanded_df = create_expanded_dataframe(aggregated_df)

        # Create percentage return dataframe
        print("Creating percentage return dataframe...")
        percent_df = create_percentage_return_dataframe(expanded_df)

        # Create timestamped output folder
        print("Creating output folder...")
        output_dir = create_output_folder()
        print(f"Output folder created: {output_dir}")

        # Export results
        print("Exporting results...")

        # Export dollar P/L as parquet and CSV
        dollar_parquet_path = output_dir / "aggregated_pnl_values_dollars.parquet"
        expanded_df.to_parquet(dollar_parquet_path, index=False)
        print(f"Exported dollar P/L to {dollar_parquet_path}")

        dollar_csv_path = output_dir / "aggregated_pnl_values_dollars.csv"
        expanded_df.to_csv(dollar_csv_path, index=False)
        print(f"Exported dollar P/L to {dollar_csv_path}")

        # Export percentage return as parquet and CSV
        percent_parquet_path = output_dir / "aggregated_pnl_values_percent.parquet"
        percent_df.to_parquet(percent_parquet_path, index=False)
        print(f"Exported percentage returns to {percent_parquet_path}")

        percent_csv_path = output_dir / "aggregated_pnl_values_percent.csv"
        percent_df.to_csv(percent_csv_path, index=False)
        print(f"Exported percentage returns to {percent_csv_path}")

        # Generate P/L progression chart
        print("Generating P/L progression chart...")
        progression_chart_path = create_pnl_progression_chart(percent_df, output_dir)
        print(f"Exported P/L progression chart to {progression_chart_path}")

        # Generate daily HLC chart
        print("Generating daily High-Low-Close chart...")
        hlc_chart_path = create_daily_hlc_chart(percent_df, output_dir)
        print(f"Exported daily HLC chart to {hlc_chart_path}")

        print(f"Final dataset has {len(expanded_df)} rows and {len(expanded_df.columns)} columns")
        print("\nFirst few rows:")
        print(expanded_df.head())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
