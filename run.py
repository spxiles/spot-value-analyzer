import contextlib
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
NLV = 1792000  # Net Liquidation Value (starting account balance)


def convert_spot_values_to_profit_over_time(spot_values_str, entry_value, quantity, final_profit):
    if pd.isna(spot_values_str) or spot_values_str == "":
        return str(round(final_profit, 2))

    spot_values = spot_values_str.split(";")
    profit_over_time = []

    for spot_value in spot_values:
        if spot_value in ["_", ""]:
            profit_over_time.append("_")
        else:
            try:
                profit = entry_value + float(spot_value) * 100 * quantity
                profit_over_time.append(str(round(profit, 2)))
            except ValueError:
                profit_over_time.append("_")

    # Find trailing empty values and replace with final profit
    while profit_over_time and profit_over_time[-1] == "_":
        profit_over_time.pop()

    # Add final profit value to the end of all lists
    profit_over_time.append(str(round(final_profit, 2)))

    return ";".join(profit_over_time)


def load_vix_data():
    """Load VIX OHLC data from CSV"""
    vix_file = Path("input_vix_ohlc.csv")
    if not vix_file.exists():
        print("Warning: input_vix_ohlc.csv not found, VIX data will be empty")
        return pd.DataFrame()

    vix_df = pd.read_csv(vix_file)
    # Convert date format from YYYYMMDD to datetime
    vix_df["Date"] = pd.to_datetime(vix_df["Date"], format="%Y%m%d")
    # Rename columns to avoid confusion
    vix_df.rename(columns={"Date": "Entry_Date", "Open": "VIX_Open"}, inplace=True)
    # Keep only the columns we need
    vix_df = vix_df[["Entry_Date", "VIX_Open"]]
    return vix_df


def aggregate_profits_by_date(df):
    """Aggregate profits by Entry_Date"""
    print("Aggregating by Entry_Date...")

    # Load VIX data
    vix_df = load_vix_data()

    # Convert Entry_Date to datetime for sorting
    df_copy = df.copy()
    df_copy["Entry_Date"] = pd.to_datetime(df_copy["Entry_Date"])

    # Group by Entry_Date
    grouped = df_copy.groupby("Entry_Date")

    daily_data = []
    cumulative_profit = 0  # Track cumulative profit within the month
    current_month = None  # Track current month to detect month changes
    current_month_year = None

    for date, group in grouped:
        # Check if we've moved to a new month
        if current_month is None or date.month != current_month or date.year != current_month_year:
            # Reset cumulative profit at the start of each new month
            cumulative_profit = 0
            current_month = date.month
            current_month_year = date.year

        # Calculate starting balance (NLV + cumulative profit from current month only)
        starting_balance = NLV + cumulative_profit

        # Sum up the final profit for this day
        total_profit = group["Profit"].sum()

        # Get Custom_Column value (use first value since all should be the same)
        custom_column = group["Custom_Column"].iloc[0] if "Custom_Column" in group.columns else ""

        # For Profit_Over_Time, we need to sum each position
        profit_lists = [p.split(";") for p in group["Profit_Over_Time"]]

        # Find max length
        max_length = max(len(p) for p in profit_lists)

        # Sum up each position, using the last available value for shorter lists
        summed_profits = []
        profit_percentages = []
        profit_values = []  # Store numeric values for high/low/close calculation

        for i in range(max_length):
            position_sum = 0
            for profit_list in profit_lists:
                if i < len(profit_list) and profit_list[i] != "_":
                    with contextlib.suppress(ValueError):
                        position_sum += float(profit_list[i])
                elif i >= len(profit_list):
                    # Use the last value from this profit list (which is the final profit)
                    with contextlib.suppress(ValueError, IndexError):
                        position_sum += float(profit_list[-1])

            summed_profits.append(str(round(position_sum, 2)))
            profit_values.append(position_sum)

            # Calculate percentage for this position
            profit_pct = (position_sum / starting_balance) * 100 if starting_balance != 0 else 0
            profit_percentages.append(str(round(profit_pct, 4)))

        # Calculate high, low, close for dollar profits
        high_profit = max(profit_values, default=0)
        low_profit = min(profit_values, default=0)
        close_profit = profit_values[-1] if profit_values else 0  # Last value is the close

        # Calculate high, low, close for percentage profits
        high_percent = (high_profit / starting_balance) * 100 if starting_balance != 0 else 0
        low_percent = (low_profit / starting_balance) * 100 if starting_balance != 0 else 0
        close_percent = (close_profit / starting_balance) * 100 if starting_balance != 0 else 0

        # Update cumulative profit for next day
        cumulative_profit += total_profit
        ending_balance = starting_balance + total_profit

        # Calculate percentage profit for the day
        percent_profit = (total_profit / starting_balance) * 100 if starting_balance != 0 else 0

        daily_data.append({
            "Entry_Date": date,
            "Starting_Balance": round(starting_balance, 2),
            "Ending_Balance": round(ending_balance, 2),
            "Total_Profit": round(total_profit, 2),
            "Percent_Profit": round(percent_profit, 4),
            "High_Profit": round(high_profit, 2),
            "Low_Profit": round(low_profit, 2),
            "Close_Profit": round(close_profit, 2),
            "High_Percent": round(high_percent, 4),
            "Low_Percent": round(low_percent, 4),
            "Close_Percent": round(close_percent, 4),
            "Profit_Over_Time": ";".join(summed_profits),
            "Percent_Profit_Over_Time": ";".join(profit_percentages),
            "Trade_Count": len(group),
            "Custom_Column": custom_column,
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(daily_data)

    # Merge with VIX data if available
    if not vix_df.empty:
        result_df = result_df.merge(vix_df, on="Entry_Date", how="left")
        # Fill missing VIX values with NaN (or you could use 0 or forward fill)
        print("VIX data merged. Missing dates will have NaN for VIX_Open")
    else:
        result_df["VIX_Open"] = None

    return result_df


def create_percent_profit_chart(daily_df, run_folder):
    """Create a line chart of daily percent profit over time with average line"""
    print("Creating percent profit over time chart...")

    # Parse the Percent_Profit_Over_Time column
    all_percent_data = []
    max_length = 0

    for _, row in daily_df.iterrows():
        percent_str = row["Percent_Profit_Over_Time"]
        if pd.notna(percent_str):
            percent_values = [float(x) for x in percent_str.split(";") if x]
            all_percent_data.append(percent_values)
            max_length = max(max_length, len(percent_values))

    # Create figure with high resolution
    plt.figure(figsize=(20, 12), dpi=150)

    # Plot each day's percent profit over time with transparency
    for i, percent_values in enumerate(all_percent_data):
        x_values = list(range(len(percent_values)))
        plt.plot(x_values, percent_values, alpha=0.1, color='blue', linewidth=0.5)

    # Calculate average at each time increment
    averages = []
    for position in range(max_length):
        position_values = []
        for percent_list in all_percent_data:
            if position < len(percent_list):
                position_values.append(percent_list[position])
            elif len(percent_list) > 0:
                # Use the last value if this list is shorter
                position_values.append(percent_list[-1])

        if position_values:
            averages.append(np.mean(position_values))

    # Plot the average line in bold red
    if averages:
        x_avg = list(range(len(averages)))
        plt.plot(x_avg, averages, color='red', linewidth=3, label='Average', zorder=1000)

    # Formatting
    plt.xlabel('Time Interval', fontsize=12)
    plt.ylabel('Percent Profit (%)', fontsize=12)
    plt.title('Daily Percent Profit Over Time\n(All Days with Average)', fontsize=14, fontweight='bold')

    # Add more y-axis ticks
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()

    # Major ticks every 2.5%
    major_ticks = np.arange(np.floor(y_min/2.5)*2.5, np.ceil(y_max/2.5)*2.5 + 2.5, 2.5)
    # Minor ticks every 0.5%
    minor_ticks = np.arange(np.floor(y_min/0.5)*0.5, np.ceil(y_max/0.5)*0.5 + 0.5, 0.5)

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.1)

    plt.legend(loc='best', fontsize=10)

    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Save the chart
    chart_path = run_folder / "percent_profit_over_time_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved to: {chart_path}")
    return chart_path


def aggregate_profits_by_month(df):
    """Aggregate profits by Month-Year"""
    print("Aggregating by Month-Year...")

    # Convert Entry_Date to datetime if it's not already
    df_copy = df.copy()
    df_copy["Entry_Date"] = pd.to_datetime(df_copy["Entry_Date"])
    df_copy["Month_Year"] = df_copy["Entry_Date"].dt.to_period("M")

    # Sort by date to ensure proper cumulative calculations
    df_copy = df_copy.sort_values("Entry_Date")

    # Group by Month_Year
    grouped = df_copy.groupby("Month_Year")

    monthly_data = []

    for month_year, group in grouped:
        # Starting balance for the month (reset to NLV at start of each month + cumulative from previous months)
        # For monthly, we reset to NLV at the beginning of each month
        starting_balance = NLV

        # Sum up the final profit for this month
        total_profit = group["Profit"].sum()

        # Get Custom_Column value (use first value since all should be the same)
        custom_column = group["Custom_Column"].iloc[0] if "Custom_Column" in group.columns else ""

        # For Profit_Over_Time, we need to sum each position
        profit_lists = [p.split(";") for p in group["Profit_Over_Time"]]

        # Find max length
        max_length = max(len(p) for p in profit_lists)

        # Sum up each position, using the last available value for shorter lists
        summed_profits = []
        profit_percentages = []
        profit_values = []  # Store numeric values for high/low/close calculation

        for i in range(max_length):
            position_sum = 0
            for profit_list in profit_lists:
                if i < len(profit_list) and profit_list[i] != "_":
                    with contextlib.suppress(ValueError):
                        position_sum += float(profit_list[i])
                elif i >= len(profit_list):
                    # Use the last value from this profit list (which is the final profit)
                    with contextlib.suppress(ValueError, IndexError):
                        position_sum += float(profit_list[-1])

            summed_profits.append(str(round(position_sum, 2)))
            profit_values.append(position_sum)

            # Calculate percentage for this position
            profit_pct = (position_sum / starting_balance) * 100 if starting_balance != 0 else 0
            profit_percentages.append(str(round(profit_pct, 4)))

        # Calculate high, low, close for dollar profits
        high_profit = max(profit_values, default=0)
        low_profit = min(profit_values, default=0)
        close_profit = profit_values[-1] if profit_values else 0  # Last value is the close

        # Calculate high, low, close for percentage profits
        high_percent = (high_profit / starting_balance) * 100 if starting_balance != 0 else 0
        low_percent = (low_profit / starting_balance) * 100 if starting_balance != 0 else 0
        close_percent = (close_profit / starting_balance) * 100 if starting_balance != 0 else 0

        ending_balance = starting_balance + total_profit

        # Calculate percentage profit for the month
        percent_profit = (total_profit / starting_balance) * 100 if starting_balance != 0 else 0

        monthly_data.append({
            "Month_Year": str(month_year),
            "Starting_Balance": round(starting_balance, 2),
            "Ending_Balance": round(ending_balance, 2),
            "Total_Profit": round(total_profit, 2),
            "Percent_Profit": round(percent_profit, 4),
            "High_Profit": round(high_profit, 2),
            "Low_Profit": round(low_profit, 2),
            "Close_Profit": round(close_profit, 2),
            "High_Percent": round(high_percent, 4),
            "Low_Percent": round(low_percent, 4),
            "Close_Percent": round(close_percent, 4),
            "Profit_Over_Time": ";".join(summed_profits),
            "Percent_Profit_Over_Time": ";".join(profit_percentages),
            "Trade_Count": len(group),
            "Custom_Column": custom_column,
        })

    return pd.DataFrame(monthly_data)


def load_and_merge_parquets():
    parquets_dir = Path("parquets")
    outputs_dir = Path("outputs")

    outputs_dir.mkdir(exist_ok=True)

    # Create timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = outputs_dir / f"run_{timestamp}"
    run_folder.mkdir(exist_ok=True)

    parquet_files = list(parquets_dir.glob("*.parquet"))

    if not parquet_files:
        print("No parquet files found in /parquets directory")
        return

    print(f"Found {len(parquet_files)} parquet files to merge")

    dataframes = []

    for file_path in parquet_files:
        print(f"Loading: {file_path.name}")
        df = pd.read_parquet(file_path)

        df_filtered = df[
            [
                "Position_Name",
                "Entry_Date",
                "Entry_Value",
                "Exit_Value",
                "Spot_Values",
                "Quantity",
                "Profit",
                "Custom_Column",
            ]
        ].copy()

        dataframes.append(df_filtered)

    merged_df = pd.concat(dataframes, ignore_index=True)

    print("Converting Spot_Values to Profit_Over_Time...")
    merged_df["Profit_Over_Time"] = merged_df.apply(
        lambda row: convert_spot_values_to_profit_over_time(
            row["Spot_Values"], row["Entry_Value"], row["Quantity"], row["Profit"]
        ),
        axis=1,
    )

    print(f"Merged dataframe shape: {merged_df.shape}")
    print(f"Columns: {merged_df.columns.tolist()}")

    # Export all files to the timestamped run folder
    output_filename = "merged_data.parquet"
    output_path = run_folder / output_filename

    merged_df.to_parquet(output_path, index=False)

    print(f"Exported merged data to: {output_path}")
    print(f"Total records: {len(merged_df)}")

    # Create daily aggregation
    daily_df = aggregate_profits_by_date(merged_df)
    daily_output_filename = "daily_aggregated.parquet"
    daily_output_path = run_folder / daily_output_filename
    daily_df.to_parquet(daily_output_path, index=False)
    print(f"Exported daily aggregated data to: {daily_output_path}")
    print(f"Total days: {len(daily_df)}")

    # Export daily aggregation to CSV without over_time columns
    daily_csv_df = daily_df.drop(columns=["Profit_Over_Time", "Percent_Profit_Over_Time"], errors="ignore")
    daily_csv_filename = "daily_aggregated.csv"
    daily_csv_path = run_folder / daily_csv_filename
    daily_csv_df.to_csv(daily_csv_path, index=False)
    print(f"Exported daily aggregated CSV (no over_time lists) to: {daily_csv_path}")

    # Create monthly aggregation
    monthly_df = aggregate_profits_by_month(merged_df)
    monthly_output_filename = "monthly_aggregated.parquet"
    monthly_output_path = run_folder / monthly_output_filename
    monthly_df.to_parquet(monthly_output_path, index=False)
    print(f"Exported monthly aggregated data to: {monthly_output_path}")
    print(f"Total months: {len(monthly_df)}")

    # Export monthly aggregation to CSV without over_time columns
    monthly_csv_df = monthly_df.drop(columns=["Profit_Over_Time", "Percent_Profit_Over_Time"], errors="ignore")
    monthly_csv_filename = "monthly_aggregated.csv"
    monthly_csv_path = run_folder / monthly_csv_filename
    monthly_csv_df.to_csv(monthly_csv_path, index=False)
    print(f"Exported monthly aggregated CSV (no over_time lists) to: {monthly_csv_path}")

    # Create percent profit over time chart
    create_percent_profit_chart(daily_df, run_folder)

    print(f"\nAll files saved to: {run_folder}")


if __name__ == "__main__":
    load_and_merge_parquets()
