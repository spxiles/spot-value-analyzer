# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spot Value Analyzer is a financial trading analysis tool that processes options trading data from parquet files. The tool aggregates profit/loss data by date and creates detailed P/L timelines showing 5-minute intervals throughout trading days.

## Core Architecture

### Data Processing Pipeline
1. **Input**: Parquet files in `/parquets/` directory containing trading data with columns:
   - `Entry_Date`, `Entry_Value`, `Profit`, `Spot_Values` (semicolon-separated list)
   - `Position_Type` (currently focused on CallCreditSpread)

2. **Processing**:
   - Parse semicolon-separated `Spot_Values` (underscores represent missing data)
   - Calculate P/L for each 5-minute interval: `P/L = Entry_Value + (spot_value * 100)`
   - Handle trailing blanks by filling with actual exit profit values
   - Append final profit as last value to ensure accurate High/Low/Close calculations

3. **Aggregation**: Group trades by `Entry_Date` and sum P/L values by position index

4. **Output**: Timestamped folders in `/outputs/` containing CSV and Parquet files

### Key Functions
- `calculate_pnl_from_spot_values()`: Core P/L calculation with trailing blank handling
- `aggregate_spot_values_by_date()`: Daily aggregation logic
- `create_expanded_dataframe()`: Converts to final output format with High/Low/Close

## Commands

### Running the Analysis
```bash
uv run python main.py
```

### Development Environment
- Uses `uv` for dependency management
- Python 3.11+ required
- Main dependencies: pandas, numpy, pyarrow

### Dependencies
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name
```

## Data Schema

### Input Schema (Parquet)
- Credit spread positions with entry/exit values
- `Spot_Values`: Semicolon-separated list where negative values indicate money owed to close position

### Output Schema (CSV/Parquet)
- `Entry_Date`: Trading date
- `High`/`Low`/`Close`: P/L statistics including final exit value
- `PnL_5min_N`: P/L at each 5-minute interval
- `Final_Exit_PnL`: Actual final exit P/L (ensures accurate High/Low calculations)

## Important Implementation Details

### P/L Calculation Logic
- For credit spreads: P/L = Entry_Value + (spot_value * 100)
- Spot values can be negative (representing cost to close position)
- Final profit is always appended as last value to capture true exit P/L

### Stop Loss Handling
- Trades with early exits have trailing blanks in `Spot_Values`
- These are filled with calculated exit spot value: `(actual_profit - entry_value) / 100`
- Ensures complete P/L timeline from entry to actual exit

### Output Organization
- All results saved to timestamped folders: `outputs/run_YYYYMMDD_HHMMSS/`
- Both CSV and Parquet formats exported
- Prevents overwriting previous analysis runs