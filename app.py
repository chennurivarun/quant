# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pytz

def validate_dates(start_date, end_date):
    """
    Validate date inputs
    """
    try:
        # Convert dates to datetime objects in UTC
        today = datetime.now(pytz.UTC).date()
        
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = st.date_input('📅 End Date', min_value=start_date, max_value=datetime.today().date())
            
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return False
            
        if end_date > today:
            st.error("End date cannot be in the future")
            return False
            
        if (end_date - start_date).days < 20:
            st.error("Date range must be at least 20 days for proper analysis")
            return False
            
        return True
    except Exception as e:
        st.error(f"Date validation error: {str(e)}")
        return False

def validate_ticker(ticker):
    """
    Validate ticker symbol
    """
    if not ticker:
        st.error("Please enter a ticker symbol")
        return False
        
    try:
        stock = yf.Ticker(ticker)
        # Try to get recent data to validate ticker
        test_data = stock.history(period='1d')
        if test_data.empty:
            st.error(f"No data available for ticker: {ticker}")
            return False
        return True
    except Exception as e:
        st.error(f"Invalid ticker symbol: {ticker}")
        return False

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance with error handling
    """
    try:
        # Convert dates to string format for yfinance
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            
        # Add buffer days for calculating moving averages
        buffer_start = start_date - timedelta(days=40)
        
        # Format dates as strings
        buffer_start_str = buffer_start.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=buffer_start_str, end=end_date_str)
        
        if df.empty:
            st.error("No data available for this ticker in the specified date range")
            return None
            
        # Convert index to datetime
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Trim to actual date range after calculations
        start_date_ts = pd.Timestamp(start_date)
        mask = df.index >= start_date_ts
        return df[mask]
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_breakouts(df, volume_threshold, price_threshold, holding_period):
    """
    Calculate breakout signals and returns with additional metrics
    """
    try:
        # Calculate technical indicators
        df['Avg_Volume_20d'] = df['Volume'].rolling(window=20).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Daily_Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Find breakout days
        volume_condition = df['Volume'] > (df['Avg_Volume_20d'] * (1 + volume_threshold/100))
        price_condition = df['Daily_Return'] > (price_threshold/100)
        
        breakout_days = df[volume_condition & price_condition].index
        
        results = []
        for day in breakout_days:
            entry_price = df.loc[day, 'Close']
            day_idx = df.index.get_loc(day)
            
            if day_idx + holding_period >= len(df):
                continue
                
            exit_day = df.index[day_idx + holding_period]
            exit_price = df.loc[exit_day, 'Close']
            
            # Calculate trade metrics
            holding_return = (exit_price - entry_price) / entry_price * 100
            max_price = df.loc[day:exit_day, 'High'].max()
            min_price = df.loc[day:exit_day, 'Low'].min()
            max_drawdown = ((max_price - min_price) / max_price) * 100
            
            results.append({
                'Entry_Date': day.strftime('%Y-%m-%d'),
                'Entry_Price': round(entry_price, 2),
                'Exit_Date': exit_day.strftime('%Y-%m-%d'),
                'Exit_Price': round(exit_price, 2),
                'Return_Percent': round(holding_return, 2),
                'Volume_Increase': round((df.loc[day, 'Volume'] / df.loc[day, 'Avg_Volume_20d'] - 1) * 100, 2),
                'Price_Change': round(df.loc[day, 'Daily_Return'] * 100, 2),
                'Max_Drawdown': round(max_drawdown, 2),
                'Volatility': round(df.loc[day, 'Daily_Volatility'] * 100, 2),
                'SMA_Distance': round((entry_price / df.loc[day, 'SMA_20'] - 1) * 100, 2)
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error in breakout calculation: {str(e)}")
        return pd.DataFrame()

def generate_summary_stats(results_df):
    """
    Generate enhanced summary statistics
    """
    if len(results_df) == 0:
        return pd.DataFrame()
        
    stats = {
        'Total Trades': len(results_df),
        'Winning Trades': len(results_df[results_df['Return_Percent'] > 0]),
        'Average Return': round(results_df['Return_Percent'].mean(), 2),
        'Median Return': round(results_df['Return_Percent'].median(), 2),
        'Best Trade': round(results_df['Return_Percent'].max(), 2),
        'Worst Trade': round(results_df['Return_Percent'].min(), 2),
        'Win Rate': round(len(results_df[results_df['Return_Percent'] > 0]) / len(results_df) * 100, 2),
        'Avg Max Drawdown': round(results_df['Max_Drawdown'].mean(), 2),
        'Avg Volume Increase': round(results_df['Volume_Increase'].mean(), 2),
        'Risk Adjusted Return': round(results_df['Return_Percent'].mean() / results_df['Return_Percent'].std(), 3)
    }
    
    return pd.DataFrame([stats])

def plot_analysis_charts(df, results_df, ticker):
    """
    Create enhanced visualization charts
    """
    if len(results_df) == 0:
        return
        
    # Create subplots
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('Cumulative Returns', 'Return Distribution', 'Trade Returns Over Time'),
                       vertical_spacing=0.15,
                       row_heights=[0.4, 0.3, 0.3])

    # 1. Cumulative Returns
    cumulative_returns = (1 + results_df['Return_Percent']/100).cumprod() - 1
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cumulative_returns))),
            y=cumulative_returns * 100,
            mode='lines+markers',
            name='Cumulative Returns',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # 2. Return Distribution
    fig.add_trace(
        go.Histogram(
            x=results_df['Return_Percent'],
            name='Return Distribution',
            nbinsx=20,
            marker_color='green'
        ),
        row=2, col=1
    )

    # 3. Trade Returns Over Time
    fig.add_trace(
        go.Bar(
            x=pd.to_datetime(results_df['Entry_Date']),
            y=results_df['Return_Percent'],
            name='Individual Trade Returns',
            marker_color='orange'
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Breakout Analysis for {ticker}",
        showlegend=True
    )

    # Update axes labels
    fig.update_xaxes(title_text="Trade Number", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=1)

    return fig

def main():
    st.title('Enhanced Stock Breakout Analysis Tool')
    
    st.markdown("""
    This tool analyzes stock breakouts based on volume and price changes. 
    It identifies potential breakout opportunities and calculates returns for a specified holding period.
    """)
    
    # Input form with validation
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input('Stock Ticker', 'AAPL').upper().strip()
        volume_threshold = st.number_input('Volume Breakout Threshold (%)', 
                                         min_value=100, 
                                         max_value=1000,
                                         value=200,
                                         help="Minimum percentage above 20-day average volume")
        holding_period = st.number_input('Holding Period (Days)', 
                                       min_value=1,
                                       max_value=100,
                                       value=10,
                                       help="Number of days to hold each position")
    
    with col2:
        start_date = st.date_input('Start Date', 
                                  datetime.now() - timedelta(days=365),
                                  help="Analysis start date")
        end_date = st.date_input('End Date', 
                                datetime.now(),
                                help="Analysis end date")
        price_threshold = st.number_input('Daily Change Threshold (%)',
                                        min_value=0.1,
                                        max_value=20.0,
                                        value=2.0,
                                        help="Minimum price increase on breakout day")
    
    if st.button('Generate Report'):
        # Input validation
        if not all([validate_dates(start_date, end_date),
                   validate_ticker(ticker)]):
            return
            
        with st.spinner('Analyzing data...'):
            try:
                # Fetch and process data
                df = get_stock_data(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    # Calculate breakouts and returns
                    results_df = calculate_breakouts(df, volume_threshold, price_threshold, holding_period)
                    
                    if len(results_df) > 0:
                        # Display summary statistics
                        st.subheader('Summary Statistics')
                        stats_df = generate_summary_stats(results_df)
                        st.dataframe(stats_df)
                        
                        # Create and display visualizations
                        try:
                            fig = plot_analysis_charts(df, results_df, ticker)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating visualizations: {str(e)}")
                        
                        # Display detailed results
                        st.subheader('Detailed Trade Results')
                        st.dataframe(results_df)
                        
                        # Create download buttons
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name=f"{ticker}_breakout_analysis.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning('No breakout signals found for the given parameters.')
                else:
                    st.error('Unable to fetch stock data. Please check the ticker and dates.')
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == '__main__':
    main()