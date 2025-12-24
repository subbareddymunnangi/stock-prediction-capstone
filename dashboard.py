import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import os

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import download_stock_data, create_features
from train_models import prepare_data, train_all_models, evaluate_models, save_models

# Page config
st.set_page_config(
    page_title="Comparative Analysis of Machine Learning Models on Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈Comparative Analysis of Machine Learning Models on Stock Price Prediction ")
st.markdown("### Comparative Analysis of 5 Machine Learning Models - Multi-Stock Support")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Multi-stock input
st.sidebar.subheader("📊 Stock Selection")
input_method = st.sidebar.radio(
    "Input Method:",
    ["Single Stock", "Multiple Stocks"]
)

if input_method == "Single Stock":
    ticker_input = st.sidebar.text_input("Stock Ticker Symbol", "AAPL", help="e.g., AAPL, GOOGL, MSFT, TSLA")
    tickers = [ticker_input.strip().upper()]
else:
    ticker_input = st.sidebar.text_area(
        "Enter Multiple Tickers (one per line or comma-separated)",
        "AAPL\nGOOGL\nMSFT\nTSLA\nAMZN",
        help="Enter stock tickers separated by commas or new lines"
    )
    # Parse input - handle both comma-separated and newline-separated
    if ',' in ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    else:
        tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

st.sidebar.markdown(f"**Selected Stocks:** {len(tickers)}")
st.sidebar.markdown("---")

# Date range
st.sidebar.subheader("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-01"))

st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Load Data & Train Models", type="primary")

# Popular stock suggestions
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Popular Stocks")
st.sidebar.markdown("""
**Technology:**
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- TSLA (Tesla)
- NVDA (Nvidia)

**Finance:**
- JPM (JP Morgan)
- BAC (Bank of America)
- GS (Goldman Sachs)

**Retail:**
- AMZN (Amazon)
- WMT (Walmart)
- TGT (Target)
""")

# Main content
if train_button:
    if len(tickers) == 0:
        st.error("❌ Please enter at least one stock ticker")
        st.stop()
    
    # Initialize storage for all results
    all_results = {}
    
    # Progress tracking
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            progress_text.markdown(f"### Processing {idx+1}/{len(tickers)}: {ticker}")
            
            # Download data
            with st.spinner(f"📥 Downloading {ticker} data..."):
                df = download_stock_data(ticker, start_date, end_date)
                
            if len(df) < 100:
                st.warning(f"⚠️ {ticker}: Not enough data. Skipping...")
                continue
                
            st.success(f"✅ {ticker}: Downloaded {len(df)} days of data")
            
            # Create features
            with st.spinner(f"🔧 {ticker}: Creating technical indicators..."):
                df = create_features(df)
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler = prepare_data(df)
            
            # Train models
            st.info(f"🤖 {ticker}: Training 5 models...")
            models = train_all_models(X_train, y_train)
            
            # Evaluate models
            results = evaluate_models(models, X_test, y_test)
            
            # Store results
            all_results[ticker] = {
                'results': results,
                'y_test': y_test,
                'models': models,
                'scaler': scaler,
                'data': df
            }
            
            # Update progress
            progress_bar.progress((idx + 1) / len(tickers))
            
        except Exception as e:
            st.error(f"❌ {ticker}: Error - {str(e)}")
            continue
    
    progress_text.empty()
    progress_bar.empty()
    
    if len(all_results) == 0:
        st.error("❌ No stocks were successfully processed")
        st.stop()
    
    # Save to session state
    st.session_state['all_results'] = all_results
    st.session_state['tickers'] = list(all_results.keys())
    
    st.success(f"✅ Successfully processed {len(all_results)} stocks!")

# Display results
if 'all_results' in st.session_state:
    all_results = st.session_state['all_results']
    tickers = st.session_state['tickers']
    
    st.markdown("---")
    st.header("📊 Results Overview")
    
    # Stock selector for detailed view
    if len(tickers) > 1:
        st.subheader("🔍 Select Stock for Detailed Analysis")
        selected_ticker = st.selectbox("Choose a stock:", tickers)
    else:
        selected_ticker = tickers[0]
    
    # Get selected stock data
    stock_data = all_results[selected_ticker]
    results = stock_data['results']
    y_test = stock_data['y_test']
    
    # Export All Results to Excel
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📥 Export Options")
    with col2:
        if st.button("📥 Export All to Excel", type="primary"):
            try:
                os.makedirs('data', exist_ok=True)
                output_file = f'data/multi_stock_prediction_results.xlsx'
                
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    # Sheet 1: Summary of all stocks
                    summary_data = []
                    for ticker, stock_data in all_results.items():
                        results = stock_data['results']
                        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
                        
                        summary_data.append({
                            'Stock': ticker,
                            'Best_Model': best_model[0],
                            'Best_RMSE': best_model[1]['rmse'],
                            'Best_MAE': best_model[1]['mae'],
                            'Best_R2': best_model[1]['r2'],
                            'Best_Direction_Accuracy': best_model[1]['directional_accuracy'],
                            'Avg_RMSE': np.mean([r['rmse'] for r in results.values()]),
                            'Avg_Training_Time': np.mean([r['train_time'] for r in results.values()])
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Individual sheets for each stock
                    for ticker, stock_data in all_results.items():
                        results = stock_data['results']
                        y_test = stock_data['y_test']
                        
                        # Performance metrics
                        metrics_data = [{
                            'Model': model_name,
                            'RMSE': data['rmse'],
                            'MAE': data['mae'],
                            'R2_Score': data['r2'],
                            'Directional_Accuracy': data['directional_accuracy'],
                            'Training_Time': data['train_time']
                        } for model_name, data in results.items()]
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        sheet_name = f"{ticker}_Metrics"[:31]  # Excel sheet name limit
                        metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Predictions
                        predictions_df = pd.DataFrame({'Actual_Price': y_test})
                        for model_name, data in results.items():
                            predictions_df[f'{model_name}_Pred'] = data['predictions']
                            predictions_df[f'{model_name}_Error'] = y_test - data['predictions']
                        
                        sheet_name = f"{ticker}_Predictions"[:31]
                        predictions_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Cross-stock comparison
                    comparison_data = []
                    for ticker, stock_data in all_results.items():
                        results = stock_data['results']
                        for model_name, data in results.items():
                            comparison_data.append({
                                'Stock': ticker,
                                'Model': model_name,
                                'RMSE': data['rmse'],
                                'MAE': data['mae'],
                                'R2': data['r2'],
                                'Direction_Accuracy': data['directional_accuracy']
                            })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df.to_excel(writer, sheet_name='Cross Comparison', index=False)
                
                st.success(f"✅ Results saved to: `{output_file}`")
                
                # Download button
                with open(output_file, 'rb') as f:
                    st.download_button(
                        label="💾 Download Excel File",
                        data=f,
                        file_name='multi_stock_prediction_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key='download_all'
                    )
            
            except Exception as e:
                st.error(f"❌ Error exporting: {str(e)}")
    
    # Comparison table for all stocks
    if len(tickers) > 1:
        st.markdown("---")
        st.subheader("📊 All Stocks Comparison")
        
        comparison_data = []
        for ticker, stock_data in all_results.items():
            results = stock_data['results']
            best_model = min(results.items(), key=lambda x: x[1]['rmse'])
            
            comparison_data.append({
                'Stock': ticker,
                'Best Model': best_model[0],
                'RMSE ($)': f"{best_model[1]['rmse']:.2f}",
                'MAE ($)': f"{best_model[1]['mae']:.2f}",
                'R² Score': f"{best_model[1]['r2']:.4f}",
                'Direction (%)': f"{best_model[1]['directional_accuracy']:.1f}",
                'Avg Training Time (s)': f"{np.mean([r['train_time'] for r in results.values()]):.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Best performing stocks
        st.markdown("### 🏆 Best Performing Stocks")
        col1, col2, col3 = st.columns(3)
        
        # Find best stocks by different metrics
        all_best = []
        for ticker, stock_data in all_results.items():
            results = stock_data['results']
            best_model = min(results.items(), key=lambda x: x[1]['rmse'])
            all_best.append((ticker, best_model[1]))
        
        most_accurate = min(all_best, key=lambda x: x[1]['rmse'])
        best_r2 = max(all_best, key=lambda x: x[1]['r2'])
        best_direction = max(all_best, key=lambda x: x[1]['directional_accuracy'])
        
        with col1:
            st.metric("Most Accurate", most_accurate[0], f"RMSE: ${most_accurate[1]['rmse']:.2f}")
        
        with col2:
            st.metric("Best R² Score", best_r2[0], f"R²: {best_r2[1]['r2']:.4f}")
        
        with col3:
            st.metric("Best Direction", best_direction[0], f"{best_direction[1]['directional_accuracy']:.1f}%")
    
    # Detailed analysis for selected stock
    st.markdown("---")
    st.header(f"📈 Detailed Analysis: {selected_ticker}")
    
    # Metrics table
    st.subheader("Model Performance")
    metrics_data = []
    for model_name, data in results.items():
        metrics_data.append({
            'Model': model_name,
            'RMSE ($)': f"{data['rmse']:.2f}",
            'MAE ($)': f"{data['mae']:.2f}",
            'R² Score': f"{data['r2']:.4f}",
            'Direction Accuracy (%)': f"{data['directional_accuracy']:.1f}",
            'Training Time (s)': f"{data['train_time']:.2f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Export single stock
    if st.button(f"📥 Export {selected_ticker} Results"):
        try:
            os.makedirs('data', exist_ok=True)
            output_file = f'data/{selected_ticker}_prediction_results.xlsx'
            
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                # Performance metrics
                metrics_export = pd.DataFrame([{
                    'Model': model_name,
                    'RMSE': data['rmse'],
                    'MAE': data['mae'],
                    'R2_Score': data['r2'],
                    'Directional_Accuracy': data['directional_accuracy'],
                    'Training_Time': data['train_time']
                } for model_name, data in results.items()])
                metrics_export.to_excel(writer, sheet_name='Performance', index=False)
                
                # Predictions
                predictions_df = pd.DataFrame({'Actual_Price': y_test})
                for model_name, data in results.items():
                    predictions_df[f'{model_name}_Prediction'] = data['predictions']
                    predictions_df[f'{model_name}_Error'] = y_test - data['predictions']
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Best model
                best_accuracy = min(results.items(), key=lambda x: x[1]['rmse'])
                best_speed = min(results.items(), key=lambda x: x[1]['train_time'])
                best_direction = max(results.items(), key=lambda x: x[1]['directional_accuracy'])
                
                best_df = pd.DataFrame({
                    'Category': ['Most Accurate', 'Fastest', 'Best Direction'],
                    'Model': [best_accuracy[0], best_speed[0], best_direction[0]],
                    'Value': [
                        f"{best_accuracy[1]['rmse']:.2f}",
                        f"{best_speed[1]['train_time']:.2f}",
                        f"{best_direction[1]['directional_accuracy']:.1f}"
                    ]
                })
                best_df.to_excel(writer, sheet_name='Best Models', index=False)
            
            st.success(f"✅ Saved to: `{output_file}`")
            
            with open(output_file, 'rb') as f:
                st.download_button(
                    label="💾 Download",
                    data=f,
                    file_name=f'{selected_ticker}_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key=f'download_{selected_ticker}'
                )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Best models for selected stock
    st.markdown("### 🏆 Best Performers")
    col1, col2, col3 = st.columns(3)
    
    best_accuracy = min(results.items(), key=lambda x: x[1]['rmse'])
    best_speed = min(results.items(), key=lambda x: x[1]['train_time'])
    best_direction = max(results.items(), key=lambda x: x[1]['directional_accuracy'])
    
    with col1:
        st.metric("Most Accurate", best_accuracy[0], f"${best_accuracy[1]['rmse']:.2f}")
    
    with col2:
        st.metric("Fastest", best_speed[0], f"{best_speed[1]['train_time']:.2f}s")
    
    with col3:
        st.metric("Best Direction", best_direction[0], f"{best_direction[1]['directional_accuracy']:.1f}%")
    
    # Visualization
    st.markdown("---")
    st.subheader("📈 Predictions vs Actual Price")
    
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(go.Scatter(
        y=y_test[:100],
        mode='lines',
        name='Actual Price',
        line=dict(color='black', width=3)
    ))
    
    # Predictions
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for idx, (model_name, data) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            y=data['predictions'][:100],
            mode='lines',
            name=model_name,
            line=dict(color=colors[idx], width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=f"{selected_ticker} - Predictions vs Actual (First 100 Test Days)",
        xaxis_title="Days",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution
    st.markdown("---")
    st.subheader("📉 Error Distribution")
    
    fig2 = go.Figure()
    
    for model_name, data in results.items():
        errors = y_test - data['predictions']
        fig2.add_trace(go.Box(
            y=errors,
            name=model_name,
            boxmean='sd'
        ))
    
    fig2.update_layout(
        title=f"{selected_ticker} - Prediction Errors by Model",
        yaxis_title="Error ($)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    ## Welcome! 👋
    
    This application compares 5 machine learning algorithms for stock price prediction:
    
    1. **Linear Regression** - Simple, fast baseline
    2. **Decision Tree** - Captures non-linear patterns
    3. **Random Forest** - Ensemble of decision trees
    4. **Support Vector Machine (SVM)** - Handles complex relationships
    5. **Neural Network** - Deep learning approach
    
    ### ✨ New Features:
    
    - **Multi-Stock Analysis**: Analyze multiple stocks at once
    - **Excel Export**: Export all results to comprehensive Excel files
    - **Cross-Stock Comparison**: Compare performance across different stocks
    - **Individual Reports**: Generate detailed reports for each stock
    
    ### How to Use:
    
    1. 👈 Select input method in sidebar:
       - **Single Stock**: Analyze one stock
       - **Multiple Stocks**: Analyze multiple stocks simultaneously
    
    2. 📝 Enter stock ticker(s):
       - Single: Type one ticker (e.g., AAPL)
       - Multiple: Enter multiple tickers (one per line or comma-separated)
    
    3. 📅 Select date range (at least 2 years recommended)
    
    4. 🚀 Click "Load Data & Train Models"
    
    5. ⏳ Wait for training (time depends on number of stocks)
    
    6. 📊 View results and export to Excel
    
    ### Example Multi-Stock Input:
```
    AAPL
    GOOGL
    MSFT
    TSLA
    AMZN
```
    
    Or comma-separated: `AAPL, GOOGL, MSFT, TSLA, AMZN`
    """)
    
    st.info("💡 Tip: Start with 2-3 stocks to see how it works, then expand to more stocks")
