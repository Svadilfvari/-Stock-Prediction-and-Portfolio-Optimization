# import numpy as np
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
# import os
# from pypfopt import expected_returns, risk_models
# from pypfopt.efficient_frontier import EfficientFrontier 

# def preprocess_stock_data(stock_data):
#     """
#     Preprocesses stock data by handling missing values and removing outliers.
    
#     Args:
#         stock_data (DataFrame): DataFrame containing stock data with at least a 'Close' column.

#     Returns:
#         DataFrame: Processed stock data with missing values handled and outliers removed.
#     """
#     stock_data.ffill(inplace=True)  # Forward fill missing values with the previous valid observation
#     stock_data.bfill(inplace=True)  # Backward fill missing values with the next valid observation
#     # Remove outliers that are more than 3 standard deviations away from the mean
#     stock_data = stock_data[(np.abs(stock_data - stock_data.mean()) <= (3 * stock_data.std())).all(axis=1)]
#     return stock_data  # Return the cleaned data

# def create_features(stock_data):
#     """
#     Creates additional features from stock data for model training.

#     Args:
#         stock_data (DataFrame): DataFrame containing stock data with at least a 'Close' column.

#     Returns:
#         DataFrame: Stock data with new features added.
#     """
#     stock_data = stock_data.copy()  # Create a copy of the DataFrame to avoid modifying the original
#     stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
#     stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
#     stock_data['Lag_1'] = stock_data['Close'].shift(1)  # Previous day's closing price
#     stock_data.dropna(inplace=True)  # Drop rows with any missing values after feature creation
#     return stock_data  # Return the DataFrame with new features

# def fetch_and_train_multiple_models(ticker_symbol, start_date, end_date):
#     """
#     Fetches stock data, preprocesses it, trains multiple models, and evaluates performance.
    
#     Args:
#         ticker_symbol (str): Ticker symbol of the stock.
#         start_date (str): Start date for fetching data.
#         end_date (str): End date for fetching data.

#     Returns:
#         tuple: Contains risk assessment, model performances, actual and predicted values, 
#                cleaned weights from portfolio optimization.
#     """
#     # Download historical stock data from Yahoo Finance
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = preprocess_stock_data(stock_data)  # Preprocess the data
#     stock_data = create_features(stock_data)  # Create additional features for modeling

#     # Define features and target variable
#     X = stock_data[['SMA_20', 'SMA_50', 'Lag_1']]  # Features for the model
#     y = stock_data['Close']  # Target variable (actual stock prices)

#     # Split the data into training and testing sets (70-30 split)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     # Standardize the features using StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
#     X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler

#     # Initialize dictionaries to store risk assessment, model performance metrics, and predictions
#     risk_assessment = {}
#     models_performance = {}
#     y_preds = {}

#     # Calculate daily returns of the stock
#     returns = stock_data['Close'].pct_change().dropna()
#     std_dev = returns.std()  # Standard deviation of returns (volatility)

#     VaR = np.percentile(returns, 5)  # 5th percentile value for VaR calculation

#     # Conditional VaR (CVaR), average of returns below the VaR threshold
#     CVaR = returns[returns <= VaR].mean()

#     # Store risk assessment metrics
#     risk_assessment['Standard Deviation'] = std_dev
#     risk_assessment['Value at Risk'] = VaR
#     risk_assessment['Conditional Value at Risk'] = CVaR

#     # Linear Regression Model
#     lr_model = LinearRegression()
#     lr_model.fit(X_train_scaled, y_train)  # Train the model
#     y_pred_lr = lr_model.predict(X_test_scaled)  # Predict test data
#     # Evaluate performance metrics
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     rmse_lr = np.sqrt(mse_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)
#     # Store performance metrics
#     models_performance['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr}
#     y_preds['Linear Regression'] = y_pred_lr  # Store predictions

#     # Random Forest Regressor Model
#     rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25)
#     rf_model.fit(X_train, y_train)  # Train the model
#     y_pred_rf = rf_model.predict(X_test)  # Predict test data
#     # Evaluate performance metrics
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     rmse_rf = np.sqrt(mse_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     # Store performance metrics
#     models_performance['Random Forest'] = {'mse': mse_rf, 'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
#     y_preds['Random Forest'] = y_pred_rf  # Store predictions

#     # MinMaxScaler for LSTM input scaling
#     min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#     X_scaled = min_max_scaler.fit_transform(X)  # Scale the features
#     y_scaled = min_max_scaler.fit_transform(y.values.reshape(-1, 1))  # Scale the target

#     # Reshape features for LSTM model input
#     X_train_lstm = np.reshape(X_scaled[:len(X_train)], (len(X_train), 1, X_train.shape[1]))
#     X_test_lstm = np.reshape(X_scaled[len(X_train):], (len(X_test), 1, X_test.shape[1]))
#     y_train_lstm = y_scaled[:len(X_train)]

#     # LSTM Model
#     lstm_model = Sequential()
#     lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])))  # First LSTM layer
#     lstm_model.add(LSTM(units=50))  # Second LSTM layer
#     lstm_model.add(Dense(1))  # Output layer
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
#     lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)  # Train the model
    
#     # Predict with LSTM and inverse transform the scaled data
#     y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
#     y_pred_lstm = min_max_scaler.inverse_transform(y_pred_lstm_scaled)
#     # Evaluate performance metrics
#     mse_lstm = mean_squared_error(y_test, y_pred_lstm)
#     mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
#     rmse_lstm = np.sqrt(mse_lstm)
#     r2_lstm = r2_score(y_test, y_pred_lstm)
#     # Store performance metrics
#     models_performance['LSTM'] = {'mse': mse_lstm, 'mae': mae_lstm, 'rmse': rmse_lstm, 'r2': r2_lstm}
#     y_preds['LSTM'] = y_pred_lstm.flatten()  # Store predictions

#     # Calculate expected returns and covariance of returns, these will be used for the asset allocation 
#     mu = expected_returns.mean_historical_return(stock_data['Close'])
#     S = risk_models.sample_cov(stock_data['Close'])
   
#     return risk_assessment, models_performance, y_test, y_preds, mu, S 

# def get_earliest_trading_date(ticker_symbol):
#     """
#     Fetches the earliest available trading date for a given stock ticker.

#     Args:
#         ticker_symbol (str): Ticker symbol of the stock.

#     Returns:
#         str: The earliest trading date in 'YYYY-MM-DD' format.
#     """
#     stock_data = yf.download(ticker_symbol, start='1900-01-01', end='2023-01-01')  # Download data from a very early date
#     return stock_data.index.min().strftime('%Y-%m-%d')  # Return the earliest date in string format

# # List of stock tickers to analyze
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# # Create a folder for saving figures if it doesn't exist
# figures_folder = 'figures'
# os.makedirs(figures_folder, exist_ok=True)  # Create directory if it does not exist

# # Store expected returns and covariance matrices
# all_expected_returns = []
# all_covariances = []

# for ticker_symbol in tickers:
#     end_date = '2023-01-01'  # Define the end date for the data
#     start_date = get_earliest_trading_date(ticker_symbol) # Define the start date for the data
#     print(f"Start Date for {ticker_symbol}: {start_date}")

#     # Fetch data, train models, and evaluate
#     risk_assessment, models_performance, y_test, y_preds, mu, S = fetch_and_train_multiple_models(ticker_symbol, start_date, end_date)
    
#     # Collecting expected returns and covariances for each stock
#     all_expected_returns.append(mu)
#     all_covariances.append(S)

#     # Print model performance metrics
#     for model_name, metrics in models_performance.items():
#         print(f"{model_name} Metrics for {ticker_symbol}:")
#         print(f"  MSE: {metrics['mse']:.4f}")
#         print(f"  MAE: {metrics['mae']:.4f}")
#         print(f"  RMSE: {metrics['rmse']:.4f}")
#         print(f"  R-squared: {metrics['r2']:.4f}")

#     # Print risk assessment metrics
#     print("##### Risk Assessment #####")
#     print(f"Standard Deviation (Volatility): {risk_assessment['Standard Deviation']}")
#     print(f"Value at Risk (VaR): {risk_assessment['Value at Risk']}")
#     print(f"Conditional Value at Risk (CVaR): {risk_assessment['Conditional Value at Risk']}")

#     # Extract the display period for plotting
#     display_start_date = y_test.index.min().strftime('%Y-%m-%d')
#     display_end_date = y_test.index.max().strftime('%Y-%m-%d')

#     # Plot actual vs predicted stock prices
#     plt.figure(figsize=(14, 7))
#     plt.plot(y_test.index, y_test.values, label='Actual', color='black')  # Actual stock prices
#     for model_name, y_pred in y_preds.items():
#         plt.plot(y_test.index, y_pred, label=model_name)  # Predicted prices for each model

#     plt.title(f'Stock Price Prediction for {ticker_symbol} from {display_start_date} to {display_end_date}')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.grid(True)

#     # Save the prediction figure
#     filename = f'stock_price_prediction_{ticker_symbol}_{display_start_date}_{display_end_date}.png'
#     filepath = os.path.join(figures_folder, filename)
#     plt.savefig(filepath)
#     plt.close()  # Close the figure to free memory

#     # Plot model performance metrics
#     metrics_labels = ['MSE', 'MAE', 'RMSE', 'R-squared']
#     models = list(models_performance.keys())
#     metrics_values = np.array([[metrics['mse'], metrics['mae'], metrics['rmse'], metrics['r2']] for metrics in models_performance.values()])

#     fig, ax = plt.subplots(figsize=(14, 7))
#     width = 0.2  # Width of bars
#     x = np.arange(len(metrics_labels))  # The label locations

#     # Clip metrics for visualization if necessary
#     metrics_percentage_clipped = np.clip(metrics_values, a_min=0, a_max=100)

#     for i, model_name in enumerate(models):
#         ax.bar(x + i * width, metrics_percentage_clipped[i], width=width, label=model_name)
#     ax.set_xlabel('Metrics')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Model Performance Metrics for {ticker_symbol} from {display_start_date} to {display_end_date}')
#     ax.set_xticks(x + width * (len(models) / 2) - width / 2)
#     ax.set_xticklabels(metrics_labels)
#     ax.legend()

#     # Save the metrics figure
#     metrics_filename = f'model_performance_metrics_{ticker_symbol}_{display_start_date}_{display_end_date}.png'
#     metrics_filepath = os.path.join(figures_folder, metrics_filename)
#     fig.savefig(metrics_filepath)
#     plt.close()  # Close the figure to free memory

#     # Plot risk metrics
#     fig, ax = plt.subplots(figsize=(10, 5))
#     risk_metrics = list(risk_assessment.keys())
#     risk_values = list(risk_assessment.values())
#     ax.bar(risk_metrics, risk_values, color=['blue', 'red', 'green'])
#     ax.set_xlabel('Risk Metrics')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Risk Metrics for {ticker_symbol}')
#     plt.grid(True)

#     # Save the risk metrics figure
#     risk_filename = f'risk_metrics_{ticker_symbol}.png'
#     risk_filepath = os.path.join(figures_folder, risk_filename)
#     plt.savefig(risk_filepath)
#     plt.close()  # Close the figure to free memory

# # Combine all expected returns and covariances for portfolio optimization
# # combined_mu = np.mean(all_expected_returns)
# # combined_S = np.cov(all_covariances)

# # # Perform portfolio optimization
# # ef = EfficientFrontier(combined_mu, combined_S)
# # weights = ef.max_sharpe()  # Optimize for maximum Sharpe ratio
# # cleaned_weights = ef.clean_weights()  # Clean up weights
# # performance = ef.portfolio_performance(verbose=True)  # Get performance metrics

# # # Plot portfolio optimization and allocation
# # plt.figure(figsize=(10, 5))
# # plt.bar(cleaned_weights.keys(), cleaned_weights.values())
# # plt.xlabel('Assets')
# # plt.ylabel('Proportion of Portfolio')
# # plt.title('Optimal Portfolio Allocation')
# # plt.grid(True)
# # portfolio_filename = 'optimal_portfolio_allocation.png'
# # portfolio_filepath = os.path.join(figures_folder, portfolio_filename)
# # plt.savefig(portfolio_filepath)
# # plt.close()

################################
# import numpy as np
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
# import os
# from pypfopt import expected_returns, risk_models
# from pypfopt.efficient_frontier import EfficientFrontier 

# def preprocess_stock_data(stock_data):
#     stock_data.ffill(inplace=True)
#     stock_data.bfill(inplace=True)
#     stock_data = stock_data[(np.abs(stock_data - stock_data.mean()) <= (3 * stock_data.std())).all(axis=1)]
#     return stock_data

# def create_features(stock_data):
#     stock_data = stock_data.copy()
#     stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
#     stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
#     stock_data['Lag_1'] = stock_data['Close'].shift(1)
#     stock_data.dropna(inplace=True)
#     return stock_data

# def fetch_and_train_multiple_models(ticker_symbol, start_date, end_date):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = preprocess_stock_data(stock_data)
#     stock_data = create_features(stock_data)

#     X = stock_data[['SMA_20', 'SMA_50', 'Lag_1']]
#     y = stock_data['Close']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     risk_assessment = {}
#     models_performance = {}
#     y_preds = {}

#     returns = stock_data['Close'].pct_change().dropna()
#     std_dev = returns.std()
#     VaR = np.percentile(returns, 5)
#     CVaR = returns[returns <= VaR].mean()

#     risk_assessment['Standard Deviation'] = std_dev
#     risk_assessment['Value at Risk'] = VaR
#     risk_assessment['Conditional Value at Risk'] = CVaR

#     lr_model = LinearRegression()
#     lr_model.fit(X_train_scaled, y_train)
#     y_pred_lr = lr_model.predict(X_test_scaled)
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     rmse_lr = np.sqrt(mse_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)
#     models_performance['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr}
#     y_preds['Linear Regression'] = y_pred_lr

#     rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     rmse_rf = np.sqrt(mse_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     models_performance['Random Forest'] = {'mse': mse_rf, 'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
#     y_preds['Random Forest'] = y_pred_rf

#     min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#     X_scaled = min_max_scaler.fit_transform(X)
#     y_scaled = min_max_scaler.fit_transform(y.values.reshape(-1, 1))

#     X_train_lstm = np.reshape(X_scaled[:len(X_train)], (len(X_train), 1, X_train.shape[1]))
#     X_test_lstm = np.reshape(X_scaled[len(X_train):], (len(X_test), 1, X_test.shape[1]))
#     y_train_lstm = y_scaled[:len(X_train)]

#     lstm_model = Sequential()
#     lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])))
#     lstm_model.add(LSTM(units=50))
#     lstm_model.add(Dense(1))
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#     lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
    
#     y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
#     y_pred_lstm = min_max_scaler.inverse_transform(y_pred_lstm_scaled)
#     mse_lstm = mean_squared_error(y_test, y_pred_lstm)
#     mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
#     rmse_lstm = np.sqrt(mse_lstm)
#     r2_lstm = r2_score(y_test, y_pred_lstm)
#     models_performance['LSTM'] = {'mse': mse_lstm, 'mae': mae_lstm, 'rmse': rmse_lstm, 'r2': r2_lstm}
#     y_preds['LSTM'] = y_pred_lstm.flatten()

#     mu = expected_returns.mean_historical_return(stock_data['Close'])
#     S = risk_models.sample_cov(stock_data['Close'])

#     return risk_assessment, models_performance, y_test, y_preds, mu, S 

# def get_earliest_trading_date(ticker_symbol):
#     stock_data = yf.download(ticker_symbol, start='1900-01-01', end='2023-01-01')
#     return stock_data.index.min().strftime('%Y-%m-%d')

# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# figures_folder = 'figures'
# os.makedirs(figures_folder, exist_ok=True)

# # Initialize lists for storing risk metrics
# all_std_devs = []
# all_vars = []
# all_cvars = []

# for ticker_symbol in tickers:
#     end_date = '2023-01-01'
#     start_date = get_earliest_trading_date(ticker_symbol)
#     print(f"Start Date for {ticker_symbol}: {start_date}")

#     risk_assessment, models_performance, y_test, y_preds, mu, S = fetch_and_train_multiple_models(ticker_symbol, start_date, end_date)
    
#     # Collect risk metrics
#     all_std_devs.append(risk_assessment['Standard Deviation'])
#     all_vars.append(risk_assessment['Value at Risk'])
#     all_cvars.append(risk_assessment['Conditional Value at Risk'])

#     for model_name, metrics in models_performance.items():
#         print(f"{model_name} Metrics for {ticker_symbol}:")
#         print(f"  MSE: {metrics['mse']:.4f}")
#         print(f"  MAE: {metrics['mae']:.4f}")
#         print(f"  RMSE: {metrics['rmse']:.4f}")
#         print(f"  R-squared: {metrics['r2']:.4f}")

#     print("##### Risk Assessment #####")
#     print(f"Standard Deviation (Volatility): {risk_assessment['Standard Deviation']}")
#     print(f"Value at Risk (VaR): {risk_assessment['Value at Risk']}")
#     print(f"Conditional Value at Risk (CVaR): {risk_assessment['Conditional Value at Risk']}")

#     display_start_date = y_test.index.min().strftime('%Y-%m-%d')
#     display_end_date = y_test.index.max().strftime('%Y-%m-%d')

#     plt.figure(figsize=(14, 7))
#     plt.plot(y_test.index, y_test.values, label='Actual', color='black')
#     for model_name, y_pred in y_preds.items():
#         plt.plot(y_test.index, y_pred, label=model_name)

#     plt.title(f'Stock Price Prediction for {ticker_symbol} from {display_start_date} to {display_end_date}')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.grid(True)

#     filename = f'stock_price_prediction_{ticker_symbol}_{display_start_date}_{display_end_date}.png'
#     filepath = os.path.join(figures_folder, filename)
#     plt.savefig(filepath)
#     plt.close()

#     metrics_labels = ['MSE', 'MAE', 'RMSE', 'R-squared']
#     models = list(models_performance.keys())
#     metrics_values = np.array([[metrics['mse'], metrics['mae'], metrics['rmse'], metrics['r2']] for metrics in models_performance.values()])

#     fig, ax = plt.subplots(figsize=(14, 7))
#     width = 0.2
#     x = np.arange(len(metrics_labels))

#     metrics_percentage_clipped = np.clip(metrics_values, a_min=0, a_max=100)

#     for i, model_name in enumerate(models):
#         ax.bar(x + i * width, metrics_percentage_clipped[i], width=width, label=model_name)
#     ax.set_xlabel('Metrics')
#     ax.set_ylabel('Value')
#     ax.set_title(f'Model Performance Metrics for {ticker_symbol} from {display_start_date} to {display_end_date}')
#     ax.set_xticks(x + width * (len(models) / 2) - width / 2)
#     ax.set_xticklabels(metrics_labels)
#     ax.legend()

#     metrics_filename = f'model_performance_metrics_{ticker_symbol}_{display_start_date}_{display_end_date}.png'
#     metrics_filepath = os.path.join(figures_folder, metrics_filename)
#     fig.savefig(metrics_filepath)
#     plt.close()

# # Box plot for risk metrics
# fig, ax = plt.subplots(figsize=(12, 8))
# risk_data = [all_std_devs, all_vars, all_cvars]
# risk_labels = ['Standard Deviation', 'Value at Risk', 'Conditional Value at Risk']

# ax.boxplot(risk_data, labels=risk_labels)
# ax.set_title('Risk Metrics Comparison')
# ax.set_ylabel('Value')
# plt.grid(True)

# boxplot_filename = 'risk_metrics_comparison.png'
# boxplot_filepath = os.path.join(figures_folder, boxplot_filename)
# plt.savefig(boxplot_filepath)
# plt.close()

# # combined_mu = np.mean(all_expected_returns, axis=0)
# # combined_S = np.mean(all_covariances, axis=0)

# # ef = EfficientFrontier(combined_mu, combined_S)
# # weights = ef.max_sharpe()
# # cleaned_weights = ef.clean_weights()
# # performance = ef.portfolio_performance(verbose=True)

# # plt.figure(figsize=(10, 5))
# # plt.bar(cleaned_weights.keys(), cleaned_weights.values())
# # plt.xlabel('Assets')
# # plt.ylabel('Proportion of Portfolio')
# # plt.title('Optimal Portfolio Allocation')
# # plt.grid(True)
# # portfolio_filename = 'optimal_portfolio_allocation.png'
# # portfolio_filepath = os.path.join(figures_folder, portfolio_filename)
# # plt.savefig(portfolio_filepath)
# # plt.close()
###########################
# import numpy as np
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense,Input
# import matplotlib.pyplot as plt
# import os
# from pypfopt import expected_returns, risk_models
# from pypfopt.efficient_frontier import EfficientFrontier 
# import pandas as pd
# def preprocess_stock_data(stock_data):
#     relevant_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
#     stock_data = stock_data.copy()
#     stock_data = stock_data[relevant_columns]
#     stock_data.ffill(inplace=True)
#     stock_data.bfill(inplace=True)
#     stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
#     stock_data.dropna(inplace=True)  # Drop rows with NaN values
#     stock_data = stock_data[(np.abs(stock_data - stock_data.mean()) <= (3 * stock_data.std())).all(axis=1)]
#     return stock_data

# def create_features(stock_data):
#     stock_data = stock_data.copy()
#     stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
#     stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
#     stock_data['Lag_1'] = stock_data['Close'].shift(1)
#     stock_data.dropna(inplace=True)
#     return stock_data

# def fetch_and_train_multiple_models(ticker_symbol, start_date, end_date):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = preprocess_stock_data(stock_data)
    
#     stock_data = create_features(stock_data)

#     X = stock_data[['SMA_20', 'SMA_50', 'Lag_1']]
#     y = stock_data['Close']
    
#     mu = expected_returns.mean_historical_return(y)
#     S = risk_models.sample_cov(y)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     risk_assessment = {}
#     models_performance = {}
#     y_preds = {}

#     returns = stock_data['Close'].pct_change().dropna()
#     std_dev = returns.std()
#     VaR = np.percentile(returns, 5)
#     CVaR = returns[returns <= VaR].mean()

#     risk_assessment['Standard Deviation'] = std_dev
#     risk_assessment['Value at Risk'] = VaR
#     risk_assessment['Conditional Value at Risk'] = CVaR

#     lr_model = LinearRegression()
#     lr_model.fit(X_train_scaled, y_train)
#     y_pred_lr = lr_model.predict(X_test_scaled)
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     rmse_lr = np.sqrt(mse_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)
#     models_performance['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr}
#     y_preds['Linear Regression'] = y_pred_lr

#     rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     rmse_rf = np.sqrt(mse_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     models_performance['Random Forest'] = {'mse': mse_rf, 'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
#     y_preds['Random Forest'] = y_pred_rf

#     min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#     X_scaled = min_max_scaler.fit_transform(X)
#     y_scaled = min_max_scaler.fit_transform(y.values.reshape(-1, 1))

#     X_train_lstm = np.reshape(X_scaled[:len(X_train)], (len(X_train), 1, X_train.shape[1]))
#     X_test_lstm = np.reshape(X_scaled[len(X_train):], (len(X_test), 1, X_test.shape[1]))
#     y_train_lstm = y_scaled[:len(X_train)]

#     # Define LSTM Model with Input layer
#     lstm_model = Sequential()
#     lstm_model.add(Input(shape=(1, X_train.shape[1])))  # Specify the input shape
#     lstm_model.add(LSTM(units=50, return_sequences=True))  # First LSTM layer
#     lstm_model.add(LSTM(units=50))  # Second LSTM layer
#     lstm_model.add(Dense(1))  # Output layer
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

#     # Train the model
#     lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
        
#     y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
#     y_pred_lstm = min_max_scaler.inverse_transform(y_pred_lstm_scaled)
#     mse_lstm = mean_squared_error(y_test, y_pred_lstm)
#     mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
#     rmse_lstm = np.sqrt(mse_lstm)
#     r2_lstm = r2_score(y_test, y_pred_lstm)
#     models_performance['LSTM'] = {'mse': mse_lstm, 'mae': mae_lstm, 'rmse': rmse_lstm, 'r2': r2_lstm}
#     y_preds['LSTM'] = y_pred_lstm.flatten()

    

#     return risk_assessment, models_performance, y_test, y_preds, mu, S 

# def get_earliest_trading_date(ticker_symbol):
#     stock_data = yf.download(ticker_symbol, start='1900-01-01', end='2023-01-01')
#     return stock_data.index.min().strftime('%Y-%m-%d')

# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META','NVDA','TSLA']

# figures_folder = 'figures'
# os.makedirs(figures_folder, exist_ok=True)

# # Define colors for each stock
# stock_colors = {
#     'AAPL': 'blue',
#     'MSFT': 'green',
#     'GOOGL': 'red',
#     'AMZN': 'purple',
#     'META': 'orange',
#     'NVDA': 'yellow',
#     'TSLA': 'grey'
# }

# # Initialize lists for storing risk metrics
# all_std_devs = []
# all_vars = []
# all_cvars = []
# tickers_list = []

# # Store expected returns and covariance matrices
# all_expected_returns = []
# all_covariances = []

# for ticker_symbol in tickers:
#     end_date = '2023-01-01'
#     start_date = '2008-01-01'#get_earliest_trading_date(ticker_symbol)
#     print(f"Start Date for {ticker_symbol}: {start_date}")

#     risk_assessment, models_performance, y_test, y_preds, mu, S = fetch_and_train_multiple_models(ticker_symbol, start_date, end_date)
    
#     all_expected_returns.append(mu)
#     all_covariances.append(S)

#     # Collect risk metrics
#     all_std_devs.append(risk_assessment['Standard Deviation'])
#     all_vars.append(risk_assessment['Value at Risk'])
#     all_cvars.append(risk_assessment['Conditional Value at Risk'])
#     tickers_list.append(ticker_symbol)

#     print("##### Risk Assessment #####")
#     print(f"Standard Deviation (Volatility): {risk_assessment['Standard Deviation']}")
#     print(f"Value at Risk (VaR): {risk_assessment['Value at Risk']}")
#     print(f"Conditional Value at Risk (CVaR): {risk_assessment['Conditional Value at Risk']}")

# # Plot comparative risk metrics
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# # Standard Deviation Plot
# for ticker, color in stock_colors.items():
#     axs[0].bar(ticker, all_std_devs[tickers_list.index(ticker)], color=color, label=ticker)
# axs[0].set_title('Standard Deviation')
# axs[0].set_xlabel('Stocks')
# axs[0].set_ylabel('Value')
# axs[0].grid(True)

# # Value at Risk Plot
# for ticker, color in stock_colors.items():
#     axs[1].bar(ticker, all_vars[tickers_list.index(ticker)], color=color, label=ticker)
# axs[1].set_title('Value at Risk (VaR)')
# axs[1].set_xlabel('Stocks')
# axs[1].grid(True)

# # Conditional Value at Risk Plot
# for ticker, color in stock_colors.items():
#     axs[2].bar(ticker, all_cvars[tickers_list.index(ticker)], color=color, label=ticker)
# axs[2].set_title('Conditional Value at Risk (CVaR)')
# axs[2].set_xlabel('Stocks')
# axs[2].grid(True)

# # Add a single legend to the figure
# handles = [plt.Line2D([0], [0], color=color, lw=4) for color in stock_colors.values()]
# labels = stock_colors.keys()
# fig.legend(handles, labels, loc='upper right', ncol=len(tickers), bbox_to_anchor=(0.5, 1.15))

# plt.suptitle('Comparative Risk Metrics Across Stocks')
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# # Save the comparative risk metrics figure
# comparative_risk_filename = 'comparative_risk_metrics.png'
# comparative_risk_filepath = os.path.join(figures_folder, comparative_risk_filename)
# plt.savefig(comparative_risk_filepath)
# plt.close()
# df = pd.DataFrame(all_expected_returns)


# combined_mu = np.mean(all_expected_returns, axis=1)  # Averaging expected returns
# all_covariances_array = np.array(all_covariances)  # Convert to 3D array
# all_covariances_array=all_covariances_array.flatten()

# # combined_S = np.mean(all_covariances_array, axis=1)  # Averaging covariance matrices


# # Combine all expected returns and covariances for portfolio optimization
# # combined_mu = np.mean(all_expected_returns)
# combined_S = np.cov(all_covariances_array)

# # Perform portfolio optimization
# ef = EfficientFrontier(combined_mu, combined_S)
# weights = ef.max_sharpe()  # Optimize for maximum Sharpe ratio
# cleaned_weights = ef.clean_weights()  # Clean up weights
# performance = ef.portfolio_performance(verbose=True)  # Get performance metrics

# # Plot portfolio optimization and allocation
# plt.figure(figsize=(10, 5))

# # Explicitly set the stock names as x-ticks labels
# plt.bar(tickers_list, cleaned_weights.values())

# # Set the x-ticks to the stock names
# plt.xticks(ticks=range(len(cleaned_weights)), labels=list(cleaned_weights.keys()))

# plt.xlabel('Assets')
# plt.ylabel('Proportion of Portfolio')
# plt.title('Optimal Portfolio Allocation')
# plt.grid(True)

# # Save the figure
# portfolio_filename = 'optimal_portfolio_allocation.png'
# portfolio_filepath = os.path.join(figures_folder, portfolio_filename)
# plt.savefig(portfolio_filepath)
# plt.show()  # Display the plot
# plt.close()



######## Revised version ########


import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
import os
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier 
import pandas as pd

def preprocess_stock_data(stock_data):
    relevant_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    stock_data = stock_data.copy()
    stock_data = stock_data[relevant_columns]
    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    stock_data.dropna(inplace=True)  # Drop rows with NaN values
    stock_data = stock_data[(np.abs(stock_data - stock_data.mean()) <= (3 * stock_data.std())).all(axis=1)]
    return stock_data

def create_features(stock_data):
    stock_data = stock_data.copy()
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data.dropna(inplace=True)
    return stock_data

def fetch_and_train_multiple_models(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data = preprocess_stock_data(stock_data)
    stock_data = create_features(stock_data)

    X = stock_data[['SMA_20', 'SMA_50', 'Lag_1']]
    y = stock_data['Close']
    
    mu = expected_returns.mean_historical_return(stock_data['Close'])
    S = risk_models.sample_cov(stock_data['Close'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    risk_assessment = {}
    models_performance = {}
    y_preds = {}

    returns = stock_data['Close'].pct_change().dropna()
    std_dev = returns.std()
    VaR = np.percentile(returns, 5)
    CVaR = returns[returns <= VaR].mean()

    risk_assessment['Standard Deviation'] = std_dev
    risk_assessment['Value at Risk'] = VaR
    risk_assessment['Conditional Value at Risk'] = CVaR

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    models_performance['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr}
    y_preds['Linear Regression'] = y_pred_lr

    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    models_performance['Random Forest'] = {'mse': mse_rf, 'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
    y_preds['Random Forest'] = y_pred_rf

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = min_max_scaler.fit_transform(X)
    y_scaled = min_max_scaler.fit_transform(y.values.reshape(-1, 1))

    X_train_lstm = np.reshape(X_scaled[:len(X_train)], (len(X_train), 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_scaled[len(X_train):], (len(X_test), 1, X_test.shape[1]))
    y_train_lstm = y_scaled[:len(X_train)]

    # Define LSTM Model with Input layer
    lstm_model = Sequential()
    lstm_model.add(Input(shape=(1, X_train.shape[1])))  # Specify the input shape
    lstm_model.add(LSTM(units=50, return_sequences=True))  # First LSTM layer
    lstm_model.add(LSTM(units=50))  # Second LSTM layer
    lstm_model.add(Dense(1))  # Output layer
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

    # Train the model
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
        
    y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
    y_pred_lstm = min_max_scaler.inverse_transform(y_pred_lstm_scaled)
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)
    models_performance['LSTM'] = {'mse': mse_lstm, 'mae': mae_lstm, 'rmse': rmse_lstm, 'r2': r2_lstm}
    y_preds['LSTM'] = y_pred_lstm.flatten()

    return risk_assessment, models_performance, y_test, y_preds, mu, S 

def get_earliest_trading_date(ticker_symbol):
    stock_data = yf.download(ticker_symbol, start='1900-01-01', end='2023-01-01')
    return stock_data.index.min().strftime('%Y-%m-%d')

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META','NVDA','TSLA']

figures_folder = 'figures'
os.makedirs(figures_folder, exist_ok=True)

# Define colors for each stock
stock_colors = {
    'AAPL': 'blue',
    'MSFT': 'green',
    'GOOGL': 'red',
    'AMZN': 'purple',
    'META': 'orange',
    'NVDA': 'yellow',
    'TSLA': 'grey'
}

# Initialize lists for storing risk metrics
all_std_devs = []
all_vars = []
all_cvars = []
tickers_list = []

# Store expected returns and covariance matrices
all_expected_returns = []
all_covariances = []
all_returns = []
for ticker_symbol in tickers:
    end_date = '2023-01-01'
    start_date = '2008-01-01'#get_earliest_trading_date(ticker_symbol)
    print(f"Start Date for {ticker_symbol}: {start_date}")

    risk_assessment, models_performance, y_test, y_preds, mu, S = fetch_and_train_multiple_models(ticker_symbol, start_date, end_date)
    
    all_expected_returns.append(mu)
    all_covariances.append(S)

    # Collect historical returns
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data = preprocess_stock_data(stock_data)
    returns = stock_data['Close'].pct_change().dropna()
    all_returns.append(returns)


    # Collect risk metrics
    all_std_devs.append(risk_assessment['Standard Deviation'])
    all_vars.append(risk_assessment['Value at Risk'])
    all_cvars.append(risk_assessment['Conditional Value at Risk'])
    tickers_list.append(ticker_symbol)

    print("##### Risk Assessment #####")
    print(f"Standard Deviation (Volatility): {risk_assessment['Standard Deviation']}")
    print(f"Value at Risk (VaR): {risk_assessment['Value at Risk']}")
    print(f"Conditional Value at Risk (CVaR): {risk_assessment['Conditional Value at Risk']}")
    # Extract the display period for plotting
    display_start_date = y_test.index.min().strftime('%Y-%m-%d')
    display_end_date = y_test.index.max().strftime('%Y-%m-%d')

    # Plot actual vs predicted stock prices
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual', color='black')  # Actual stock prices
    for model_name, y_pred in y_preds.items():
        plt.plot(y_test.index, y_pred, label=model_name)  # Predicted prices for each model

    plt.title(f'Stock Price Prediction for {ticker_symbol} from {display_start_date} to {display_end_date}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    # Save the prediction figure
    filename = f'stock_price_prediction_{ticker_symbol}_{display_start_date}_{display_end_date}.png'
    filepath = os.path.join(figures_folder, filename)
    plt.savefig(filepath)
    plt.close()  # Close the figure to free memory

# Plot comparative risk metrics
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Standard Deviation Plot
for ticker, color in stock_colors.items():
    axs[0].bar(ticker, all_std_devs[tickers_list.index(ticker)], color=color, label=ticker)
axs[0].set_title('Standard Deviation')
axs[0].set_xlabel('Stocks')
axs[0].set_ylabel('Value')
axs[0].grid(True)

# Value at Risk Plot
for ticker, color in stock_colors.items():
    axs[1].bar(ticker, all_vars[tickers_list.index(ticker)], color=color, label=ticker)
axs[1].set_title('Value at Risk (VaR)')
axs[1].set_xlabel('Stocks')
axs[1].grid(True)

# Conditional Value at Risk Plot
for ticker, color in stock_colors.items():
    axs[2].bar(ticker, all_cvars[tickers_list.index(ticker)], color=color, label=ticker)
axs[2].set_title('Conditional Value at Risk (CVaR)')
axs[2].set_xlabel('Stocks')
axs[2].grid(True)

# Add a single legend to the figure
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in stock_colors.values()]
labels = stock_colors.keys()
fig.legend(handles, labels, loc='upper right', ncol=len(tickers), bbox_to_anchor=(0.5, 1.15))

plt.suptitle('Comparative Risk Metrics Across Stocks')
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the comparative risk metrics figure
comparative_risk_filename = 'comparative_risk_metrics.png'
comparative_risk_filepath = os.path.join(figures_folder, comparative_risk_filename)
plt.savefig(comparative_risk_filepath)
plt.close()

# Compute combined expected returns and covariance matrices for portfolio optimization
combined_mu = np.mean(all_expected_returns, axis=0)  # Averaging expected returns
# Compute combined correlation matrix
returns_df = pd.concat(all_returns, axis=1)
returns_df.columns = tickers
correlation_matrix = returns_df.corr()

# Compute standard deviations for each stock
std_devs = returns_df.std()

# Convert correlation matrix to covariance matrix
cov_matrix = correlation_matrix * np.outer(std_devs, std_devs)

# Extracting the values and creating a new pandas Series
mu = pd.Series([ret['Close'] for ret in all_expected_returns], index=tickers)

# Perform portfolio optimization
ef = EfficientFrontier(mu, cov_matrix)
weights = ef.max_sharpe()  # Optimize for maximum Sharpe ratio
cleaned_weights = ef.clean_weights()  # Clean up weights
performance = ef.portfolio_performance(verbose=True)  # Get performance metrics

# Plot portfolio optimization and allocation
plt.figure(figsize=(10, 5))

# Explicitly set the stock names as x-ticks labels
plt.bar(tickers_list, cleaned_weights.values())

# Set the x-ticks to the stock names
plt.xticks(ticks=range(len(cleaned_weights)), labels=list(cleaned_weights.keys()))

plt.xlabel('Assets')
plt.ylabel('Proportion of Portfolio')
plt.title('Optimal Portfolio Allocation')
plt.grid(True)

# Save the figure
portfolio_filename = 'optimal_portfolio_allocation.png'
portfolio_filepath = os.path.join(figures_folder, portfolio_filename)
plt.savefig(portfolio_filepath)
plt.show()  # Display the plot
plt.close()



# model_names = ['Linear Regression', 'Random Forest', 'LSTM']
# metrics = ['mse', 'mae', 'rmse', 'r2']
# # Initialize a DataFrame to hold the performance metrics
# performance_df = pd.DataFrame(index=tickers_list, columns=model_names)

# # Fill the DataFrame with the collected metrics
# for ticker in tickers_list:
#     for model_name in model_names:
#         performance_df.loc[ticker, model_name] = models_performance[model_name][metrics].values

# # Convert to float for plotting
# performance_df = performance_df.astype(float)

# # Plot comparative performance metrics for each model
# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Model Performance Metrics Across Stocks', fontsize=20)

# # MSE Plot
# performance_df[model_names].plot(kind='bar', ax=axs[0, 0], color=['blue', 'green', 'orange'])
# axs[0, 0].set_title('Mean Squared Error (MSE)')
# axs[0, 0].set_ylabel('MSE Value')
# axs[0, 0].set_xlabel('Stocks')
# axs[0, 0].grid(True)

# # MAE Plot
# performance_df[model_names].plot(kind='bar', ax=axs[0, 1], color=['blue', 'green', 'orange'])
# axs[0, 1].set_title('Mean Absolute Error (MAE)')
# axs[0, 1].set_ylabel('MAE Value')
# axs[0, 1].set_xlabel('Stocks')
# axs[0, 1].grid(True)

# # RMSE Plot
# performance_df[model_names].plot(kind='bar', ax=axs[1, 0], color=['blue', 'green', 'orange'])
# axs[1, 0].set_title('Root Mean Squared Error (RMSE)')
# axs[1, 0].set_ylabel('RMSE Value')
# axs[1, 0].set_xlabel('Stocks')
# axs[1, 0].grid(True)

# # R² Plot
# performance_df[model_names].plot(kind='bar', ax=axs[1, 1], color=['blue', 'green', 'orange'])
# axs[1, 1].set_title('R² Score')
# axs[1, 1].set_ylabel('R² Value')
# axs[1, 1].set_xlabel('Stocks')
# axs[1, 1].grid(True)

# # Adjust layout and show plot
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig(os.path.join(figures_folder, 'model_performance_metrics.png'))
# plt.show()




############################### Comparing the metrics of all the models for each stock ###########



# import numpy as np
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Input
# import matplotlib.pyplot as plt
# import os
# from pypfopt import expected_returns, risk_models
# from pypfopt.efficient_frontier import EfficientFrontier
# import pandas as pd

# def preprocess_stock_data(stock_data):
#     relevant_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
#     stock_data = stock_data.copy()
#     stock_data = stock_data[relevant_columns]
#     stock_data.ffill(inplace=True)
#     stock_data.bfill(inplace=True)
#     stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
#     stock_data.dropna(inplace=True)  # Drop rows with NaN values
#     stock_data = stock_data[(np.abs(stock_data - stock_data.mean()) <= (3 * stock_data.std())).all(axis=1)]
#     return stock_data

# def create_features(stock_data):
#     stock_data = stock_data.copy()
#     stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
#     stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
#     stock_data['Lag_1'] = stock_data['Close'].shift(1)
#     stock_data.dropna(inplace=True)
#     return stock_data

# def fetch_and_train_multiple_models(ticker_symbol, start_date, end_date):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = preprocess_stock_data(stock_data)
#     stock_data = create_features(stock_data)

#     X = stock_data[['SMA_20', 'SMA_50', 'Lag_1']]
#     y = stock_data['Close']

#     mu = expected_returns.mean_historical_return(stock_data['Close'])
#     S = risk_models.sample_cov(stock_data['Close'])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     risk_assessment = {}
#     models_performance = {}
#     y_preds = {}

#     returns = stock_data['Close'].pct_change().dropna()
#     std_dev = returns.std()
#     VaR = np.percentile(returns, 5)
#     CVaR = returns[returns <= VaR].mean()

#     risk_assessment['Standard Deviation'] = std_dev
#     risk_assessment['Value at Risk'] = VaR
#     risk_assessment['Conditional Value at Risk'] = CVaR

#     lr_model = LinearRegression()
#     lr_model.fit(X_train_scaled, y_train)
#     y_pred_lr = lr_model.predict(X_test_scaled)
#     mse_lr = mean_squared_error(y_test, y_pred_lr)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     rmse_lr = np.sqrt(mse_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)
#     models_performance['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr}
#     y_preds['Linear Regression'] = y_pred_lr

#     rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     rmse_rf = np.sqrt(mse_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     models_performance['Random Forest'] = {'mse': mse_rf, 'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
#     y_preds['Random Forest'] = y_pred_rf

#     min_max_scaler = MinMaxScaler(feature_range=(0, 1))
#     X_scaled = min_max_scaler.fit_transform(X)
#     y_scaled = min_max_scaler.fit_transform(y.values.reshape(-1, 1))

#     X_train_lstm = np.reshape(X_scaled[:len(X_train)], (len(X_train), 1, X_train.shape[1]))
#     X_test_lstm = np.reshape(X_scaled[len(X_train):], (len(X_test), 1, X_test.shape[1]))
#     y_train_lstm = y_scaled[:len(X_train)]

#     # Define LSTM Model with Input layer
#     lstm_model = Sequential()
#     lstm_model.add(Input(shape=(1, X_train.shape[1])))  # Specify the input shape
#     lstm_model.add(LSTM(units=50, return_sequences=True))  # First LSTM layer
#     lstm_model.add(LSTM(units=50))  # Second LSTM layer
#     lstm_model.add(Dense(1))  # Output layer
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

#     # Train the model
#     lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
        
#     y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
#     y_pred_lstm = min_max_scaler.inverse_transform(y_pred_lstm_scaled)
#     mse_lstm = mean_squared_error(y_test, y_pred_lstm)
#     mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
#     rmse_lstm = np.sqrt(mse_lstm)
#     r2_lstm = r2_score(y_test, y_pred_lstm)
#     models_performance['LSTM'] = {'mse': mse_lstm, 'mae': mae_lstm, 'rmse': rmse_lstm, 'r2': r2_lstm}
#     y_preds['LSTM'] = y_pred_lstm.flatten()

#     return risk_assessment, models_performance, y_test, y_preds, mu, S 

# def get_earliest_trading_date(ticker_symbol):
#     stock_data = yf.download(ticker_symbol, start='1900-01-01', end='2023-01-01')
#     return stock_data.index.min().strftime('%Y-%m-%d')

# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META','NVDA','TSLA']

# figures_folder = 'figures'
# os.makedirs(figures_folder, exist_ok=True)

# # Define colors for each stock
# stock_colors = {
#     'AAPL': 'blue',
#     'MSFT': 'green',
#     'GOOGL': 'red',
#     'AMZN': 'purple',
#     'META': 'orange',
#     'NVDA': 'yellow',
#     'TSLA': 'grey'
# }

# # Initialize lists for storing risk metrics
# all_std_devs = []
# all_vars = []
# all_cvars = []
# tickers_list = []

# # Store expected returns and covariance matrices
# all_expected_returns = []
# all_covariances = []
# all_returns = []
# for ticker_symbol in tickers:
#     end_date = '2023-01-01'
#     start_date = '2008-01-01'  # get_earliest_trading_date(ticker_symbol)
#     print(f"Start Date for {ticker_symbol}: {start_date}")

#     risk_assessment, models_performance, y_test, y_preds, mu, S = fetch_and_train_multiple_models(ticker_symbol, start_date, end_date)
    
#     all_expected_returns.append(mu)
#     all_covariances.append(S)

#     # Collect historical returns
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = preprocess_stock_data(stock_data)
#     returns = stock_data['Close'].pct_change().dropna()
#     all_returns.append(returns)

#     # Collect risk metrics
#     all_std_devs.append(risk_assessment['Standard Deviation'])
#     all_vars.append(risk_assessment['Value at Risk'])
#     all_cvars.append(risk_assessment['Conditional Value at Risk'])
#     tickers_list.append(ticker_symbol)

#     print("##### Risk Assessment #####")
#     print(f"Standard Deviation (Volatility): {risk_assessment['Standard Deviation']}")
#     print(f"Value at Risk (VaR): {risk_assessment['Value at Risk']}")
#     print(f"Conditional Value at Risk (CVaR): {risk_assessment['Conditional Value at Risk']}")

#     # Extract the display period for plotting
#     display_start_date = y_test.index.min().strftime('%Y-%m-%d')
#     display_end_date = y_test.index.max().strftime('%Y-%m-%d')

#     # Plot actual vs predicted stock prices
#     plt.figure(figsize=(14, 7))
#     plt.plot(y_test.index, y_test.values, label='Actual', color='black')  # Actual stock prices
#     for model_name, y_pred in y_preds.items():
#         plt.plot(y_test.index, y_pred, label=model_name)  # Predicted prices for each model

#     plt.title(f'{ticker_symbol} Actual vs Predicted Stock Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.grid()
#     plt.xlim(display_start_date, display_end_date)
#     plt.savefig(os.path.join(figures_folder, f"{ticker_symbol}_actual_vs_predicted.png"))
#     plt.close()

# # Create a DataFrame for risk metrics
# risk_metrics_df = pd.DataFrame({
#     'Ticker': tickers_list,
#     'Standard Deviation': all_std_devs,
#     'Value at Risk': all_vars,
#     'Conditional Value at Risk': all_cvars
# })

# print("\n##### Risk Metrics DataFrame #####")
# print(risk_metrics_df)

# # Plotting model performance metrics
# model_names = list(models_performance.keys())
# metrics = ['mse', 'mae', 'rmse', 'r2']
# metrics_values = {metric: [] for metric in metrics}

# # Collect metrics values for each model
# for model in model_names:
#     for metric in metrics:
#         metrics_values[metric].append(models_performance[model][metric])

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# x = np.arange(len(model_names))  # the label locations
# width = 0.2  # the width of the bars

# for i, metric in enumerate(metrics):
#     ax.bar(x + i * width, metrics_values[metric], width, label=metric)

# ax.set_xlabel('Models')
# ax.set_title('Model Performance Metrics')
# ax.set_xticks(x + width / 2)
# ax.set_xticklabels(model_names)
# ax.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(figures_folder, "model_performance_metrics.png"))
# plt.show()
