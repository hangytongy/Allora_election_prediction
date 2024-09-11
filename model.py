import random
import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense, Bidirectional
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    print(f"token {token}, taining_days : {training_days}")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)


def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:,['open','high','low','close']].dropna()
    df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def create_lstm_data(df, target_col, n_steps):
    x, y = [], []
    for i in range(n_steps, len(df)):
        x.append(df.iloc[i-n_steps:i].values)
        y.append(df.iloc[i][target_col])

    return np.array(x),np.array(y)

def train_model(timeframe):
    # Load the price data
    print("wtf am i running this???")
    price_data = pd.read_csv(training_price_data_path)

    '''
    df = load_frame(price_data, timeframe)

    print(df.tail())

    y_train = df['close'].shift(-1).dropna().values
    X_train = df[:-1]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    # Add more models here
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    model.fit(X_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    
    '''

    if MODEL == 'GRU':

        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        df = pd.DataFrame()

        # Convert 'date' to a numerical value (timestamp) we can use for regression
        df["date"] = pd.to_datetime(price_data["date"])
        df["date"] = df["date"].map(pd.Timestamp.timestamp)

        df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

        #Feature scaling
        scaler = MinMaxScaler()

        df_scaled = scaler.fit_transform(df[['date', 'price']])
        df_scaled = pd.DataFrame(df_scaled, columns =['date', 'price'])

        # Prepare data for LSTM
        n_steps = 35  #Number of time steps
        x, y = create_lstm_data(df_scaled, target_col='price', n_steps = n_steps)
        # Split the data into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Reshape x for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # Build the LSTM model
        grumodel = Sequential()
        grumodel.add(GRU(units=120, return_sequences=True, input_shape = (x_train.shape[1], x_train.shape[2])))
        grumodel.add(Dropout(0.2))
        grumodel.add(GRU(units=80, return_sequences=False))
        grumodel.add(Dropout(0.2))
        grumodel.add(Dense(units=1))

        # Compile the model
        grumodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss ='mean_squared_error')

        # Implement early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience =5, restore_best_weights=True)

        #Train the model
        grumodel.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=2)
        '''
        # Make predictions
        y_pred = grumodel.predict(x_test)
        y_test_inverse = scaler.inverse_transform(np.concatenate((x_test[:, -1, :1], y_test.reshape(-1, 1)), axis=1))[:, -1]
        y_pred_inverse = scaler.inverse_transform(np.concatenate((x_test[:, -1, :1], y_pred), axis=1))[:, -1]


        # Calculate the mean squared error
        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        print(f'Mean Squared Error: {mse}')

        #Calculate R-squared
        r2 = r2_score(y_test_inverse, y_pred_inverse)
        print(f'R-squared score: {r2}')

        # Print a few actual vs predicted values
        for actual, predicted in zip(y_test_inverse[:10], y_pred_inverse[:10]):
            print(f'Actual: {actual}, Predicted: {predicted}')
        '''
        # create the model's parent directory if it doesn't exist
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

        grumodel.save(model_file_path)

    print(f"Trained model saved to {model_file_path}")


def get_inference(token, timeframe, region, data_provider):
    """Load model and predict current price."""
    '''
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)
    
    print(X_new.tail())
    print(X_new.shape)

    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0]
    '''

    loaded_model = tf.keras.models.load_model(model_file_path)

    # Load and preprocess the data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to numerical timestamp and calculate price
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Feature scaling
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['date', 'price']])

    # Prepare the latest sequence for prediction
    n_steps = 35
    if len(df_scaled) < n_steps:
        raise ValueError("Not enough data to make a prediction. Need at least {} rows, but got {}.".format(n_steps, len(df_scaled)))

    latest_data = df_scaled[-n_steps:]  # Get the latest n_steps data
    latest_sequence = np.expand_dims(latest_data, axis=0)  # Shape: (1, n_steps, 2)

    # Predict the next price
    current_price_pred = loaded_model.predict(latest_sequence)

    # Reverse the scaling for the predicted price
    last_price_value = latest_data[-1, 1]  # Last price value (used for reverse scaling)

    # Combine last_date_value with predicted price
    pred_data = np.array([[last_price_value, current_price_pred[0, 0]]])  # Shape (1, 2)

     # Create a new scaler for reverse scaling, using 'price' column only
    scaler_price = MinMaxScaler()
    scaler_price.fit(df[['price']])  # Ensure this fits to the 'price' column used during training
    pred_data_unscaled = scaler_price.inverse_transform(pred_data)


    # Extract the actual predicted price
    predicted_price = pred_data_unscaled[0, 1]  # The unscaled predicted price


    return predicted_price
