import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# Prediction class that uses the LSTM Model trained in the notebooks
class LSTMForecaster:
    # Defining static variables of the repositories where the necessary stuff is
    models_save_repo = "../Saved Models"
    scalers_save_repo = "../Scalers"
    
    def __init__(self)->None:
        # Loading the separate models
        self.model_1_2 = tf.keras.models.load_model(LSTMForecaster.models_save_repo+'/m_1_2.keras')
        self.model_others = tf.keras.models.load_model(LSTMForecaster.models_save_repo+'/m_others.keras')
        self.model_3 = tf.keras.models.load_model(LSTMForecaster.models_save_repo+'/m_3.keras')
        self.model_6 = tf.keras.models.load_model(LSTMForecaster.models_save_repo+'/m_6.keras')
        self.model_empty = tf.keras.models.load_model(LSTMForecaster.models_save_repo+'/m_empty.keras')
        
        # Loading all the scalers
        with open(LSTMForecaster.scalers_save_repo+'/scaler_1_2.pickle', 'rb') as f:
            self.scalers_1_2 = pickle.load(f)

        with open(LSTMForecaster.scalers_save_repo+'/scaler_3.pickle', 'rb') as f:
            self.scalers_3 = pickle.load(f)

        with open(LSTMForecaster.scalers_save_repo+'/scaler_6.pickle', 'rb') as f:
            self.scalers_6 = pickle.load(f)

        with open(LSTMForecaster.scalers_save_repo+'/scaler_others.pickle', 'rb') as f:
            self.scalers_others = pickle.load(f)

        with open(LSTMForecaster.scalers_save_repo+'/scaler_empty.pickle', 'rb') as f:
            self.scalers_empty = pickle.load(f)

    # Method to add the lag 
    def add_lag_to_df(self,df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for i in range(1, 31):
            df_copy[f'REVENUE_LAG_{i}'] = df_copy['REVENUE'].shift(i)
            df_copy[f'IS_CODE6_ENABLED_{i}'] = df_copy['IS_CODE6_ENABLED'].shift(i).astype(float)
        
        df_copy.dropna(subset=[col for col in df_copy.columns if col != 'REPORT_DATE'],inplace=True)
        
        if df_copy.shape[0] == 0:
            raise ValueError("No valid data left after applying lags.")
        
        df.reset_index(drop=True, inplace=True)
        return df_copy

    # Method to split dataframes based on the subscription
    def split_dataframes_by_subscription(self,grouped_df)->dict:    
        empty_subscription_df =  grouped_df[grouped_df["IN_SUBSCRIPTION_TYPE"].isin(np.array([-1]))].drop(columns=["IN_SUBSCRIPTION_TYPE"]).reset_index(drop=True)
        subscriptions_1_2_df = grouped_df[grouped_df["IN_SUBSCRIPTION_TYPE"].isin(np.array([1,2]))].groupby(["REPORT_DATE"]).agg({
            "REVENUE":"sum",
            "IS_CODE6_ENABLED":"first",
        }).reset_index()
        subscriptions_3_df = grouped_df[grouped_df["IN_SUBSCRIPTION_TYPE"].isin(np.array([3]))].drop(columns=["IN_SUBSCRIPTION_TYPE"]).reset_index(drop=True)
        subscriptions_6_df = grouped_df[grouped_df["IN_SUBSCRIPTION_TYPE"].isin(np.array([6]))].drop(columns=["IN_SUBSCRIPTION_TYPE"]).reset_index(drop=True)
        other_subscriptions_df = grouped_df[grouped_df["IN_SUBSCRIPTION_TYPE"].isin(np.array([4,5,7,8,9]))].groupby(["REPORT_DATE"]).agg({
            "REVENUE":"sum",
            "IS_CODE6_ENABLED":"first",
        }).reset_index()
    
        return empty_subscription_df,subscriptions_1_2_df,subscriptions_3_df,subscriptions_6_df,other_subscriptions_df

    # Method responsible for the prediction
    def predict(self,n_days: int = 1, revenue_by_offer_data: pd.DataFrame = pd.DataFrame(), code6_activation_status: list = [True]) -> np.ndarray:
        assert n_days > 0 and n_days < 45, "The number of days should be a positive number and under 45 days ahead."
        
        # Ensure the list and the n_days match-up
        if len(code6_activation_status) != n_days:
            raise Exception("The number of days to predict must match the code6_activation list length!")
    
        if revenue_by_offer_data.shape[0] < 300:
            raise Exception("At least 30 days of data are required for all subscriptions!")
    
        # Split the data into different subscription types
        empty_subscription_df, subscriptions_1_2_df, subscriptions_3_df, subscriptions_6_df, other_subscriptions_df = self.split_dataframes_by_subscription(revenue_by_offer_data)
    
        predictions = []
    
        # Create a new row with the predicted REVENUE and update 'IS_CODE6_ENABLED' based on the code6_activation_status
        row_maker = lambda pred, status: pd.DataFrame({'REVENUE': [pred[0]], 'IS_CODE6_ENABLED': [status]})
    
        for day in range(n_days):
            # Extract the last 30 days of lagged data for each subscription type
            X_1_2 = self.add_lag_to_df(subscriptions_1_2_df.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values
            X_3 = self.add_lag_to_df(subscriptions_3_df.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values
            X_6 = self.add_lag_to_df(subscriptions_6_df.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values
            X_others = self.add_lag_to_df(other_subscriptions_df.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values
            X_empty = self.add_lag_to_df(empty_subscription_df.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values
    
            # Setting the IS_CODE6_ENABLED to today's status
            X_1_2[0][0] = code6_activation_status[day]
            X_3[0][0] = code6_activation_status[day]
            X_6[0][0] = code6_activation_status[day]
            X_others[0][0] = code6_activation_status[day]
            X_empty[0][0] = code6_activation_status[day]
    
            # Scale the data using the corresponding scaler
            X_1_2_scaled = self.scalers_1_2['scaler_X'].transform(X_1_2).reshape(-1, X_1_2.shape[0], X_1_2.shape[1])
            X_3_scaled = self.scalers_3['scaler_X'].transform(X_3).reshape(-1, X_3.shape[0],  X_3.shape[1])
            X_6_scaled = self.scalers_6['scaler_X'].transform(X_6).reshape(-1, X_6.shape[0],  X_6.shape[1])
            X_others_scaled = self.scalers_others['scaler_X'].transform(X_others).reshape(-1, X_others.shape[0],  X_others.shape[1])
            X_empty_scaled = self.scalers_empty['scaler_X'].transform(X_empty).reshape(-1, X_empty.shape[0],  X_empty.shape[1])
    
            # Predict using each model
            pred_1_2 = self.model_1_2.predict(X_1_2_scaled)
            pred_3 = self.model_3.predict(X_3_scaled)
            pred_6 = self.model_6.predict(X_6_scaled)
            pred_others = self.model_others.predict(X_others_scaled)
            pred_empty = self.model_empty.predict(X_empty_scaled)
    
            # Inverse scale the predictions
            pred_1_2_original = self.scalers_1_2['scaler_y'].inverse_transform(pred_1_2)
            pred_3_original = self.scalers_3['scaler_y'].inverse_transform(pred_3)
            pred_6_original = self.scalers_6['scaler_y'].inverse_transform(pred_6)
            pred_others_original = self.scalers_others['scaler_y'].inverse_transform(pred_others)
            pred_empty_original = self.scalers_empty['scaler_y'].inverse_transform(pred_empty)
    
            # Sum the predictions from all models to get the final predicted revenue for the day
            total_pred = (pred_1_2_original + pred_3_original + pred_6_original + pred_others_original + pred_empty_original)
            
            # Add the total prediction to the list of predictions
            predictions.append(total_pred[0][0])
    
            # Update the dataframes by appending the new row properly
            subscriptions_1_2_df = pd.concat([subscriptions_1_2_df, row_maker(pred_1_2_original, code6_activation_status[day])], ignore_index=True)
            subscriptions_3_df = pd.concat([subscriptions_3_df, row_maker(pred_3_original, code6_activation_status[day])], ignore_index=True)
            subscriptions_6_df = pd.concat([subscriptions_6_df, row_maker(pred_6_original, code6_activation_status[day])], ignore_index=True)
            other_subscriptions_df = pd.concat([other_subscriptions_df, row_maker(pred_others_original, code6_activation_status[day])], ignore_index=True)
            empty_subscription_df = pd.concat([empty_subscription_df, row_maker(pred_empty_original, code6_activation_status[day])], ignore_index=True)
        
        return np.array(predictions)
    
    
# Prediction class that uses the XGBoost Model trained in the notebooks
class XGBForecaster:
    path = "../Saved Models/xgboost_model.pickle"
    def __init__(self):
        # Loading the model from the save location
        with open(XGBForecaster.path, 'rb') as file:
            self.model = pickle.load(file)

    # Method to add the lag 
    def add_lag_to_df(self,df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for i in range(1, 31):
            df_copy[f'REVENUE_LAG_{i}'] = df_copy['REVENUE'].shift(i)
            df_copy[f'IS_CODE6_ENABLED_{i}'] = df_copy['IS_CODE6_ENABLED'].shift(i).astype(float)
        
        df_copy.dropna(subset=[col for col in df_copy.columns if col != 'REPORT_DATE'],inplace=True)
        
        if df_copy.shape[0] == 0:
            raise ValueError("No valid data left after applying lags.")
        
        df.reset_index(drop=True, inplace=True)
        return df_copy

    # Method responsible for the prediction
    def predict(self,n_days: int = 1, revenue_by_day_data: pd.DataFrame = pd.DataFrame(), code6_activation_status: list = [True]) -> np.ndarray:
        assert n_days > 0 and n_days < 45, "The number of days should be a positive number and under 45 days ahead."
        
        # Ensure the list and the n_days match-up
        if len(code6_activation_status) != n_days:
            raise Exception("The number of days to predict must match the code6_activation list length!")
        
        # Grouping the data by day so that even if the df contains the subscription it can handle it
        revenue_by_day_data = revenue_by_day_data.groupby(["REPORT_DATE"]).agg({
            "REVENUE":"sum",
            "IS_CODE6_ENABLED":"first"
        }).reset_index()
    
        if revenue_by_day_data.shape[0] < 30:
            raise Exception("At least 30 days of data are required!")

        # Create a new row with the predicted REVENUE and update 'IS_CODE6_ENABLED' based on the code6_activation_status
        row_maker = lambda pred, status: pd.DataFrame({'REVENUE': [pred[0]], 'IS_CODE6_ENABLED': [status]})

        predictions=[]

        for day in range(n_days):
            # Lagging the data of the last recorded day
            last_day_lagged_data = self.add_lag_to_df(revenue_by_day_data.iloc[-31:]).drop(columns=["REPORT_DATE","REVENUE"]).values

            # Updating the today's code6 status
            last_day_lagged_data[0][0]=  code6_activation_status[day] 

            # Saving the prediction in the array
            predictions.append(self.model.predict(last_day_lagged_data))

            # Adding the new prediction to the df for further predictions when slicing
            revenue_by_day_data = pd.concat([revenue_by_day_data, row_maker(predictions[-1], code6_activation_status[day])], ignore_index=True)

        return revenue_by_day_data