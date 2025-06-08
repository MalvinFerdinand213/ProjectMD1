import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import NotFittedError


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)


class DataCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cleaned = X.copy()
        X_cleaned['wip'] = X_cleaned['wip'].fillna(0.0)
        X_cleaned.loc[X_cleaned['department'] == 'sweing', 'department'] = 'sewing'
        X_cleaned.loc[X_cleaned['department'] == 'finishing ', 'department'] = 'finishing'
        return X_cleaned


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['date'] = pd.to_datetime(X_transformed['date'])
        X_transformed['day'] = X_transformed['date'].dt.day
        X_transformed['month'] = X_transformed['date'].dt.month
        
        columns_to_drop = ['date', 'quarter']
        if 'year' in X_transformed.columns:
            columns_to_drop.append('year')

        X_transformed.drop(columns=[col for col in columns_to_drop if col in X_transformed.columns],
                           axis=1, inplace=True)
        return X_transformed


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model_pipeline = None
        self.best_model = None
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.y_predict = None

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state
        )

    def create_pipeline(self):
        initial_preprocessing_pipeline = Pipeline(steps=[
            ('cleaner', DataCleanerTransformer()),
            ('date_features', DateFeatureTransformer()),
        ])

        numerical_features_final = [
            'team', 'targeted_productivity', 'smv', 'wip', 'over_time',
            'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers',
            'day', 'month'
        ]
        categorical_features_final = ['department']

        final_feature_transformer = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('scaler', MinMaxScaler())]), numerical_features_final),
                ('cat', Pipeline(steps=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))]), categorical_features_final)
            ],
            remainder='drop'
        )

        self.model_pipeline = Pipeline(steps=[
            ('initial_preprocess', initial_preprocessing_pipeline),
            ('final_feature_process', final_feature_transformer),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

    def train_model(self):
        if self.best_model:
            pass
        elif self.model_pipeline:
            self.model_pipeline.fit(self.x_train, self.y_train)
            self.best_model = self.model_pipeline

    def evaluate_model(self):
        predictions = self.best_model.predict(self.x_test)
        
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mse)

        print("\nTest Set Evaluation Metrics:")
        print(f"  MSE: {mse:.8f}")
        print(f"  R²: {r2:.8f}")
        print(f"  MAE: {mae:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        return r2

    def makePrediction(self):
        self.y_predict = self.best_model.predict(self.x_test)

    def createReport(self):
        mse = mean_squared_error(self.y_test, self.y_predict)
        r2 = r2_score(self.y_test, self.y_predict)
        mae = mean_absolute_error(self.y_test, self.y_predict)
        rmse = np.sqrt(mse)

        print('--- Final Regression Report ---')
        print(f"MSE: {mse:.8f}")
        print(f"R²: {r2:.8f}")
        print(f"MAE: {mae:.8f}")
        print(f"RMSE: {rmse:.8f}")
        print("-------------------------------")

    def tuningParameter(self):
        parameters = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 5, 10]
        }

        grid_search = GridSearchCV(
            estimator=self.model_pipeline,
            param_grid=parameters,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.x_train, self.y_train)

        print("Tuned Hyperparameters :", grid_search.best_params_)
        print("Best Cross-Validation Score (negative MSE) :", grid_search.best_score_)
        
        self.best_model = grid_search.best_estimator_

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
                pickle.dump(self.best_model, file)


FILE_PATH = 'garments_worker_productivity.csv'
TARGET_COL = 'actual_productivity'
OUTPUT_MODEL = 'trained_model.pkl'

data_handler = DataHandler(FILE_PATH)
data_handler.load_data()
data_handler.create_input_output(TARGET_COL)

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()
model_handler.create_pipeline()

# Pre-Tuning
model_handler.train_model()
model_handler.makePrediction()
model_handler.evaluate_model()
model_handler.createReport()

# Post-Tuning
model_handler.tuningParameter()
model_handler.makePrediction()
model_handler.evaluate_model()
model_handler.createReport()

model_handler.save_model_to_file(OUTPUT_MODEL)
