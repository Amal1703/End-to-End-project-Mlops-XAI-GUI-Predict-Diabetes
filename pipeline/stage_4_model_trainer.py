from utils.common import load_yaml_config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb  # XGboost
from sklearn.svm import SVC  # SVM
# ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2 as l2_reg
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random


def ANN(hidden_layer_sizes, activation_hidden_layer, l2, dropout_values, optimizer,
        lr, epochs, batch_size, X_train_scaled, y_train_2, X_val_scaled, y_val_2):

    seed = 45
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_ANN = Sequential([
        Dense(hidden_layer_sizes[0], activation=activation_hidden_layer, kernel_regularizer=l2_reg(l2), input_shape=(X_train_scaled.shape[1],)),
        Dropout(dropout_values[0]),
        Dense(hidden_layer_sizes[1], activation=activation_hidden_layer, kernel_regularizer=l2_reg(l2)),
        Dropout(dropout_values[1]),
        Dense(1, activation='sigmoid')  # Binary output; for multi-class use softmax with n_classes
                                ])

    model_ANN.compile(
        optimizer=getattr(tf.keras.optimizers, optimizer)(learning_rate=lr),
        loss='binary_crossentropy',   # 'sparse_categorical_crossentropy' for multi-class
        metrics=['accuracy'])

# Native early stopping, based on val_loss
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=10,
                               restore_best_weights=True)

# Reduce LR by factor of 10 when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,  # Divide by 10 (multiply by 0.1)
                                  patience=5,  # Wait 5 epochs before reducing
                                  min_lr=1e-7,  # Minimum learning rate
                                  verbose=2)

# Training
    history = model_ANN.fit(
        X_train_scaled, y_train_2,
        validation_data=(X_val_scaled, y_val_2),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop, reduce_lr])

# --- Plot loss train vs val ---
    best_epoch = len(history.history['loss']) - early_stop.patience
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], label='Train loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation loss')
    plt.axvline(best_epoch, color='red', linestyle='--', alpha=0.7,
                label=f'Meilleure epoch restaurée ({best_epoch})')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model_ANN

def SVM(X_train_1, X_test_1, y_train, y_test, C, gamma) -> None :
    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_1)
    X_test_scaled = scaler.transform(X_test_1)
    svm_model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    calculate_errors(y_pred_svm, y_test)
    search_false_positives_or_false_negatives_in_predictions(y_pred_svm, y_test)


def XGBoost(X_train, X_test, y_train, y_test, n_estimators) -> None :
    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    errors, n_errors = calculate_errors(y_pred_xgb, y_test)
    return errors, n_errors


def read_file(file_path):
    """Return the lines of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [item.rstrip('\n') for item in lines]  # delete \n

            if not lines:
                print("❌ The file is empty")
                return

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' does not exist.")

    return lines

def search_false_positives_or_false_negatives_in_predictions(y_pred, y_test) -> None:
    y_test_arr = np.array(y_test).flatten()
    false_positives_idx = np.where((y_pred == 1) & (y_test_arr == 0))[0]
    false_negatives_idx = np.where((y_pred == 0) & (y_test_arr == 1))[0]
    print(f"False positives (predicted diabetic, but not): {len(false_positives_idx)}")
    print(f"False negatives (predicted non-diabetic, but actually is): {len(false_negatives_idx)}")

def calculate_errors(y_pred, y_test)  -> None:
    errors = np.abs(y_pred - y_test.values)
    n_errors = (errors == 1).sum()
    print(f"Number of errors: {n_errors} / {len(y_test)}")
    print(f"Error rate: {n_errors / len(y_test):.2%}")

    return errors, n_errors

class ModelTrainer:
    # Shared variables for all functions
    min_index = 0  #  Corresponds to the index that shows the minimum error rate after dropping the features using XGBoost

    def __init__(self, config_path: str = "yaml file/config.yaml", schema_filepath: str = "yaml file/schema.yaml", params_filepath: str = "yaml file/params.yaml"):
        self.config = load_yaml_config(config_path)
        self.schema = load_yaml_config(schema_filepath)
        self.params = load_yaml_config(params_filepath)
        self.df = pd.read_csv(self.config["data_validation"]["unzip_data_dir"])
        self.list_feature_drop = read_file(self.config["data_transformation"]["list_feature_drop_file"])
        self.target = 'Outcome'
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]
        self.df_train = None
        self.df_test = None

    def split_data_train_test(self, test_size) -> None:

        X_train, X_test, y_train, y_test = train_test_split(
                                            self.X,  self.y,
                                            test_size=test_size,
                                            random_state=42
                                            )
        print("test_size:", test_size)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        os.makedirs(os.path.dirname(self.config["model_trainer"]["train_data_path"]) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.config["model_trainer"]["test_data_path"]) or ".", exist_ok=True)

        # Concat X_train and y_train, then save it
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train.to_csv(self.config["model_trainer"]["train_data_path"], index=False)

        # Concat X_test and y_test, then save it
        df_test = pd.concat([X_test, y_test], axis=1)
        df_test.to_csv(self.config["model_trainer"]["test_data_path"], index=False)

        # Save test size value in a file
        os.makedirs(os.path.dirname(self.config["model_trainer"]["test_size_value_file"]) or ".", exist_ok=True)
        with open(self.config["model_trainer"]["test_size_value_file"], "w") as f:
            f.write(str(test_size) + "\n")

    def XGBoost_evaluate_feature_drop(self) -> None :

        if not os.path.exists(self.config["model_trainer"]["train_data_path"]):
            print("File train.csv doesn't exist, you need to split data")

        if not os.path.exists(self.config["model_trainer"]["test_data_path"]):
            print("File test.csv doesn't exist, you need to split data")

        self.df_train = pd.read_csv(self.config["model_trainer"]["train_data_path"])
        self.df_test = pd.read_csv(self.config["model_trainer"]["test_data_path"])

        X_train = self.df_train.drop(self.target, axis=1)
        y_train = self.df_train[self.target]

        X_test = self.df_test.drop(self.target, axis=1)
        y_test = self.df_test[self.target]

        # get n_estimators value from params.yaml
        for item in self.params["models"]:
            if item['model_name'] == 'XGBoost':
                n_estimators = item['params']["n_estimators"]
                print("n_estimators value in params.yaml:", n_estimators)

        drop_feature=[]
        error_rate = []

        for i in range(len(self.list_feature_drop) + 1):
            print("drop features", drop_feature)
            X_train_1 = X_train.drop(drop_feature, axis=1)
            X_test_1 = X_test.drop(drop_feature, axis=1)
            errors, n_errors = XGBoost(X_train_1, X_test_1, y_train, y_test, n_estimators)
            error_rate.append(n_errors)
            if i < len(self.list_feature_drop):
                drop_feature.append(self.list_feature_drop[i])
            else:
                break

        # search for the minimum error rate and the features that were removed using XGBoost
        print("")
        ModelTrainer.min_index = np.argmin(error_rate)
        print(f"The features removed using XGBoost correspond to the lowest error rate: {self.list_feature_drop[:ModelTrainer.min_index]}")
        print("For other training models, those features will be eliminated")

    def SVM_evaluate_feature_drop(self) -> None :

        if not os.path.exists(self.config["model_trainer"]["train_data_path"]):
            print("File train.csv doesn't exist, split data")

        if not os.path.exists(self.config["model_trainer"]["test_data_path"]):
            print("File test.csv doesn't exist, split data")

        self.df_train = pd.read_csv(self.config["model_trainer"]["train_data_path"])
        self.df_test = pd.read_csv(self.config["model_trainer"]["test_data_path"])

        X_train = self.df_train.drop(self.target, axis=1)
        y_train = self.df_train[self.target]

        X_test = self.df_test.drop(self.target, axis=1)
        y_test = self.df_test[self.target]

        X_train_1 = X_train.drop(self.list_feature_drop[:ModelTrainer.min_index], axis=1)
        X_test_1 = X_test.drop(self.list_feature_drop[:ModelTrainer.min_index], axis=1)

        # get SVM params value from params.yaml
        for item in self.params["models"]:
            if item['model_name'] == 'SVM':
                C = item['params']["C"]
                gamma = item['params']["gamma"]
                print("C value in params.yaml:", C)
                print("gamma value in params.yaml:", gamma)

        print("For SVM training, those features will be eliminated:", self.list_feature_drop[:ModelTrainer.min_index])
        SVM(X_train_1, X_test_1, y_train, y_test, C, gamma)

    def ANN_evaluate_feature_drop(self) -> None :

        if not os.path.exists(self.config["model_trainer"]["train_data_path"]):
            print("File train.csv doesn't exist, split data")

        if not os.path.exists(self.config["model_trainer"]["test_data_path"]):
            print("File test.csv doesn't exist, split data")

        self.df_train = pd.read_csv(self.config["model_trainer"]["train_data_path"])
        self.df_test = pd.read_csv(self.config["model_trainer"]["test_data_path"])

        X_train = self.df_train.drop(self.target, axis=1)
        y_train = self.df_train[self.target]

        X_test = self.df_test.drop(self.target, axis=1)
        y_test = self.df_test[self.target]

        X_train_1 = X_train.drop(self.list_feature_drop[:ModelTrainer.min_index], axis=1)
        X_test_1 = X_test.drop(self.list_feature_drop[:ModelTrainer.min_index], axis=1)

        print("For ANN training, those features will be eliminated:", self.list_feature_drop[:ModelTrainer.min_index])

        # Split train data into train and validation sets
        X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(
            X_train_1, y_train, test_size=0.10, random_state=42, stratify=y_train)

        os.makedirs(os.path.dirname(self.config["model_trainer"]["train_data_ANN_path"]) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.config["model_trainer"]["val_data_ANN_path"]) or ".", exist_ok=True)

        # Concat X_train_2 and y_train_2, then save it
        df_train_ANN = pd.concat([X_train_2, y_train_2], axis=1)
        df_train_ANN.to_csv(self.config["model_trainer"]["train_data_ANN_path"], index=False)

        # Concat X_val_2 and y_val_2, then save it
        df_val_ANN = pd.concat([X_val_2, y_val_2], axis=1)
        df_val_ANN.to_csv(self.config["model_trainer"]["val_data_ANN_path"], index=False)

        print(f"Split train data (train.csv) into train and validation sets: Train: {X_train_2.shape}, Validation: {X_val_2.shape}, Test: {X_test_1.shape}")

        # Normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_2)
        X_val_scaled = scaler.transform(X_val_2)
        X_test_scaled = scaler.transform(X_test_1)

        # get ANN params value from params.yaml
        for item in self.params["models"]:
            if item['model_name'] == 'ANN':
                epochs = item['params']["epochs"]
                batch_size = item['params']["batch_size"]
                hidden_layer_sizes = item['params']["hidden_layer_sizes"]
                hidden_layer_sizes = [int(x) for x in hidden_layer_sizes.split(',')]
                activation_hidden_layer = item['params']["activation_hidden_layer"]
                dropout_values = item['params']["dropout_values"]
                dropout_values = [float(x) for x in dropout_values.split(',')]
                l2 = item['params']["l2"]
                # Optimizer
                optimizer = item['params']["optimizer"]
                lr = item['params']["lr"]

        model_ANN = ANN(hidden_layer_sizes, activation_hidden_layer, l2, dropout_values, optimizer,
                        lr, epochs, batch_size, X_train_scaled, y_train_2, X_val_scaled, y_val_2)

        # Prediction
        y_pred_prob = model_ANN.predict(X_test_scaled, verbose=0)

        # Conversion with threshold at 0.5 (standard)
        y_pred_ANN = (y_pred_prob.flatten() > 0.5).astype(int)

        # Evaluation
        calculate_errors(y_pred_ANN, y_test)
        search_false_positives_or_false_negatives_in_predictions(y_pred_ANN, y_test)

        # Save ANN model and scaler
        joblib.dump(model_ANN, self.config["model_trainer"]["model_path"])
        joblib.dump(scaler, self.config["model_trainer"]['scaler_path'])
        print("")
        print("model ANN and scaler are saved")