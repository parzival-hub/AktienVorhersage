import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import LSTM, Dense, Activation
from tensorflow import optimizers, keras

look_back = 30
input_shape_size = 1


def main():
    df_train = pd.read_csv("AAPL_2004.csv")
    df_test = pd.read_csv("AAPL_Test.csv")
    df_2022 = pd.read_csv("AAPL_2022.csv")

    create_new = False
    if create_new:
        model = create_model()
        model = train_model(model, df_train)
        model.save("model")
    else:
        model = keras.models.load_model('model')

    df_train["abs_changes"] = abs(df_train["DailyChanges"])
    print("Avg changes: ", df_train["abs_changes"].mean())

    df_train["change_predictions"] = get_predictions(model, df_train)
    df_test["change_predictions"] = get_predictions(model, df_test)
    df_2022["change_predictions"] = get_predictions(model, df_2022)

    print("Train:")
    df_train["diff"] = abs(df_train["DailyChanges"] - df_train["change_predictions"])

    print(df_train["diff"].mean())
    print("Test")
    df_test["diff"] = abs(df_test["DailyChanges"] - df_test["change_predictions"])
    print(df_test["diff"].mean())

    # df_test["rightDir"] = True if df_test["DailyChanges"] < 0 & df_train["change_predictions"] < 0 | df_train[
    #     "change_predictions"] > 0 & df_train["change_predictions"] > 0 else False

    up = 0
    down = 0
    rightDirection = 0
    for (a, b) in zip(df_test["DailyChanges"], df_test["change_predictions"]):
        if abs(a + b) == abs(a) + abs(b):
            rightDirection += 1
        if b > 0:
            up += 1
        elif b < 0:
            down += 1
    print("[TEST] Richtiges Vorzeichen: ", rightDirection / len(df_test["DailyChanges"]))
    print("UP: %d   Down: %d" % (up, down))
    
    # 2022 Data
    rightDirection = 0
    for (a, b) in zip(df_test["DailyChanges"], df_2022["change_predictions"]):
        if abs(a + b) == abs(a) + abs(b):
            rightDirection += 1
    print("[2022] Richtiges Vorzeichen: ", rightDirection / len(df_2022["DailyChanges"]))

    # Random Choice
    rightDirection = 0
    for (a, b) in zip(df_test["DailyChanges"], [random.choice([-1, 1]) for _ in range(len(df_test["DailyChanges"]))]):
        if abs(a + b) == abs(a) + abs(b):
            rightDirection += 1
    print("[RND] Richtiges Vorzeichen: ", rightDirection / len(df_test["DailyChanges"]))

    # Train Data
    rightDirection = 0
    for (a, b) in zip(df_train["DailyChanges"], df_train["change_predictions"]):
        if abs(a + b) == abs(a) + abs(b):
            rightDirection += 1
    print("[Train] Richtiges Vorzeichen: ", rightDirection / len(df_train["DailyChanges"]))
    # draw_graph(df_test)


def get_predictions(model, df):
    changes = np.array(df[["DailyChanges"]])
    X = []
    # Nur train_size Elemente werden berücksichtigt
    for i in range(look_back, len(changes) - 1):
        X.append(np.array(changes[i - look_back:i]))
    X = np.array(X).reshape(-1, look_back, 1)
    predictions = model.predict(X)
    df.drop(df.index[-(look_back + 1):], inplace=True)
    predictions = predictions.reshape(-1)
    return predictions


def draw_graph(df):
    predictions = df["change_predictions"]
    plt.plot(df["Date"], df["Open"] * (1 + predictions), label="Prediction")
    plt.plot(df["Date"], df["Close"], label="Real chart")
    plt.legend()
    plt.show()


def train_model(model, pdf):
    X = []
    Y = []
    changes = pdf["DailyChanges"]
    # Nur train_size Elemente werden berücksichtigt
    for i in range(look_back, len(changes) - 1):
        X.append(np.array(changes[i - look_back:i]))
        Y.append(changes[i])

    # Damit es auf unseren Shape passt
    X = np.array(X).reshape(-1, look_back, 1)
    Y = np.array(Y)

    model.fit(X, Y, batch_size=32, epochs=30)
    return model


def create_model():
    lstm_input = Input(shape=(look_back, input_shape_size), name='lstm_input')
    inputs = LSTM(21, name='first_layer')(lstm_input)
    inputs = Dense(16, name='first_dense_layer')(inputs)
    inputs = Dense(1, name='second_dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    mymodel = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(learning_rate=0.0008)
    mymodel.compile(optimizer=adam, loss='mse')
    return mymodel


main()
