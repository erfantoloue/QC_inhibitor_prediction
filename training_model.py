import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import regularizers

#Selecting Data 

first = pd.read_csv('Data/InhibitorsDescriptors.csv')
m = first.drop(columns=['Name'])
m1 = m.fillna(0)
X = m1
data = pd.read_csv('Data/InhibitorsandPIC50.csv')

Y = data['pIC50']
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
selected_columns = selection.fit(X).get_support(indices=True)
X_selected = X.iloc[:, selected_columns]

scaler = StandardScaler()
X_standard = pd.DataFrame(scaler.fit_transform(X_selected), columns=X_selected.columns)



X_train, X_test, Y_train, Y_test = train_test_split(X_standard, Y, test_size=0.2)
additionaltest1 = X_test[:20]
additionaltest2 = X_train[:10]
ad_test_X = np.concatenate((additionaltest1, additionaltest2), axis=0)
additionaltest1x = Y_test[:20]
additionaltest2x = Y_train[:10]
ad_test_y= np.concatenate((additionaltest1x, additionaltest2x), axis=0)
X_train = X_train[10:]
X_test = X_test[20:]
Y_train = Y_train[10:]
Y_test = Y_test[20:]

# Model Development
model = keras.Sequential([
    layers.Reshape(target_shape=(19,47,1), input_shape=(893,)),
    layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='sigmoid'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform', activation='sigmoid'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform', activation='sigmoid'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    
    layers.Dense(655, activation='sigmoid'),
    layers.Dropout(0.5),
    layers.Dense(616, activation='sigmoid'),
    layers.Dropout(0.5),

    layers.Dense(1)
])
print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mean_squared_error','mean_absolute_error']
)


history = model.fit(
    X_train, Y_train,
    epochs=250,
    batch_size=32,
    validation_data=(X_test, Y_test)

)
model.save('Models/new_model.h5')
# Evaluating the model
results = model.evaluate(X_test, Y_test, verbose=0)
print('####################')
print(f"Test loss: {results[0]}, Test MAE: {results[1]}, Test MSE: {results[2]}")
print('Testing the model on a random part of initial data that has not been selected for training')
print('####################')
results2 = model.evaluate(ad_test_X, ad_test_y, verbose=0)
print('Testing on a random part of initial data that has not been selected for training or test at all')
print(f"Test loss2: {results2[0]}, Test MAE2: {results2[1]}, Test MSE2: {results2[2]}")
print('####################')

