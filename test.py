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
from keras.models import load_model
from padelpy import from_smiles

model = load_model('Models/model_final.h5')

initial = pd.read_csv('compounds.csv')
smiles_list = initial['SMILES'].astype(str).tolist()
descriptors = from_smiles(smiles_list,output_csv='descriptors.csv',timeout=2000)
desc = pd.read_csv('descriptors.csv')
desc = desc.drop(['Name'],axis=1)
desc = desc.fillna(0)
desc_updated = desc.replace([np.inf, -np.inf], np.nan, inplace=True)
desc_updated2 = desc.fillna(0)

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
common_cols = set(desc_updated2.columns).intersection(set(X_standard.columns))
desc_updated2 = desc_updated2[common_cols].reindex(columns=X_standard.columns)
X12 = desc_updated2.values
from sklearn.preprocessing import StandardScaler
X3 = StandardScaler().fit_transform(X12)
predictions = list(model.predict(X3)[:,0])
smiles = initial['SMILES'].astype(str).tolist()
results_df = pd.DataFrame({'SMILES': (smiles), 'Predictions': predictions})
results_df.to_csv('predictions1.csv')
