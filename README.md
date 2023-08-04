# QC_inhibitor_prediction
Made for a paper submission on "Predicting glutaminyl cyclase inhibitors using deep learning" by Hamid Reza Kalhor and M. Erfan Toloue Sadegh Azadi <br>
Correspondent email : kalhor@sharif.edu

------

Requirements:
- python
- numpy
- pandas
- rdkit
- padelpy
- sklearn
- tensorflow

------
## Cloning the environment
Run <br>
`git clone https://github.com/erfantoloue/QC_inhibitor_prediction.git`

## Training the model
The trained model is saved as `model_final.h5` in the **Models** directory. However the training process can be repeated by running
 <br>`python training.py` <br>
The newly trained model will be saved as `new_model.h5` in the **Models** directory and can be used for prediction.

## Using the model
In order to use the model to predict the pIC50 for novel compounds, simply import SMILES of structures in the `compounds.csv` file. After that run
 <br> `python test.py` <br>
The results will be saved in the `predictions.csv` file.
## Predicted datasets
The predicitons are available in the **Predictions** directory. These compounds are filtered by Lipinski's Rule of 5.
