'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:        [ GG_3045 ]
# Author List:    [ Gowthaman L L,Fredrick Simon P,Saran V,Manoj Kumar K ]
# Filename:       task_1a.py
# Functions:      [`data_preprocessing`, `identify_features_and_targets`, `load_as_tensors`,
#                  `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
#                  `validation_function` ]

####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

##############################################################

def data_preprocessing(task_1a_dataframe):

    le = LabelEncoder()
    columns = ["Education", "City", "Gender", "EverBenched"]
    task_1a_dataframe[columns] = task_1a_dataframe[columns].apply(le.fit_transform)
    encoded_dataframe = task_1a_dataframe
    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):

    X = encoded_dataframe.drop(columns=['LeaveOrNot'])
    y = encoded_dataframe['LeaveOrNot']
    return [X, y]

def load_as_tensors(features_and_targets):

    X, y = features_and_targets
    X_train = torch.tensor(X.values, dtype=torch.float32)
    y_train = torch.tensor(y.values, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    return [X_train, y_train, dataloader]

class Salary_Predictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(Salary_Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x

def model_loss_function():
    return torch.nn.BCELoss()

def model_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

def model_number_of_epochs():
    return 150

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):

    X_train, y_train, dataloader = tensors_and_iterable_training_data

    for epoch in range(number_of_epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            batch_y = batch_y.unsqueeze(1)
            loss = loss_function(output, batch_y)
            loss.backward()
            optimizer.step()

def validation_function(trained_model, tensors_and_iterable_training_data):

    X_train, y_train, _ = tensors_and_iterable_training_data
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_train)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_train).float().mean()
    return accuracy.item()

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########    
'''
    Purpose:
    ---
    The following is the main function combining all the functions
    mentioned above. Go through this function to understand the flow
    of the script
'''
if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and 
    # converting it to a pandas Dataframe
    task_1a_dataframe = pd.read_csv('/content/drive/MyDrive/Task_1A/task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor(features_and_targets[0].shape[1])  # Pass the input dimension here

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

   # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")