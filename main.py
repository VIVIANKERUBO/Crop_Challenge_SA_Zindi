import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def read_data(input_csv):

  # assign X and y
  X = 
  y =

  # normalize 
  scaler = StandardScaler()
  X = scaler.fit_transform(X))
  #X = scaler.transform(X)) use when making predictions

  #split to train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  #convert to torch
  X_train =  torch.from_numpy(X_train)
  X_test = torch.from_numpy(X_test)
  y_train = torch.from_numpy(y_train)
  y_test = torch.from_numpy(y_test)

  # Dataloader
  #train
  train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

  #test
  test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

  return train_loader, test_loader

  


