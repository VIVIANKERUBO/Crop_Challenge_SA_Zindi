import numpy as np
import torch
import pandas as pd
import utils
from utils import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_data(input_csv):
  s1 = pd.read_csv(input_csv)
  s1 = s1.to_numpy(s1)
  # assign X and y
  X = s1[:, 1:20] 
  y = s1[:, -2]
  

  # normalize 
  scaler = StandardScaler()
  X = scaler.fit_transform(X))
  #X = scaler.transform(X)) use when making predictions

  #split to train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

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

def train_epoch(model, optimizer, criterion, dataloader, device):
   model.train()
   losses = list()
   with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
     for idx, batch in iterator:
       optimizer.zero_grad()
       x, y = batch
       x = x.float()
       x, y = x.to(device), y.to(device)
       output = model.forward(x)
       loss =  criterion(output, torch.max(y,1)[1])
       loss.backward()
       optimizer.step()
       losses.append(loss)
   return torch.stack(losses)

def test_epoch(model,criterion, dataloader,device):
  model.eval()
  with torch.no_grad():
    losses = list()
    y_true_list = list()
    y_pred_list = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
      for idx, batch in iterator:
        x_test, y_true = batch
        x_test = x_test.float()
        y_pred = model.forward(x_test.to(device))
        loss = criterion(y_pred, y_true.to(device))
        losses.append(loss)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred.argmax(-1))
  return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)

def training(input_dir, output_dir, model, epochs, trainloader, testloaderm use_gpu = False):

  # choice of the device
  if torch.cuda.is_available() and use_gpu:
    device = "cuda"
  else:
    device = "cpu"
  device = torch.device(device)


  criterion = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = Adam(model.parameters(), lr=0.001)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)

  for epoch in range(epochs):
    
    train_loss = train_epoch(model, optimizer, criterion, trainloader, device)
    train_loss = train_loss.cpu().detach().numpy()[0]
    test_loss, y_true, y_pred  = test_epoch(model, optimizer, criterion, testloader, device)
    test_loss = test_loss.cpu().detach().numpy()[0]
    
    scores = metrics(y_true_.cpu(), y_pred.cpu())
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])

    print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
    scores["epoch"] = epoch
    scores["trainloss"] = train_loss
    scores["testloss"] = test_loss
    log.append(scores)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "trainlog.csv"))
    

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train and evaluate sentinel 1 and sentine2 satellite time series data
                                                 'with Inception time models'
                                                 'This script trains a model on training dataset'
                                                 'evaluates performance on a validation data'
                                                 'and stores progress and model paths in --logdir.')
    parser.add_argument('-i','--input_dir', action="store", help='input directory', type=str)  
    parser.add_argument('-d','--output_dir', action="store", help='output directory', type=str)
    parser.add_argument(
        '-b','--batchsize', default=64, action="store", help='batch size', type=int)

    parser.add_argument('-e','--epochs', default=10, action="store", help='partition id for grouped datasets', type=int)   
    parser.add_argument(
        '-g','--use_gpu', default="False", action="store", type=lambda x: (str(x).lower() == 'true'), help='select whether to use GPU or not.')
    args = parser.parse_args() 

    return args

if __name__ == "__main__":    
    args = parse_args()
    
    training(args.input_dir, args.output_dir, args.batchsize,args.epochs, args.use_gpu)
  

  


