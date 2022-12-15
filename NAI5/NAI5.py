"""
Authors: Mateusz Pioch s21331, StanisÅ‚aw Dominiak s18864
The program first classifies wheat seeds (their type based on their area, perimeter, compactness,
                                          kernel length and width, asymmetry coefficient and the length
                                          of the kernel groove),
"""


import pandas as pd
import pydotplus as pdp
from IPython.display import Image
from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from six import StringIO
import torchmetrics


""" 
pytorch specific dataloading
"""
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
""" 
pytorch aspects
"""
from torch import nn
import torch.nn.functional as F
from torch import optim

"""
also lightning. Yeah it does get a bit bloated
"""
import pytorch_lightning as pl

"""
Step 1: Copy the code from the previous test, to compare it with the current mode of operation

"""
df = pd.read_csv("wheat_seeds_dataset.csv", delimiter="\t")

"""
x axis is for input data, y for output (seed class)
"""
x = df.drop('class', axis=1)
y = df['class']

"""
Split the data to approx. 67% given to training and 33% to test purposes.
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=33)


"""
Perform SVM classification with linear kernel
"""
svc = svm.SVC(kernel='linear').fit(x,y)
svc.fit(X_train, Y_train)
"""
Check accurancy of the model
"""
y_pred = svc.predict(X_test)
print("Wheat grain accuracy in SVM is: ",metrics.accuracy_score(Y_test, y_pred))



"""
Step 2: perform the same on a neural network of the pytorch lightning to compare the results
"""
#train_data, test_data, train_targets, test_targets = train_test_split(data,targets,train_size = 0.66,stratify = targets, random_state=33)

"""
Let's first create a dataset that's pytorch lightning friendly
"""

class DigitsDataset(Dataset):
  def __init__(self, data, targets):
    self.data = data
    self.targets = targets

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    x = self.data[idx]/16
    y = self.targets[idx]

    return x, y

class DigitsDatamodule(pl.LightningDataModule):
  def __init__(self, batch_size = 32):
    super().__init__()
    self.batch_size = batch_size
  def prepare_data(self):
    self.data = df
    self.targets = x
  def setup(self, stage = None):
    self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(self.data,self.targets,test_size = 0.33,stratify = self.targets                                                          )
    self.train_dataset = DigitsDataset(self.train_data,self.train_targets)
    self.test_dataset =  DigitsDataset(self.test_data, self.test_targets)
  def train_dataloader(self):
    return  DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
  def val_dataloader(self):
    return  DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
    
class DigitsModel(pl.LightningModule):
  def __init__(self, input_size, num_classes):
    super().__init__()

    self.loss_function = nn.CrossEntropyLoss()

    self.fc1 = nn.Linear(input_size, 100) ### first layer
    self.fc2 = nn.Linear(100, num_classes) ### last layer

    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()

    self.train_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')
    self.val_macro_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro')

  def forward(self, x): ### going through the network
    out = self.fc1(x) 
    out = F.relu(out) 
    out = self.fc2(out) 
   
    return out

  def configure_optimizers(self):
    optimizer =  optim.Adam(self.parameters())
    return optimizer

  def training_step(self, train_batch, batch_idx):
    inputs, labels = train_batch 


    outputs = self.forward(inputs.float()) 

    loss = self.loss_function(outputs, labels) 
    self.log('train_loss', loss)

    outputs = F.softmax(outputs, dim =1)

    self.train_acc(outputs, labels)
    self.log('train_acc', self.train_acc, on_epoch=True, on_step= False)

    self.train_macro_f1(outputs, labels)
    self.log('train_macro_f1', self.train_macro_f1, on_epoch=True, on_step= False)

    return loss

  def validation_step(self, val_batch, batch_idx):
    inputs, labels = val_batch 


    outputs = self.forward(inputs.float()) 
    loss = self.loss_function(outputs, labels) 

    self.log('val_loss', loss)

    outputs = F.softmax(outputs, dim =1)

    self.val_acc(outputs, labels)
    self.log('val_acc', self.val_acc, on_epoch=True, on_step= False)

    self.val_macro_f1(outputs, labels)
    self.log('val_macro_f1', self.val_macro_f1, on_epoch=True, on_step= False)

    return loss


data_module = DigitsDatamodule()

"""
limiting to 20 epochs so that it doesn't take forever
"""
trainer = pl.Trainer(
    max_epochs = 20
)

model = DigitsModel(64,10)

"""
time to train
"""
trainer.fit(model, data_module)


"""
check out this confusion matrix!
"""

true_labels = []

pred_labels = []
pred_scores = []

cm = confusion_matrix(true_labels, pred_labels)

cm_plot = ConfusionMatrixDisplay(cm)
cm_plot.plot()



















