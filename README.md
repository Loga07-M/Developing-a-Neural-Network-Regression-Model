# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: LOGA SRI M

### Register Number: 212225230153

```python

#%% packages
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% import data
df=pd.read_csv(r"C:\Users\admin\OneDrive\Documents\DL\venv\Exp-1.csv")
df

# %% separate independent / dependent features
x=df[["Input"]].values
y=df[["Output"]].values

# %% Train / Test Split
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=52) # check if requires_grad is true, false if not directly specified


#%% scale the data
scale1=MinMaxScaler()
scale2=MinMaxScaler()
xt=scale1.fit_transform(xt)
xst=scale1.transform(xst)


xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)


# %% simple network
class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)


        )
    def forward(self, x):
        return self.network(x)

# %% create model
model=neuralNetwork()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)



# %% training
epochs=5000
losses=[]

for i in range(epochs):
    optimizer.zero_grad()
    predictions=model(xt)
    loss=criterion(predictions,yt)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"epoch: {i}, loss: {loss.item()}")
        losses.append(loss.item())
# %% prediction
new_data=torch.FloatTensor([[16]])
new_data=torch.Tensor(scale1.transform(new_data))
predictions=model(new_data)
print(predictions.item())
# %% test
xst = torch.FloatTensor(xst)
yst = torch.FloatTensor(yst)
with torch.no_grad():
    predictions_l=model(xst)
    test_loss=criterion(predictions_l,yst)
    print(f"test loss: {test_loss.item()}")

# %% plot
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.show()
```





### Dataset Information
Include screenshot of the generated data
<img width="778" height="651" alt="image" src="https://github.com/user-attachments/assets/3c2d6a00-3fc5-4fc6-8d63-8e09c8f730a4" />

### OUTPUT
<img width="785" height="713" alt="image" src="https://github.com/user-attachments/assets/99ef9862-c4c1-41c2-ae42-20e06fe0b763" />


### Training Loss Vs Iteration Plot
Include your plot here

<img width="778" height="597" alt="image" src="https://github.com/user-attachments/assets/63faf463-693c-4095-a78d-faf0cba1a5bd" />

### New Sample Data Prediction
Include your sample input and output here

<img width="407" height="53" alt="image" src="https://github.com/user-attachments/assets/66bdd043-c184-4e8d-a508-746fb51e2b78" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
