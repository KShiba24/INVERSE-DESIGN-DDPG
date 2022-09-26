import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional
import torch.utils.data
from torch.autograd import Variable
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.model_selection import train_test_split
import csv
from matplotlib.font_manager import FontProperties
import cloudpickle


#4 layer NN
class Net(torch.nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(900, 700)
    self.fc2 = torch.nn.Linear(700, 600)
    self.fc3 = torch.nn.Linear(600, 500)
    self.dropout = torch.nn.Dropout(0.3)

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = self.dropout(x)
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x


num_epochs = 8000
df1 = pd.read_csv('dataset/RCM_input_data.csv', header=None)   #coulomb matrix data    
df2 = pd.read_csv('dataset/RCM_label_data.csv', header=None)   #spectra data
df3 = pd.read_csv('dataset/wavelengeth.csv')
wavelength_abs = np.array(df3)
X_train = np.array(df1)/1000
y_train = np.array(df2)/1000


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
x_tensor = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
x_test = Variable(torch.from_numpy(X_test).float(), requires_grad=True)
y_tensor = Variable(torch.from_numpy(y_train).float())
y_test = Variable(torch.from_numpy(y_test).float())



net = Net()
net.train()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
criterion_MAE = torch.nn.L1Loss()
kldiv = torch.nn.KLDivLoss()

def wavelength(pre, tra):
  wavelength_peak_pre = []
  wavelength_peak_tra = []
  for s in range(len(pre)):
    pre_wave = np.argmax(pre[s])
    tra_wave = np.argmax(tra[s])
    wavelength_peak_pre.append(pre_wave)
    wavelength_peak_tra.append(tra_wave)
  return wavelength_peak_pre, wavelength_peak_tra




outputs_test = net(x_test)
loss_test = criterion(outputs_test, y_test) 
print(loss_test)
pre_predicted = outputs_test.detach().numpy().copy() * 1000
pre_predicted0 = pre_predicted[0]
pre_predicted1 = pre_predicted[1]
pre_predicted2 = pre_predicted[2]
pre_predicted3 = pre_predicted[3]

epoch_loss = []
epoch_loss_MAE = []
epoch_loss_kl = []
for epoch in range(num_epochs):
  if epoch % 1000 == 0:
    print('time: ',epoch)
  outputs = net(x_tensor)
  loss = criterion(outputs, y_tensor)
  loss_MAE = criterion_MAE(outputs, y_tensor)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  epoch_loss_MAE.append(loss_MAE.data.numpy().tolist())
  epoch_loss.append(loss.data.numpy().tolist())


outputs_train = net(x_tensor)
predicted_spectrum_train = outputs_train.detach().numpy().copy() * 1000
train_wave_pre, train_wave_target  = wavelength(predicted_spectrum_train, y_train)



outputs_test = net(x_test)
loss_test = criterion(outputs_test, y_test) 
print(loss_test)

predicted_spectrum = outputs_test.detach().numpy().copy() * 1000
test_wave_pre, test_wave_target = wavelength(predicted_spectrum, y_test)




predicted_spectrum0 = predicted_spectrum[0]
predicted_spectrum1 = predicted_spectrum[1]
predicted_spectrum2 = predicted_spectrum[2]
predicted_spectrum3 = predicted_spectrum[3]

target_spectrum = y_test.detach().numpy().copy() * 1000
target_spectrum0 = target_spectrum[0]
target_spectrum1 = target_spectrum[1]
target_spectrum2 = target_spectrum[2]
target_spectrum3 = target_spectrum[3]



#plot loss curve 
fig1 = plt.figure()
ax = fig1.add_subplot()
ax.plot(list(range(len(epoch_loss))), epoch_loss, label='MSE')
ax.plot(list(range(len(epoch_loss))), epoch_loss_MAE, label='MAE')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend(loc = 'upper right')

#plot predicted 
fig2 = plt.figure()
ax = fig2.add_subplot()
ax.plot(wavelength_abs, pre_predicted0, label="before_learning")
ax.plot(wavelength_abs, predicted_spectrum0, label="predict")
ax.plot(wavelength_abs, target_spectrum0, label="target")
ax.set_xlabel('wavelength[nm]')
ax.set_ylabel('Intensity [a.u.]')
ax.legend(loc = 'upper left')

#plot predict
fig3 = plt.figure()
ax = fig3.add_subplot()
ax.plot(wavelength_abs, pre_predicted1, label="before_learning")
ax.plot(wavelength_abs, predicted_spectrum1,label='predict')
ax.plot(wavelength_abs, target_spectrum1, label='target')
ax.set_xlabel('wavelength[nm]')
ax.set_ylabel('Intensity [a.u.]')
ax.legend(loc = 'upper left')

#plot predict
fig4 = plt.figure()
ax = fig4.add_subplot()
ax.plot(wavelength_abs, pre_predicted2, label="before_learning")
ax.plot(wavelength_abs, predicted_spectrum2, label='predict')
ax.plot(wavelength_abs, target_spectrum2, label='target')
ax.set_xlabel('wavelength[nm]')
ax.set_ylabel('Intensity [a.u.]')
ax.legend(loc = 'upper left')

# fig5 = plt.figure()
# ax = fig5.add_subplot()
# ax.plot(wavelength, pre_predicted3, label="before_learning")
# ax.plot(wavelength, predicted_spectrum3, label='predict')
# ax.plot(wavelength, target_spectrum3, label='target')
# ax.set_xlabel('wavelength[nm]')
# ax.set_ylabel('Intensity [a.u.]')
# ax.legend(loc = 'upper left' )

# fig8 = plt.figure()
# ax = fig8.add_subplot()
# y_tensor = y_tensor * 1000
# for i in range(len(y_tensor)):
#   ax.plot(wavelength[150:430], y_tensor[i][150:430], label='predict')
#   ax.set_xlabel('wavelength[nm]')
#   ax.set_ylabel('Intensity [a.u]')

plt.show()
