#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# In[55]:


# Read data from current directory
df = pd.read_csv('./train.csv')
#test = pd.read_csv('./test.csv')


# In[56]:


# show data
df


# In[57]:


# Check the missing values and drop columns which have missing values more than 70%
df.isnull().sum().sort_values(ascending=False)[0:33],sns.heatmap(df.isnull(),yticklabels=False, cmap='plasma')


# In[58]:


# Not much information
df.Utilities.value_counts(dropna=False)


# In[59]:


# Drop the missing values in these columns
list1 = ['Alley','Utilities', 'PoolQC', 'Fence', 'MiscFeature']
for item in list1:
    df.drop(columns=item, inplace=True)


# In[60]:


# Get the list of object and numerical type column
def get_object_cols(df):
    return list(df.select_dtypes(include='object').columns)

def get_numerical_cols(df):
    return list(df.select_dtypes(exclude='object').columns)


# In[61]:


# list the object type column
object_cols_train = get_object_cols(df)
object_cols_train, len(object_cols_train)


# In[62]:


## mapping object type to numerical
## selecting columns for mapping dict

dict={'Y' : 1, 'N' : 2, 'Ex': 1, 'Gd' : 2, 'TA' :3, 'Fa' : 4, 'Po' : 5,  
     'GLQ' : 1, 'ALQ' : 2, 'BLQ' : 3, 'Rec' : 4, 'LwQ' : 5, 'Unf' : 6, 'NA' :7,
     'Gd' : 1 , 'Av' :2, 'Mn' : 3, 'No' :4, 'Gtl' : 1, 'Mod' : 2, 'Sev' :3,
      'Reg' : 1, 'IR1' :2, 'IR2' :3, 'IR3' :4}


# 'RL':1, 'RM':2,'FV':3,'RH':4,'C (all)':5, 'Pave':1, 'Grvl':2,'Lvl':1,'Bnk':1,'HLS':2,'Low':3,'Inside':1, 'Corner':2,'CulDSac':3,
# 'FR2':4, 'FR3':5, 'Y':1, 'N':2, 'P':3,'Norm':1, 'Feedr':2, 'PosN':3, 'Artery':4, 'RRAe':5, 'RRNn':6, 'RRAn':7, 'PosA':8,'RRNe':9 
cols=['KitchenQual','LotShape','LandSlope','HeatingQC','FireplaceQu','ExterQual','ExterCond','BsmtQual',
     'BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','CentralAir']
for i in cols:
    df[i]=df[i].map(dict)


# In[63]:


dict = {'RL' :1, 'RM' :2,'FV':3,'RH' :4,'C (all)' :5, 'Pave' :1, 'Grvl' :2,
        'Lvl' :1,'Bnk' :1,'HLS':2,'Low' :3,'Inside' :1, 'Corner' :2,'CulDSac' :3,
       'FR2' :4, 'FR3' :5, 'Y':1, 'N' :2, 'P':3,'Norm' :1, 'Feedr' :2, 'PosN' :3, 'Artery' :4,
        'RRAe' :5, 'RRNn' :6, 'RRAn' :7, 'PosA':8,'RRNe' :9, 'TA': 1, 'Fa':2, 'Gd':3, 'Po':4, 'Ex':5}
cols=['Street','GarageQual','MSZoning', 'LandContour', 'Condition1', 'Condition2', 'GarageCond']
# 
for i in cols:
    df[i]=df[i].map(dict)


# In[64]:


df


# In[65]:


df.info()


# In[66]:


# obtain object type column and numerical
object_cols_train = get_object_cols(df)
# train numerical cols
numerical_cols_train = get_numerical_cols(df)


# In[67]:


object_cols_train,len(object_cols_train)


# In[68]:


df.Neighborhood.value_counts(dropna=False)


# In[69]:


#Create the dictionary to map the label for each column
df.Neighborhood.unique()

dict = {'CollgCr':1, 'Veenker':2, 'Crawfor':3, 'NoRidge':4, 'Mitchel':5, 'Somerst':6,
       'NWAmes':7, 'OldTown':8, 'BrkSide':9, 'Sawyer':10, 'NridgHt':11, 'NAmes':12,
       'SawyerW':13, 'IDOTRR':14, 'MeadowV':15, 'Edwards':16, 'Timber':17, 'Gilbert':18,
       'StoneBr':19, 'ClearCr':20, 'NPkVill':21, 'Blmngtn':22, 'BrDale':23, 'SWISU':24,
       'Blueste':25}

cols=['Neighborhood']
for i in cols:
    df[i]=df[i].map(dict)


# In[70]:


df.LotConfig.unique()
dict={'Inside':1, 'FR2':2, 'Corner':3, 'CulDSac':4, 'FR3':5}
cols=['LotConfig']
for i in cols:
    df[i]=df[i].map(dict)


# In[71]:


df.BldgType.unique()
dict={'1Fam':1, '2fmCon':2, 'Duplex':3, 'TwnhsE':4, 'Twnhs':5}
cols=['BldgType']
for i in cols:
    df[i]=df[i].map(dict)


# In[72]:


df.HouseStyle.unique()
dict={'2Story':1, '1Story':2, '1.5Fin':3, '1.5Unf':4, 'SFoyer':5, 'SLvl':6, '2.5Unf':7,
       '2.5Fin':8}
cols=['HouseStyle']
for i in cols:
    df[i]=df[i].map(dict)


# In[73]:


df.RoofStyle.unique()
dict={'Gable':1, 'Hip':2, 'Gambrel':3, 'Mansard':4, 'Flat':5, 'Shed':6}
cols=['RoofStyle']
for i in cols:
    df[i]=df[i].map(dict)


# In[74]:


df.RoofMatl.unique()
dict={'CompShg':1, 'WdShngl':2, 'Metal':3, 'WdShake':4, 'Membran':5, 'Tar&Grv':6,
       'Roll':7, 'ClyTile':8}
cols=['RoofMatl']
for i in cols:
    df[i]=df[i].map(dict)


# In[75]:


df.Exterior1st.unique()
dict={'VinylSd':1, 'MetalSd':2, 'Wd Sdng':3, 'HdBoard':4, 'BrkFace':5, 'WdShing':6,
       'CemntBd':7, 'Plywood':8, 'AsbShng':9, 'Stucco':10, 'BrkComm':11, 'AsphShn':12,
       'Stone':13, 'ImStucc':14, 'CBlock':15}
cols=['Exterior1st']
for i in cols:
    df[i]=df[i].map(dict)


# In[76]:


df.Exterior2nd.unique()
dict={'VinylSd':1, 'MetalSd':2, 'Wd Shng':3, 'HdBoard':4, 'Plywood':5, 'Wd Sdng':6,
       'CmentBd':7, 'BrkFace':8, 'Stucco':9, 'AsbShng':10, 'Brk Cmn':11, 'ImStucc':12,
       'AsphShn':13, 'Stone':14, 'Other':15, 'CBlock':16}
cols=['Exterior2nd']
for i in cols:
    df[i]=df[i].map(dict)


# In[77]:


df.MasVnrType.unique()
dict={'BrkFace':1, 'None':2, 'Stone':3, 'BrkCmn':4}
cols=['MasVnrType']
for i in cols:
    df[i]=df[i].map(dict)


# In[78]:


df.Foundation.unique()
dict={'PConc':1, 'CBlock':2, 'BrkTil':3, 'Wood':4, 'Slab':5, 'Stone':6}
cols=['Foundation']
for i in cols:
    df[i]=df[i].map(dict)


# In[79]:


df.Heating.unique()
dict={'GasA':1, 'GasW':2, 'Grav':3, 'Wall':4, 'OthW':5, 'Floor':6}
cols=['Heating']
for i in cols:
    df[i]=df[i].map(dict)


# In[80]:


df.Electrical.unique()
dict={'SBrkr':1, 'FuseF':2, 'FuseA':3, 'FuseP':4, 'Mix':5}
cols=['Electrical']
for i in cols:
    df[i]=df[i].map(dict)


# In[81]:


df.Functional.unique()
dict={'Typ':1, 'Min1':2, 'Maj1':3, 'Min2':4, 'Mod':5, 'Maj2':6, 'Sev':7}
cols=['Functional']
for i in cols:
    df[i]=df[i].map(dict)


# In[82]:


df.GarageType.unique()
dict={'Attchd':1, 'Detchd':2, 'BuiltIn':3, 'CarPort':4, 'Basment':5, '2Types':6}
cols=['GarageType']
for i in cols:
    df[i]=df[i].map(dict)


# In[83]:


df.GarageFinish.unique()
dict={'RFn':1, 'Unf':2, 'Fin':3}
cols=['GarageFinish']
for i in cols:
    df[i]=df[i].map(dict)


# In[84]:


df.PavedDrive.unique()
dict={'Y':1, 'N':2, 'P':3}
cols=['PavedDrive']
for i in cols:
    df[i]=df[i].map(dict)


# In[85]:


df.SaleType.unique()
dict={'WD':1, 'New':2, 'COD':3, 'ConLD':4, 'ConLI':5, 'CWD':6, 'ConLw':7, 'Con':8, 'Oth':9}
cols=['SaleType']
for i in cols:
    df[i]=df[i].map(dict)


# In[86]:


df.SaleCondition.unique()
dict={'Normal':1, 'Abnorml':2, 'Partial':3, 'AdjLand':4, 'Alloca':5, 'Family':6}
cols=['SaleCondition']
for i in cols:
    df[i]=df[i].map(dict)


# In[87]:


# obtain object type column and numerical
object_cols_train = get_object_cols(df)
object_cols_train


# In[88]:


df.FireplaceQu.value_counts(dropna=False)


# In[89]:


df.isnull().sum().sort_values(ascending=False)


# In[90]:


# Set NaN to mean
for item in df.columns:
    df[item].fillna((df[item].mean()), inplace=True)


# In[91]:


# Create x and y for training a model not including object column
X=df.drop(['SalePrice'],axis=1)
Y=df['SalePrice']
#for item in object_cols_train:
    #X.drop(columns=item, inplace=True)


# In[92]:


X


# In[93]:


# split data to train and test as 80% and 20%
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=42)


# Create a model

xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)


# In[94]:


xgbr.fit(xtrain, ytrain)


# In[95]:


score = xgbr.score(xtrain, ytrain)


# In[96]:


print("Training score: ", score)


# In[97]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[98]:


# Predict output
y_pred=xgbr.predict(xtest)
y_pred


# In[99]:


ytest


# In[100]:


ytest-y_pred


# In[101]:


from sklearn.metrics import mean_absolute_error
vytest= ytest.to_numpy()
print("Mean Absolute Error:", mean_absolute_error(vytest, y_pred))
print("Mean Squared Error:", mean_squared_error(vytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(vytest, y_pred))  )


# In[102]:


# Compare actual and predicted values

fig, axs = plt.subplots(3)
axs[0].set_title('Actual value')
axs[0].plot(vytest)
axs[1].set_title('Predicted value')
axs[1].plot(y_pred)
axs[2].set_title('Compared values')
axs[2].plot(vytest)
axs[2].plot(y_pred)
for ax in axs.flat:
    ax.label_outer()


# In[103]:


# XGB with regularization
reg_xgb = xgb.XGBRegressor(colsample_bytree=0.45, gamma=0.045, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.8, n_estimators=2200,
                             reg_alpha=0.45, reg_lambda=0.85,
                             subsample=0.52, silent=1,
                             random_state =6, nthread = -1,verbosity=0)
print(reg_xgb)


# In[104]:



reg_xgb.fit(xtrain, ytrain)


# In[53]:


score = reg_xgb.score(xtrain, ytrain)
print("Training score: ", score)


# In[54]:


scores = cross_val_score(reg_xgb, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[55]:


mean = mean_absolute_error(vytest, y_pred)
mean


# In[106]:


# Predict output
y_pred=reg_xgb.predict(xtest)
y_pred


# In[107]:


ytest


# In[108]:


ytest-y_pred


# In[109]:


vytest= ytest.to_numpy()
print("Mean Absolute Error:", mean_absolute_error(vytest, y_pred))
print("Mean Squared Error:", mean_squared_error(vytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(vytest, y_pred))  )


# In[60]:


# Compare actual and predicted values
fig, axs = plt.subplots(3)
axs[0].set_title('Actual value')
axs[0].plot(vytest)
axs[1].set_title('Predicted value')
axs[1].plot(y_pred)
axs[2].set_title('Compared values')
axs[2].plot(vytest)
axs[2].plot(y_pred)
for ax in axs.flat:
    ax.label_outer()


# In[41]:


vxtrain= xtrain.to_numpy()
vytrain= ytrain.to_numpy()
vxtest= xtest.to_numpy()
vytest= ytest.to_numpy()
vxtrain.shape,vytrain.shape


# In[62]:


# Create deep learning model
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.layers import LSTM
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(32, kernel_initializer='normal',input_dim = vxtrain.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.1))
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.1))
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.1))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[54]:


from keras.callbacks import ModelCheckpoint
# Create Checkpoint to save the improved weight to computer with .hdf5
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[64]:


NN_model.fit(vxtrain, vytrain, epochs=1500, batch_size=5, validation_split = 0.2, callbacks=callbacks_list)


# In[65]:


y_pred=NN_model.predict(xtest)
vytest= ytest.to_numpy()
print("Mean Absolute Error:", mean_absolute_error(vytest, y_pred))
print("Mean Squared Error:", mean_squared_error(vytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(vytest, y_pred))  )


# In[66]:


fig, axs = plt.subplots(3)
axs[0].set_title('Actual value')
axs[0].plot(vytest)
axs[1].set_title('Predicted value')
axs[1].plot(y_pred)
axs[2].set_title('Compared values')
axs[2].plot(vytest)
axs[2].plot(y_pred)
for ax in axs.flat:
    ax.label_outer()


# In[52]:


Nmodel = Sequential()

# The Input Layer :
Nmodel.add(Dense(32, kernel_initializer='normal',input_dim = vxtrain.shape[1], activation='relu'))

# The Hidden Layers :
Nmodel.add(Dense(16, kernel_initializer='normal',activation='relu'))
# The Output Layer :
Nmodel.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
Nmodel.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
Nmodel.summary()


# In[55]:


Nmodel.fit(vxtrain, vytrain, epochs=1500, batch_size=5, validation_split = 0.2, callbacks=callbacks_list)


# In[56]:


y_pred=Nmodel.predict(vxtest)
vytest= ytest.to_numpy()
print("Mean Absolute Error:", mean_absolute_error(vytest, y_pred))
print("Mean Squared Error:", mean_squared_error(vytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(vytest, y_pred))  )


# In[57]:


fig, axs = plt.subplots(3)
axs[0].set_title('Actual value')
axs[0].plot(vytest)
axs[1].set_title('Predicted value')
axs[1].plot(y_pred)
axs[2].set_title('Compared values')
axs[2].plot(vytest)
axs[2].plot(y_pred)
for ax in axs.flat:
    ax.label_outer()


# ## Start Here

# In[42]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init


# In[43]:


class tmodel(nn.Module):
    def __init__(self,begin_size):
        super(tmodel,self).__init__()
        self.fc1 = nn.Linear(begin_size,32)
        init.normal_(self.fc1.weight)
        self.fc2 = nn.Linear(32,16)
        init.normal_(self.fc2.weight)
        self.fc3 = nn.Linear(16,1)
        init.normal_(self.fc3.weight)
        
    def forward(self,x):
        y_pred = F.relu(self.fc1(x))
        y_pred = F.relu(self.fc2(y_pred))
        y_pred = self.fc3(y_pred)
        return y_pred
ntmodel = tmodel(vxtrain.shape[1])
print(ntmodel)


# In[44]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(ntmodel)


# In[45]:


new_size_fvytrain = np.resize(vytrain,(len(vytrain),1))
fvxtrain = vxtrain.astype(np.float32)
fvytrain = new_size_fvytrain.astype(np.float32)
tytrain = torch.from_numpy(fvytrain)
txtrain = torch.from_numpy(fvxtrain)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(ntmodel.parameters(), lr=1e-4)


# In[46]:


datatrain = torch.utils.data.TensorDataset(txtrain,tytrain)


# In[47]:


traindata=torch.utils.data.DataLoader(datatrain, batch_size=5, shuffle= True)
traindata


# In[48]:


losses1 = []
for epoch in range(500):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(traindata, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = ntmodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
    losses1.append(loss.item())
print('Finished Training')


# In[49]:


fvxtest = vxtest.astype(np.float32)
fvytest = vytest.astype(np.float32)
tytest = torch.from_numpy(fvytest)
txtest = torch.from_numpy(fvxtest)


# In[50]:


y_pred2 = ntmodel(txtest)
y_pred2 = y_pred2.detach().numpy()


# In[52]:


fig, axs = plt.subplots(3)
axs[0].set_title('Actual value')
axs[0].plot(vytest)
axs[1].set_title('Predicted value')
axs[1].plot(y_pred2)
axs[2].set_title('Compared values')
axs[2].plot(vytest)
axs[2].plot(y_pred2)
for ax in axs.flat:
    ax.label_outer()


# In[53]:


print("Mean Absolute Error:", mean_absolute_error(vytest, y_pred2))
print("Mean Squared Error:", mean_squared_error(vytest, y_pred2))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(vytest, y_pred2))  )

