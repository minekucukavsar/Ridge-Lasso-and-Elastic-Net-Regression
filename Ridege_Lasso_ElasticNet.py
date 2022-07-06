###Ridge_Regression###
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

df.head()
df.shape
X.loc[0:10,:]
#Our main goal is to estimate the "Salary" .

#Ridge Regression
ridge_model=Ridge(alpha=0.1).fit(X_train,y_train)
ridge_model.get_params()
#Out{'alpha': 0.1,'copy_X': True,'fit_intercept': True,'max_iter': None,'normalize': False,'random_state': None,'solver': 'auto','tol': 0.001}

ridge_model.coef_
ridge_model.intercept_ #-4.5786269057224445
#array([ -1.77435737,   8.80240528,   7.29595605,  -3.33257639,
       # -2.08316481,   5.42531283,   7.58514945,  -0.13752764,
       #-0.20779701,  -0.60361067,   1.7927957 ,   0.72866408,
       #-0.68710375,   0.26153564,   0.26888652,  -0.52674278,
       #112.14640272, -99.80997876, -48.07152768])

#Different coefficients will be generated for different lambda values, and the models established on each lambda value and the resulting errors will be discussed.

ridge_model=Ridge(alpha=0.2).fit(X_train,y_train)
ridge_model.coef_#changed alpha=0.2
ridge_model.intercept_#-4.511933538563767


#How coefficient values change for different lambda values
lambda_values=10**np.linspace(10,-2,100)*0.5
ridge_model=Ridge()
coefficients_=[]

for i in lambda_values:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    coefficients_.append(ridge_model.coef_)

lambda_values

ax=plt.gca()
ax.plot(lambda_values,coefficients_)
ax.set_xscale("log")
plt.show()

#Prediction
ridge_model=Ridge().fit(X_train,y_train)
y_pred=ridge_model.predict(X_train)
y_pred[0:10]
y_train

#TrainError
RMSE=np.sqrt(mean_squared_error(y_train,y_pred))

np.sqrt(np.mean(-cross_val_score(ridge_model,X_train,y_train,cv=10,scoring="neg_mean_squared_error")))#351.39315856063536
#Lower error is always better not a good argument every time. The important thing is to get a more accurate error.

#TestError
y_pred=ridge_model.predict(X_test)
RMSE=np.sqrt(mean_squared_error(y_test,y_pred))
RMSE#356.80829057302293

#To sum up, we built a model and the model used with its default value without parameter optimization .


###Model_Tuning###
lambda_values=10**np.linspace(10,-2,100)*0.5
lambda_values_2=np.random.randint(0,1000,100)

#lambda_values
ridgeCV=RidgeCV(alphas=lambda_values,scoring="neg_mean_squared_error",cv=10,normalize=True)
ridgeCV.fit(X_train,y_train)

ridgeCV.alpha_#0.7599555414764666, optimum parameter

#finalModel

ridge_Tuned=Ridge(alpha=ridgeCV.alpha_).fit(X_train,y_train)
y_pred=ridge_Tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#356.8583047271514

#lambda_values_2
ridgeCV=RidgeCV(alphas=lambda_values_2,scoring="neg_mean_squared_error",cv=10,normalize=True)
ridgeCV.fit(X_train,y_train)

ridgeCV.alpha_#13, optimum parameter

#finalModel

ridge_Tuned=Ridge(alpha=ridgeCV.alpha_).fit(X_train,y_train)
y_pred=ridge_Tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#356.3258325177

###Lasso_Regression###
df=pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

Lasso_model=Lasso().fit(X_train,y_train)

Lasso_model.intercept_ #-5.5874506773358235
Lasso_model.coef_
#array([-1.74875691e+00,  8.59204135e+00,  6.67993798e+00, -3.06715333e+00,
       #-1.91843070e+00,  5.32372890e+00,  8.39184117e+00, -1.63172447e-01,
       #-8.22311277e-02, -3.93602861e-01,  1.71118530e+00,  6.55730545e-01,
       #-6.48379405e-01,  2.59815358e-01,  2.73041157e-01, -4.41440454e-01,
       #8.54474011e+01, -9.59701213e+01, -2.13086605e+01])

#coefficients corresponding to different lambda values
lasso=Lasso()
coefs=[]
alphas=np.random.randint(0,1000,10)
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)

ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
plt.show()


lasso=Lasso()
coefs=[]
alphas=10**np.linspace(10,-2,100)*0.5
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)

ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
plt.show()

###Prediction###
Lasso_model.get_params()
#{'alpha': 1.0,'copy_X': True,'fit_intercept': True,'max_iter': 1000,'normalize': False,'positive': False,'precompute': False,'random_state': None,'selection': 'cyclic','tol': 0.0001,'warm_start': False}

Lasso_model.predict(X_train)[0:5]
#array([377.26270596, 786.51524513, 495.14140718, 117.19492966,429.04228506])

Lasso_model.predict(X_test)[0:5]
#array([ 609.18826367,  696.96810702, 1009.06157391,  412.22773375,409.25851712])

y_pred=Lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#356.0975884554034 our test error.But we have not yet optimized the model.

#Model_Tuning#
Lasso_cv_model=LassoCV(cv=10,max_iter=100000).fit(X_train,y_train)#no lambda value entered.
Lasso_cv_model.alpha_
#563.4670501833854

Lasso_tuned=Lasso().set_params(alpha=Lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred=Lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))#373.5957225069794

#same process with different lambda values
Lasso_cv_model=LassoCV(alphas=lambda_values,cv=10,max_iter=100000).fit(X_train,y_train)
Lasso_cv_model.alpha_#201.85086292982749
Lasso_tuned=Lasso().set_params(alpha=Lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred=Lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))#363.6832708037447

pd.Series(Lasso_tuned.coef_,index=X_train.columns)
#The zero coefficients we see in this output have actually turned into meaningless variables.
#That is, when their coefficients equals to zero, their effect on the salary becomes zero.
#To sum up, variables that come with a value of zero here are not effective in determining a player's salary.



###ElasticNet_Regression###

df=pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

enet_model=ElasticNet().fit(X_train,y_train)
enet_model.coef_
#array([ -1.86256172,   8.70489065,   5.10426375,  -2.89875799,
       # -1.28642985,   5.24343682,   6.04480276,  -0.14701495,
       #-0.21566628,  -0.7897201 ,   1.80813117,   0.80914508,
       #-0.61262382,   0.26816203,   0.27172387,  -0.36530729,
       #19.2186222 , -31.16586592,   8.98369938])

enet_model.intercept_
#-6.465955602111762

#Predict
enet_model.predict(X_train)[0:10]
enet_model.predict(X_test)[0:10]

y_pred=enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model_Tuning
enet_cv_model=ElasticNetCV(cv=10).fit(X_train,y_train)
enet_cv_model.alpha_#5230.7647364798695
enet_cv_model.intercept_#-38.51940558394301

#FinalModel
enet_tuned=ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train,y_train)
y_pred=enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))#394.1528056321879

?ElasticNet
#l1_ratio : float, default=0.5
#The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
#``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it is an L1 penalty.
# For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.
















