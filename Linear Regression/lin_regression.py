#%%
import pandas as pd
#import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
#%%
data=pd.read_csv("Salary_Data.csv")
data.tail()
data.info()
data.describe()
# %%
plt.figure(figsize=(12,6))
# sns.pairplot(data)
sns.pairplot(data,x_vars=["YearsExperience"],y_vars=["Salary"],size=7,kind="scatter")
plt.xlabel("years")
plt.ylabel("salary")
plt.title("salary prediction")
plt.show()

# %%

x = data.iloc[:,:-1]
y = data.iloc[:,1]

x.head()
y.head()

X_train , X_test , y_train , y_test = train_test_split(x,y,train_size=0.8,random_state = 10)


my_model = LinearRegression()
my_model.fit(X_train,y_train)
my_model.score(X_train,y_train)
my_model.score(X_test,y_test)


# %%
#hena kant el data bta3t train
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, my_model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# %%
#data bta3t el test b2a 
y_pred = my_model.predict(X_test)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%
#plotting actual and predicted values
c = [i for i in range(1,len(y_test)+1)]
plt.plot(c,y_test,color="r",linestyle='-')
plt.plot(c,y_pred,color="b",linestyle='-')
plt.xlabel("salary")
plt.ylabel("yearsExperience")
plt.title("Prediction")
plt.show()
# %%
#plotting the error
c = [i for i in range(1,len(y_test)+1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()




# %%
# calculate Mean square error
#nesbet error
mse = mean_squared_error(y_test,y_pred)
# Calculate R square vale
#accuracy of model 
rsq = r2_score(y_test,y_pred)
print('mean squared error :',mse)
print('r square :',rsq)
# %%
print('Intercept of the model:',my_model.intercept_)
print('Coefficient of the line:',my_model.coef_)
#7at el X hena b ay value b2a elly howa years of exp
#w byshof b2a el expected salary kam (4.5 years)
y_hat = 9356 * 4.5 +  26089 
print(y_hat)
# %%
