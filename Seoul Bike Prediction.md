https://chatgpt.com/share/68752ad9-8f54-8006-9a9d-8a5623e2641e
**Seoul Bike Prediction**




def.head / df.tail = gets to know the columns and data

df.info() - know data types which are objects i.e Categorical

df.describe - count,unique,freq,mean,25%,50%,75% max

df.isnull().sum() - doing sum because it only returns 0/1





**Feature Engineering**

df\["Date"] = pd.to\_datetime(df\["Date"])



df\["Weekday"] = df\["Date"].dt.day\_name

df\["Day"] = df\["Date"].dt.day

df\["Month"] = df\["Date"].dt.month

df\["Year"] = df\["Date"].dt.year



df.drop("Date",axis=1,inplace=True)



**Exploratory Data Analysis - EDA**



sns.pairplot(df) - it provides with all the comparison charts to make the analysis easy



In which month the bike count is less or more.

plt.figure(figsize=(8,6))

Month = df.groupby("Month").sum().reset\_index()

sns.barplot(x="Month",y="Rented Bike Count",data=Month)



same for day.

plt.figure(figsize=(10,7))

Month = df.groupby("Day").sum().reset\_index()

sns.barplot(x="Day",y="Rented Bike Count",data=Month)



Same for hour

plt.figure(figsize=(10,7))

Month = df.groupby("Hour").sum().reset\_index()

sns.barplot(x="Hour",y="Rented Bike Count",data=Month)



from this analysis we get to know which months are peak and which days bike are rented more and peak hours



plt.figure(figsize=(8,6))

sns.barplot(x="Holiday",y="Rented Bike Count",data=df)



plt.figure(figsize=(8,6))

sns.barplot(x="Seasons",y="Rented Bike Count",data=df)



**Skewed Data**

df.select\_dtypes(include=\['number']).skew().sort\_values()

Year                         -2.978262

Visibility (10m)             -0.701786

Dew point temperature(°C)    -0.367298

Temperature(°C)              -0.198326

Month                        -0.010458

Hour                          0.000000

Day                           0.007522

Humidity(%)                   0.059579

Wind speed (m/s)              0.890955

Rented Bike Count             1.153428

Solar Radiation (MJ/m2)       1.504040

Snowfall (cm)                 8.440801

Rainfall(mm)                 14.533232



negative value = left skewed data   (year)

positive value = right skewed data   (snowfall,rainfall)





**Outlier**

identified using a box plot and scatter plot
handled with help of Inter-Quartile range and many more technique



**Collinearity and Multi-Collinearity**

sns.heatmap(df.corr(),annot=True)



temperature and dew-point temperature has a co-relation value of 0.91 which is too high close to 1

if temperature increases dew-point temperature also increases



also while using a heatmap i found out collineaty issue

temperature and dew point temperature has value of 0.9 quite close to 1





also to  calculate Multi collinearity i have calculated VIF

Multicollinearity occurs when two or more independent variables in a regression model are highly correlated, which can cause issues in estimating the coefficients of the model accurately. A high VIF indicates that an independent variable is highly correlated with one or more other independent variables,



VIF = 1: There is no correlation between the independent variable and other variables.

1 < VIF < 5: Moderate correlation. Generally considered acceptable, but caution is advised.

VIF ≥ 5: Indicates high correlation with other variables, which may be problematic.

VIF ≥ 10: Suggests serious multicollinearity, and it is usually recommended to consider removing or combining variables.



so i have to drop dew point temperature



**Encoding**

**converting the categorical variable into the numerical like holiday , functioning day , seasons etc**



**df\["Holiday"] = df\["Holiday"].map({"No Holiday":0,"Holiday":1})**

**df\["Functioning Day"] = df\["Functioning Day"].map({"No":0,"Yes":1})**



**and season into four column and marking 1**



**Split data for training and testing**



X = df.drop("Rented Bike Count",axis = 1)

Y = df\["Rented Bike Count"]



x\_train,x\_test,y\_train,y\_test = train\_test\_split(X,Y, test\_size=0.2,random\_state=42)



**Standard Scaling**



sc = StandardScaler()

sc.fit(X\_train)



x\_train = sc.transform(x\_train)

x\_test = sc.transform(x\_test)



**Training ML Model**

lr = LinearRegression()

lr.fit(x\_train,y\_train)

y\_pred\_lr = lr.predict(x\_test)



rir = Ridge()

rir.fit(x\_train,y\_train)

y\_pred\_rir = rir.predict(x\_test)



lar = Lasso()

lar.fit(x\_train,y\_train)

y\_pred\_lar = rir.predict(x\_test)



svr = SVR()

svr.fit(x\_train,y\_train)

y\_pred\_svr = svr.predict(x\_test)



knnr = KNeighborsRegressor()

knnr.fit(x\_train,y\_train)

y\_pred\_knnr = knnr.predict(x\_test)



dtr = DecisionTreeRegressor()

dtr.fit(x\_train,y\_train)

y\_pred\_dtr = dtr.predict(x\_test)



rfr = RandomForestRegressor()

rfr.fit(x\_train,y\_train)

y\_pred\_rfr = rfr.predict(x\_test)



poly = PolynomialFeatures(2)

x\_train\_poly = poly.fit\_transform(x\_train)

x\_test\_poly = poly.fit\_transform(x\_test)

polyr = LinearRegression().fit(x\_train\_poly,y\_train)

y\_pred\_poly = polyr.predict(x\_test\_poly)



xgb = XGBRegressor()

xgb.fit(x\_train,y\_train)

y\_pred\_xgb = xgb.predict(x\_test)



**Evaluation**



def get\_metrics(y\_test,y\_pred,model\_name):

&nbsp;   MSE = mean\_squared\_error(y\_test,y\_pred)

&nbsp;   RMSE = np.sqrt(MSE)

&nbsp;   MAE = mean\_absolute\_error(y\_test,y\_pred)

&nbsp;   R2 = r2\_score(y\_test,y\_pred)



&nbsp;   print(f"{model\_name}:\\nMSE:{MSE}\\nRMSE:{RMSE}\\nMAE:{MAE}\\nR2:{R2}\\n\\n")



**Hyperparameter Tuning**

random forest R2 value increase from 0.902 to 0.909

XGBoost R2 value increase from 0.917 to 0.928



**Saving model using pickle**

import pickle

import os



dir = ""

model\_file\_name = "xgboost\_regessorr2\_0\_928\_v1.pkl"

model\_file\_path = os.path.join(dir,model\_file\_name)

pickle.dump(xgbr,open(model\_file\_path,"wb"))



In new Tab

model = pickle.load(open(model\_path),"rb"))

also do same for standard scaling


User Input
match the user input with the actual data of the model that is converting categorical into numerical and create function to make new columns and add values of date and seasons
at last apply standard scaling


