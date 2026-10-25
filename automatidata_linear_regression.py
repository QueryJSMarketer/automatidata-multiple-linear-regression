import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import missingno as msno
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
images_dir = current_dir / "images"
print("Ruta de la carpeta images:", images_dir)
images_dir.mkdir(parents=True, exist_ok=True)

file_path = current_dir / "2017_Yellow_Taxi_Trip_Data.csv"
df0 = pd.read_csv(file_path)
df = df0.copy()
print(f'The shape of dataset: {df.shape}')
print()
df.info()

#Data cleaning
df.isna()
print(df.drop_duplicates().shape)
print(df.isna().sum())
print(df.head(5))

#visualization of clean data
msno.matrix(df.head(3000))
plt.savefig(os.path.join(images_dir, "missingno_matrix.png"), bbox_inches="tight", dpi=150)
plt.close()
msno.bar(df)
plt.savefig(os.path.join(images_dir, "missingno_bar.png"), bbox_inches="tight", dpi=300)
plt.close()
msno.dendrogram(df)
plt.savefig(os.path.join(images_dir, "missingno_dendrogram.png"), bbox_inches="tight", dpi=300)
plt.close()
#plt.show()

#Descriptive analysis of the data
print(df.describe())

#Convert pickup & dropoff columns to datetime
#First check the format of the data
print(df['tpep_pickup_datetime'][0])

#Convert datetime columns to datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
print('Data type of pickup column: ', df['tpep_pickup_datetime'].dtype)
print('Data type of dropoff column: ', df['tpep_dropoff_datetime'].dtype)
print(df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head())

#Create duration column
#Create a new column called duration that represents the total number of minutes that each taxi ride took.
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds().div(60).round().astype(int)
print(df['duration'])

#Outliers
#Check the info
print(df.info())

'''
Keeping in mind that many of the features will not be used to fit your model, 
the most important columns to check for outliers are likely to be:
trip_distance
fare_amount
duration

'''
#Plot a box plot for each feature: trip_distance, fare_amount, duration.
fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outliers detection')
sns.boxplot(ax=axes[0], x=df['trip_distance'])
sns.boxplot(ax=axes[1], x=df['fare_amount'])
sns.boxplot(ax=axes[2], x=df['duration'])
plt.savefig(os.path.join(images_dir, "boxplots_outliers.png"), bbox_inches="tight", dpi=300)
plt.show()

'''
All 3 variables contain outliers; 
the trip_distance variable has the highest number of extreme values, 
points outside the whiskers of the box plot are visibl, 
located primarily on the right side, 
with very few on the left side. 
For the trip_distance variable, 
typical values fall within the 0 to 5-mile range, 
but there are outliers that exceed this range, 
extending up to 35 miles. 
A 35-mile trip in New York is likely to be trouble-free, 
but there are instances with values of 0—or close to 0—for fares or trip durations that do not add up and require investigation. 
The results of this examination are not normal, 
as there are instances of 0 distance and 0 duration—cases where the only logical conclusion is that the trip was either 
cancelled before it began or that a system error occurred. 
In the chart for the variable fare_amount, 
negative values can be seen (they appear below zero); this is impossible, as no passenger could have paid a negative amount, 
this is often the result of refunds, adjusted charges,
or data entry errors that must be filtered out or cleaned up before training any model.
'''

#Imputations
#trip_distance outliers
print(sorted(set(df['trip_distance']))[:10])

'''
The fact that the values range from 0.0 to 0.01 and so on indicates that the system records the distance traveled with high precision. 
Therefore, many of these zeros or infinitesimal distances do, in fact, correspond to actual, 
extremely short trips—rather than necessarily being "bad data"—though a few extreme cases may still require a review of their fares.
'''

print(sum(df['trip_distance']==0))

'''
148 out of ~23,000 rides is relatively insignificant. 
You could impute it with a value of 0.01, but it's unlikely to have much of an effect on the model. 
Therefore, the trip_distance column will remain untouched with regard to outliers.
'''

#fare_amount outliers
print(df['fare_amount'].describe())

'''
The range of values in the fare_amount column is large and the extremes don't make much sense.

Low values: Negative values are problematic. 
Values of zero could be legitimate if the taxi logged a trip that was immediately canceled.

High values: The maximum fare amount in this dataset is nearly $1,000, which seems very unlikely. 
High values for this feature can be capped based on intuition and statistics. The interquartile range (IQR) is $8. 
The standard formula of Q3 + (1.5 * IQR) yields $26.50. That doesn't seem appropriate for the maximum fare cap. In this case, 
we'll use a factor of 6, 
which results in a cap of $62.50.

'''

df.loc[df['fare_amount'] < 0, 'fare_amount'] = 0
print(df['fare_amount'].describe())

#Now impute the maximum value as Q3 + (6 * IQR).
def outlier_imputer(column_list, iqr_factor):
    for col in column_list:
        minimum = df.loc[df[col] < 0, col] = 0
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + (iqr_factor * iqr)
        print(col)
        print('minimum:', minimum)
        print('q3:', q3)
        print('q1:', q1)
        print('upper_limit:', upper_limit)
        print()
        df.loc[df[col] > upper_limit, col] = upper_limit
        print(df[col].describe())
        print()

#duration outliers
print(df['duration'].describe())
print(outlier_imputer(['duration'], 6))

#Feature engineering
#Create mean_distance column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
print(df['pickup_dropoff'].head(2))

grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
print(grouped[:5])
grouped_dict = grouped.to_dict()
grouped_dict = grouped_dict['trip_distance']
df['mean_distance'] = df['pickup_dropoff'].copy()
df['mean_distance'] = df['mean_distance'].map(grouped_dict)
print(df[(df['PULocationID']==100) & (df['DOLocationID']==231)][['mean_distance']])


#create a mean_duration column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
group = df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
grouped_dicted = group.to_dict()
grouped_dicted = grouped_dicted['duration']
df['mean_duration'] = df['pickup_dropoff'].copy()
df['mean_duration'] = df['mean_duration'].map(grouped_dicted)
print(df[(df['PULocationID']==100) & (df['DOLocationID']==231)][['mean_duration']])

#Create day and month columns
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

#Create rush_hour column
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour

'''
Any weekday (not Saturday or Sunday) AND
Either from 06:00–10:00 or from 16:00–20:00
'''

def rush_hourizer(hour):
    if 6 <= hour < 10:
        return 1
    elif 16 <= hour < 20:
        return 1
    else:
        return 0

df['rush_hour'] = df['rush_hour'].apply(rush_hourizer)
df.loc[df['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0
print(df.head())

#Scatter plot
sns.set_theme(style='whitegrid')
f= plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
           scatter_kws= {'alpha': 0.5, 's': 5},
           line_kws= {'color': 'red'})
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.title('Mean duration x fare amount')
plt.savefig(os.path.join(images_dir, "mean_duration_vs_fare.png"), bbox_inches="tight", dpi=300)
#plt.show()

df[df['fare_amount'] > 50]['fare_amount'].value_counts().head()
pd.set_option('display.max_columns', None)
print(df[df['fare_amount']==52].head(30))

'''
It seems that almost all of the trips in the first 30 rows where the fare amount was $52 either begin or end at location 132, 
and all of them have a RatecodeID of 2.

There is no readily apparent reason why PULocation 132 should have so many fares of 52 dollars. 
They seem to occur on all different days, at different times, with both vendors, in all months. 
However, there are many toll amounts of  5.76 and 5.54. This would seem to indicate that location 132 
is in an area that frequently requires tolls to get to and from. 
It's likely this is an airport.

The data dictionary says that RatecodeID of 2 indicates trips for JFK, which is John F. Kennedy International Airport. 
A quick Google search for "new york city taxi flat rate $52" indicates that in 2017 (the year that this data was collected) 
there was indeed a flat fare for taxi trips between JFK airport (in Queens) and Manhattan.

Because RatecodeID is known from the data dictionary, 
the values for this rate code can be imputed back into the data after the model makes its predictions. 
This way you know that those data points will always be correct.

'''
#Isolate modeling variables
print(df.info())
df2 = df.copy()
df2 = df2.drop(['Unnamed: 0', 'tpep_dropoff_datetime', 'tpep_pickup_datetime',
               'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
               'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount', 'tpep_dropoff_datetime', 'tpep_pickup_datetime', 'duration',
               'pickup_dropoff', 'day', 'month'
               ], axis=1)

print(df2.info())

#Pair plot
g = sns.pairplot(df2[['fare_amount', 'mean_duration', 'mean_distance']],
             plot_kws={'alpha':0.4, 'size':5},
             );
g.savefig(os.path.join(images_dir, "pairplot_matrix.png"), bbox_inches="tight", dpi=300)
#Identify correlations
print(df2.corr(method='pearson'))

#Correlation heatmap of the data.
plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=18)
plt.savefig(os.path.join(images_dir, "correlation_heatmap.png"), bbox_inches="tight", dpi=300)
#plt.show()

'''
In conclusion, based on the correlation matrix and the heatmap, 
the only variables that show a strong, direct, positive correlation with the fare amount (fare_amount) are: 
mean_distance: Shows a very high correlation coefficient of 0.91 mean_duration: 
Shows a strong correlation coefficient of 0.86. 
This logically confirms that the greater the distance traveled or the longer the travel time, 
the higher the fare will be. Conversely, 
the remaining variables (VendorID, passenger_count, and rush_hour) have coefficients close to zero (0.001, 0.015, and -0.02, respectively), 
which indicates that they do not have a significant linear correlation with the final fare.

'''
#Split data into outcome variable and features
print(df2.info())
X = df2.drop(columns=['fare_amount'])
y = df2[['fare_amount']]
print(X.head())

#Pre-process data
X['VendorID'] = X['VendorID'].astype(str)
X = pd.get_dummies(X, drop_first=True)
print(X.head())

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Standardize the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)

#Fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

#Evaluate model
#Train data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))

#Test data
X_test_scaled = scaler.transform(X_test)
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))

#Results
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
print(results.head())

#Visualize model results
fig, ax = plt.subplots(figsize=(6, 6))
sns.set_theme(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax)

plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.title('Actual vs. predicted')
plt.savefig(os.path.join(images_dir, "actual_vs_predicted.png"), bbox_inches="tight", dpi=300)
plt.close()

sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count')
plt.savefig(os.path.join(images_dir, "residuals_distribution.png"), bbox_inches="tight", dpi=300)
plt.close()

print(results['residual'].mean())

#Scatterplot of the residuals over predicted
sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
#plt.show()
plt.savefig(os.path.join(images_dir, "residuals_vs_predicted.png"), bbox_inches="tight", dpi=300)
plt.close()
#Coefficients
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
print(coefficients)

'''
Given that the independent variables were standardized using StandardScaler, 
the coefficients represent the expected change in the target variable (fare_amount) 
for each one-standard-deviation increase in the predictor variable, 
while holding all other variables constant.

mean_distance (~7.17): This is the strongest predictor. 
A one-standard-deviation increase in the mean distance raises the estimated fare by approximately $7.17.

mean_duration (~2.77): This also has a significant positive impact; 
a one-standard-deviation increase in the mean duration raises the fare by about $2.77.

passenger_count, rush_hour, and VendorID_2: 
These show much smaller coefficients—close to zero—indicating 
that they have a minimal or negligible linear effect on the fare compared to distance and time.
'''
#Conclusion
'''
Importance of feature engineering: The created variables (mean_distance and mean_duration) 
are fundamental because they reflect the actual relationship with the trip fare.

Predictive power: Trip distance and duration are, by a wide margin, 
the primary determinants of the fare amount (fare_amount), 
far outweighing factors such as the number of passengers or the provider (VendorID).

Normalization: The use of scalers, such as StandardScaler, 
is essential for fairly comparing the weight of each variable within linear regression models.

Model performance: The multiple linear regression model explains a high proportion of the variance in taxi fares (with an  R2
coefficient of determination of approximately 0.84 on the training data). 
Cost drivers: It was statistically determined that for every one-standard-deviation increase in trip distance or duration, 
the fare increases in direct proportion. Identified business rules: 
The impact of regulated fixed fares (such as the $52.00 flat rate to JFK Airport) was validated, 
as was the proper handling of outliers during data preparation to prevent bias in the model.
'''