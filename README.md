Project is about - Car insurannce fraud detection
To detect any fraud is happening in the insurance.
Objective: To detect whether an insurance claim is fraudulent or authentic. (Supervised, classification task)

Dataset: https://www.kaggle.com/buntyshah/auto-insurance-claims-data

Steps:

Data Gathering and Preparation
Data Analysis and Visualization
Explanatory Model Building
Predictive Model Building

Journey:

Data Gathering and Preparation

import pandas as pd
df = pd.read_csv(path to file)
df

df = df.drop('policy_number',  axis = 1)
df = df.drop('_c39', axis = 1)
import numpy as np
#Deriving a feature months between 'policy bind date' and 'incident date'
df['incident_date'] = pd.to_datetime(df.incident_date)
df['policy_bind_date'] = pd.to_datetime(df.policy_bind_date)
df['months_bw_incident_and_bind'] = ((df.incident_date - df.policy_bind_date)/np.timedelta64(1, 'M'))
df['months_bw_incident_and_bind'] = df['months_bw_incident_and_bind'].astype(int)
df

#Deriving a feature indicating whether months between policy bind date and incident date falls within months as customer
df['incident_within_customership'] = df[['months_bw_incident_and_bind','months_as_customer']].apply(lambda x: 1 if x.months_as_customer >= x.months_bw_incident_and_bind and x.months_bw_incident_and_bind > 0 else 0, axis=1)
df['capital-loss'] = df['capital-loss'].abs()
Sub-step :

I. Data quality checks:

a) Data shouldn’t be erronous

No Typing error
All data in same format or data type as expected in respective columns.
No structure issues
Data should be in expected range or accepted set of values wherever applicable. Eg: Age<=0 is invalid.
b) Checking for duplicate rows.

c) Checking for inconsistent data. Eg: Same customer, different age

d) Checking whether data is updated. Eg: Last date recorded for a customer being too obsolete indicates non-updation.

#Data Quality Check
#Check for consistency in data: Total claim must be equal to sum of "injury_claim", "property_claim", "vehicle_claim" and print the number of rows where it does not hold true
print (df[df['total_claim_amount'] != df['injury_claim'] + df['property_claim']+ df['vehicle_claim']].shape[0])
#Check for accepted range/set of values and print the number of rows where value < 0 in each numerical column 
for i in dict(df.dtypes):
 if (dict(df.dtypes)[i] == 'int64' or dict(df.dtypes)[i] ==   'float64'):
  print(i , " : ", df[df[i]<0].shape[0])

#Treating unaccepted value by row removal
df.drop(df.index[df[df['umbrella_limit']<0].index[0]], inplace = True)
print(df[df['umbrella_limit']<0].shape[0])

Sub-step :

II. Ways of detecting extreme values (outliers):

Boxplots
Z-score
Inter Quantile Range(IQR)
import matplotlib.pyplot as plt
for i in dict(df.dtypes):
 if dict(df.dtypes)[i] == 'int64' or dict(df.dtypes)[i] ==  'float64':
  plt.boxplot(df[i], vert=False)
  plt.title("Detecting outliers using Boxplot")
  plt.xlabel(i)
  plt.show()

import numpy as np
outliers = []
def detect_outliers_zscore(data):
 thres = 3
 mean = np.mean(data)
 std = np.std(data)
 for i in data:
  z_score = (i-mean)/std
  if (np.abs(z_score) > thres):
   outliers.append(i)
 return outliers
Sub-step :

III. Ways of handling extreme values (outliers):

Trimming/removing the outlier
Quantile based flooring and capping
Mean/Median imputation
Outliers here are treated by Median imputation as:

The size of dataset is small and Trimming/removing outliers would further decrease available information
As the mean value is highly influenced by the outliers, it is practice to replace the outliers with the median value.
for i in dict(df.dtypes):
 outliers = []
 if dict(df.dtypes)[i] == 'int64' or dict(df.dtypes)[i] ==  'float64':
  sample_outliers = detect_outliers_zscore(df[i])
  print("Column: ", i)
  print("Outliers from Z-scores method: ", sample_outliers)
  if len(sample_outliers) > 0:
   median = np.median(df[i])
   for j in sample_outliers:
    df[i] = np.where(df[i]==j, median, df[i])
   print("After treatment: ")
   outliers = []
   sample_outliers = detect_outliers_zscore(df[i])
   print("Outliers from Z-scores method: ", sample_outliers)

Sub-step :

IV. Ways of handling missing data:

Deleting the Entire Row
Deleting the Entire Column
Replacing With Arbitrary Value
Replacing With Mean
Replacing With Mode
Replacing With Median
Replacing with Previous Value — Forward Fill
Replacing with Next Value — Backward Fill
Replacing with the Value “missing”, which treats it as a separate Category
Missing values here are treated with Mode imputation as:

Values were missing in categorical columns.
df = df.replace("?", np.NaN)
for i in df.columns:
 print ("Column: ", i)
 print ("Number of nulls: ", df[i].isnull().sum())

#To distinguish between continuous and categorical columns. High number of unique values in a column indicative of the column being continuous, low number of unique value indicates it is a categorical column.
df.nunique()

#All values in this column are different
df = df.drop('incident_location', axis = 1)
#Missing values here are treated with Mode imputation as values were missing in categorical columns
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
Sub-step:

V. Stepwise feature engineering:

Encoding
Feature Selection
Binning
Feature Scaling
Steps carried out here:

Encoding
Feature Selection
#Encoding
mappings = {}
for i in dict(df.dtypes):
 if dict(df.dtypes)[i] == 'O':
  mappings[i] = dict(zip(df[i].unique(), range(len(df[i].unique()))))
  df[i] = df[i].map(lambda x: mappings[i][x])
df

#Feature selection
import seaborn as sns
plt.figure(figsize = (18, 12))
corr = df.corr()
sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1)
plt.show()

df = df.drop('vehicle_claim', axis = 1) #as its correlation with another feature (total_claim_amount) > 95%. Keeping multiple strongly correlated featured would make the collection of such features influential.
df = df.drop(['policy_bind_date', 'incident_date'], axis = 1) #as information from these features has been exported to new derived features
from sklearn.ensemble import ExtraTreesClassifierimport matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

#Finding 10 most important features
feature_importances = pd.Series(model.feature_importances_, index = x.columns)
feature_importances.nlargest(10).plot(kind = 'barh')
plt.show()

#Finding 10 least important features
feature_importances.nsmallest(10).plot(kind = 'barh')
plt.show()
