#!/usr/bin/env python
# coding: utf-8

# # Overview of Problem 
# 
# ## Aim
# Classify fetal health in order to prevent child and maternal mortality. Classified into 3 classes:
# 
# - Normal
# - Suspect
# - Pathological
# 
# ## Dataset
# - 2126 fetal cardiotocograms (CTG) 
# 
# ### Features
# 
# #### Response Variable
# - fetal health: 
#     - 1= normal
#     - 2 = suspect
#     - 3= pathological
# 
# #### Predictor Variables
# - baseline value:Baseline Fetal Heart Rate (FHR)
# - accelerations: Number of accelerations per second
# - fetal_movement:Number of fetal movements per second
# - uterine_contractions: Number of uterine contractions per second
# - light_decelerations: per second
# - severe_decelerations: per second
# - prolongued_decelerations: per second
# - abnormal_short_term_variability: Percentage of time with abnormal short term variability 
# - mean_value_of_short_term_variability: Mean value of short term variability
# - percentage_of_time_with_abnormal_long_term_variability: Percentage of time with abnormal long term variability
# - Mean value of long term variability
# - histogram_width: Width of the histogram made using all values from a record
# - Histogram minimum value
# - histogram_max
# - histogram_number_of_peaks:Number of peaks in the exam histogram
# - histogram_number_of_zeroes: Number of zeroes in the exam histogram
# - histogram_mode
# - histogram_mean
# - histogram_median
# - histogram_variance
# - Histogram trend

# # General Thoughts Before Starting
# 
# ## Common Algorithms For Multi-Class Classification 
# - k-Nearest Neighbors.
# - Naive Bayes.
# - Decision Trees.
# - Ensemble Models: Random Forest.
# - Boosting Models: AdaBoost and XGBoost
# 
# ### Measure of Focus
# Common measures used to evaluate the outcome of classification problems include AUC, F1 score, Precision and Recall.
# 
# Further, when analysing the data, it is apparent that there is considerable class imbalance and therefore accuracy is not recommended as primary metric.
# 
# In this problem, the cost of not correctly identifying risks with child birth (ie. Case 3 of the predictor variable, that being Pathological) is high and there recall for case 3 should be the metric of focus), this measures: for pathological outcomes what proportion of actual positives was identified correctly?
# 

# # Importing Packages

# In[1]:


get_ipython().magic('pip install xgboost')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, precision_score, recall_score


# # Initial Loading and Data Analysis

# In[3]:


df = pd.read_csv('fetal_health.csv')


# In[4]:


print(f"Shape of Dataset: {df.shape}")


# 22 columns (21 predictor variables and 1 response variable) and 2126 rows

# In[5]:


df.info()


# No missing values

# In[6]:


df.describe().T


# In[7]:


df.head()


# In[8]:


df.nunique()


# In[9]:


print("Number of unqiue values by column \n")

for i in df:
    print("{}:".format(i), df[i].nunique(), "unique values \n", df[i].unique(), "\n")


# ## Setting Target Variable (Initial Loading and Data Analysis)

# In[10]:


#axis = 0 to drop labels from the index oe axis=1 to drop labels from columns
X=df.drop(['fetal_health'],axis=1)
y= df.fetal_health


# ## EDA

# ### Analysing Target Variable

# In[11]:


vis_fetal_health = y.value_counts().plot(figsize=(20, 5), kind="bar", color = ["green", "orange", "red"])
plt.title("Fetal health count")
plt.xlabel("Fetal helth")
plt.ylabel("Cases")


#Counting labels
print("Breakdown of unique values:\n",y.value_counts())


# In[12]:


plt.title("Fetal state")

plt.pie(y.value_counts(),labels=["Normal", "Suspect", "Pathological"], colors = ["green", "orange", "red"],autopct="%1.1f%%",radius=1.2)
plt.xlabel("Fetal health")
plt.ylabel("Cases")
plt.show()


# #### Discussion on target variable
# This is an imbalanced dataset (78% of observations are 'Normal'). This means we have to be careful on our evaluation metric, in which the metric accuracy (Total correct/ total) is misleading. 
# - For example: Imagine if we classified all the data as 0 (ie. Normal). Our accuracy would be 78%.
# 
# For this reason and for the fact the identifying case 3 when it occurs (ie. high recall) is of significant importance, we will use recall for case 3 as are primary evaluation metric.
# 
# Other metrics of importance include the F1 score (precision and recall) and recall for case 2 ('Suspect') as misclassified suspect cases (in particularlty if they are misclassified as 'normal') can lead to the missing of 'pathological' issues

# ### Correlation

# In[13]:


# Correlation between different variables
corr = df.corr()
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(18, 15))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)


# #### Discussion on Correlation
# - histogram mode, mean and median are highly correlated (obvious reasons), they are also correlated with the baseline value. This correlation is analysed below.
# - Target variable: 
#     - positive correlation: prolongued_decelerations, abnormal_short_term_variability, percentage_of_time_with_abnormal_long_term_variability
#     - negative correlation: accelerations
#     - the target variable correlations are analyses below this section

# ## Predictor Variable Correlations
# Histogram mean, mode, median and baseline value

# In[14]:


sns.regplot(x=df['histogram_mean'], y=df['histogram_median'], line_kws={"color":'black'})


# In[15]:


sns.regplot(x=df['histogram_mean'], y=df['histogram_mode'], line_kws={"color":'black'})


# In[16]:


sns.regplot(x=df['baseline value'], y=df['histogram_mode'], line_kws={"color":'black'})


# In[17]:


sns.lmplot(x='histogram_mean', y='histogram_mode', hue='fetal_health', data=df)


# In[18]:


sns.lmplot(x='baseline value', y='histogram_mode', hue='fetal_health', data=df)


# In[19]:


sns.lmplot(x='baseline value', y='histogram_mean', hue='fetal_health', data=df)


# The histogram mean, median and mode are highly correlated, we have removed all except one (mode).
# 
# Although mode and baseline are correlated (71%) we have kept the two variables as they vary when the mode is small (as shown above)

# ## Target Variable Correlation

# In[20]:


Pos_Num_feature = df.corr()["fetal_health"].sort_values(ascending=False).to_frame()
Neg_Num_feature = df.corr()["fetal_health"].sort_values(ascending=True).to_frame()


# In[21]:


Pos_Num_feature[1:6], Neg_Num_feature[0:5]


# In[22]:


#Correlation b/w all features and Target Variable
Pos_Num_feature[1:], Neg_Num_feature


# #### Discussion on Target Variable Correlation
# Based on above correlation, let's analyse these further:
# - prolongued_decelerations: 0.484859 correlation
# - abnormal_short_term_variability: 0.471191 correlation
# - percentage_of_time_with_abnormal_long_term_variance: 0.426146 correlation
# - accelerations: -0.364066 correlation

# ### Graphing Prominant Variables

# ### Discrete Variables that are highly correlated to fetal health

# #### Prolongued Decelerations

# In[23]:


counts_df = df.groupby(["prolongued_decelerations", "fetal_health"])["fetal_health"].count().unstack()
# Transpose so fetal_health categories add up to 1, divide by the total number (transposed), then transpose one more time for plotting
fetal_health_percents_df = counts_df.T.div(counts_df.T.sum()).T


# In[24]:


fig, ax = plt.subplots()


fetal_health_percents_df.plot(kind="bar", stacked=True, color=["green", "orange","red"], ax=ax)

sns.catplot(x="prolongued_decelerations", hue="fetal_health", data=df, kind="count", palette=sns.color_palette(['green', 'orange','red'])
             )


# Majority of prolongued decelerations are 0.0, when higher, there is high occurance of pathological fetal health

# ### Continous Variables that are highly correlated to fetal health

# #### abnormal_short_term_variability

# In[25]:


sns.stripplot(x="fetal_health", y="abnormal_short_term_variability", data=df)


# #### percentage_of_time_with_abnormal_long_term_variance

# In[26]:


sns.stripplot(x="fetal_health", y="percentage_of_time_with_abnormal_long_term_variability", data=df)


# Although abnormal long term variability is somewhat spread for case 3. When above 80, all observations are case 3

# #### Accelerations

# In[27]:


sns.stripplot(x="fetal_health", y="accelerations", data=df)


# High accelerations good for fetal health

# ### Distribution of All Variables

# In[28]:


df_hist_plot = df.hist(figsize = (20,15), color = "#000054")


# - Three types of skewed distributions. A right (or positive) skewed distribution, left (or negative) skewed distribution, and normal distribution.
# 
#     - A left-skewed distribution (negatively-skewed) has a long left tail.
#     - A right-skewed distribution (positively-skewed) has a long right tail
#     - The skewness for a normal distribution is zero and looks a bell curve.

# #### Histogram Variance Outliers
# Outliers are present in Histogram Variance, let's have a closer look at these

# In[29]:


df[df['histogram_variance'] > 180]


# Removing outliers is risky as they may contain valuable information about the data. Here we can see that extreme vairance for Histogram Variance largely correlate with fetal health being pathological, therefore we will keep these outliers.

# ## Looking at the range of values

# In[30]:


df_box_plot = df.boxplot(vert=False, color = "#000054")


# The above plot shows the range of our feature attributes. All the features are in different ranges. To fit this dataset in a KNN model we must scale it to the same range. This is not required for decision trees and Ensemble methods as they are not sensitive to the the variance in the data.

# # Feature Eng

# In[31]:


#Keeping Histogram mode out of the 3 highly correlated variables, reasons discussed above
X = X.drop(['histogram_median'],axis=1)
X = X.drop(['histogram_mean'],axis=1)


# In[32]:


X.head()


# ## Spliting Train/Test

# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=1, stratify = y)


# In[34]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# ## Dealing with Missing Data

# No missing values so imputation/ dropping features not required

# ### Normalize Data
# Reasons discussed above in the graph of boxplots
# Note: Decision trees and ensemble methods do not require feature scaling to be performed as they are not sensitive to the the variance in the data.

# In[35]:


scaler = MinMaxScaler()


# In[36]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# # Training

# ## Selecting Models

# ### MODEL 1: k-Nearest Neighbors

# In[37]:


knn_classification = KNeighborsClassifier(n_neighbors = 3)

# fit the model using fit() on train data
# Used scaled data for KNN
knn_model = knn_classification.fit(scaled_X_train, Y_train)


# In[38]:


knn_pred = knn_model.predict(scaled_X_test)


# In[39]:


print(classification_report(Y_test,knn_pred))


# In[40]:


ax= plt.subplot()
sns.heatmap(confusion_matrix(Y_test, knn_pred), annot=True, ax = ax, cmap = "Blues");

ax.set_xlabel("Predicted");
ax.set_ylabel("Actual"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);
ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# 77% recall for case of focus (3) and overpredicting case 1 as shown through high accuracy but low precision and recall for classes that are underrepresented in the data (imbalance in dataset). Although recall for case 2 is not the prominant metric of focus, it is important especially if this misclassification is being classified as normal as the cost of missing ill fetal health it high (which it is misclassified as normal in 91% of its false positives).

# ###  MODEL 2: Decision Tree

# In[41]:


decisionTreeClassifier = DecisionTreeClassifier()


# In[42]:


decisionTreeClassifier.fit(X_train,Y_train)


# In[43]:


decisionTreeClassifier_pred = decisionTreeClassifier.predict(X_test)


# In[44]:


print(classification_report(Y_test,decisionTreeClassifier_pred))


# In[45]:


ax= plt.subplot()
sns.heatmap(confusion_matrix(Y_test, decisionTreeClassifier_pred), annot=True, ax = ax, cmap = "Blues");

ax.set_xlabel("Predicted");
ax.set_ylabel("Actual"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);
ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# Recall for case 3 is stronger (87%) and recall for case 2 has improved (81%).
# 
# As decision trees are prone to overfitting, let's see if random forest can improve this score

# ###  MODEL 3: Random Forest (Bagging Ensemble Method)

# RF algorithm is an ensemble method that uses multiple weak learners (ie. decision trees) and aggregates then up (bagging -> boostrapping + aggregation) to vote on the outcome of each prediction -> idea here is to reduce overfitting 

# In[46]:


rfc = RandomForestClassifier()


# In[47]:


rfc.fit(X_train,Y_train)


# In[48]:


rfc_pred = rfc.predict(X_test)


# In[49]:


print(classification_report(Y_test,rfc_pred))


# In[50]:


ax= plt.subplot()
sns.heatmap(confusion_matrix(Y_test, rfc_pred), annot=True, ax = ax, cmap = "Blues");

ax.set_xlabel("Predicted");
ax.set_ylabel("Actual"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);
ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# Recall for class 3 has improved as well at the overall f1 score.
# Let's try a second type of ensemble method, Boosting, we will look at AdaBoost (uses stumps as weak learners) and XGBoost

# ###  MODEL 4: AdaBoost (Boosting Ensemble Method 1)

# #### Build weak learner

# In[51]:


#n_estimators -> the number of weak learner we are going to use.
#building the weak learners
base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=1)
AdaBoost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=400, learning_rate=1)


# In[52]:


AdaBoostModel = AdaBoost.fit(X_train, Y_train)


# In[53]:


ada_pred = AdaBoostModel.predict(X_test)


# In[54]:


print(classification_report(Y_test,ada_pred))


# In[55]:


ax= plt.subplot()
sns.heatmap(confusion_matrix(Y_test, ada_pred), annot=True, ax = ax, cmap = "Blues");

ax.set_xlabel("Predicted");
ax.set_ylabel("Actual"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);
ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# Case 3 recall is solid (91%).
# 
# In saying this, it is important to note that Recall for case 2 is poor (71%), although this is not the main metric of focus it is important as looking at the confusion matrix 100% False Negatives for 'Suspect' are incorrectly classified as 'Normal'. The issue here is 'Suspect' classifications give a call to action to doctors to look further at risks with child birth, so misclassifying them as Pathological would be prefered over misclassification of 'Normal'.
# 
# Let's try another ensemble boosting method (XGBoost)

# ###  MODEL 5: XGBoost (Boosting Ensemble Method 2)

# In[56]:


XGB = XGBClassifier()
XGB_Model = XGB.fit(X_train,Y_train)


# In[57]:


XGB_pred = XGB_Model.predict(X_test)


# In[58]:


print(classification_report(Y_test,XGB_pred))


# In[59]:


ax= plt.subplot()
sns.heatmap(confusion_matrix(Y_test, XGB_pred), annot=True, ax = ax, cmap = "Blues");

ax.set_xlabel("Predicted");
ax.set_ylabel("Actual"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);
ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# Again recall for case 3 is strong (94%) but also recall for case 2 is higher than random forest (80%), the importance of this is discussed above, reducing the amount of false negatives being classified as normal.

# # Summary
# 
# XGBoost scores the highes for recall on outcome 3 (94%). This is important as identifying Pathological cases is the most important outcome in this problem (in order to prevent child and maternal mortality).
# 
# Further, a lot of models were scoring in the 80-90% range for case 3 recall but scoring poorly for case 2 recall (60-70% range), largely misclassifying these instances as 'Normal'. It is important to have strong recall for case 2 as it acts as an alarm/call to action for doctors to look further.
# 
# Therefore XGBoost is the best model for this problem.

# In[ ]:




