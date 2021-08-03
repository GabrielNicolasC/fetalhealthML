# fetalhealthML
Using Machine Learning to predict outcome for fetal health
# Overview of Problem 

## Aim
Classify fetal health in order to prevent child and maternal mortality. Classified into 3 classes:

- Normal
- Suspect
- Pathological

## Dataset
- 2126 fetal cardiotocograms (CTG) 

### Features

#### Response Variable
- fetal health: 
    - 1= normal
    - 2 = suspect
    - 3= pathological

#### Predictor Variables
- baseline value:Baseline Fetal Heart Rate (FHR)
- accelerations: Number of accelerations per second
- fetal_movement:Number of fetal movements per second
- uterine_contractions: Number of uterine contractions per second
- light_decelerations: per second
- severe_decelerations: per second
- prolongued_decelerations: per second
- abnormal_short_term_variability: Percentage of time with abnormal short term variability 
- mean_value_of_short_term_variability: Mean value of short term variability
- percentage_of_time_with_abnormal_long_term_variability: Percentage of time with abnormal long term variability
- Mean value of long term variability
- histogram_width: Width of the histogram made using all values from a record
- Histogram minimum value
- histogram_max
- histogram_number_of_peaks:Number of peaks in the exam histogram
- histogram_number_of_zeroes: Number of zeroes in the exam histogram
- histogram_mode
- histogram_mean
- histogram_median
- histogram_variance
- Histogram trend


### Measure of Focus
Common measures used to evaluate the outcome of classification problems include AUC, F1 score, Precision and Recall.

Further, when analysing the data, it is apparent that there is considerable class imbalance and therefore accuracy is not recommended as primary metric.

In this problem, the cost of not correctly identifying risks with child birth (ie. Case 3 of the predictor variable, that being Pathological) is high and there recall for case 3 should be the metric of focus), this measures: for pathological outcomes what proportion of actual positives was identified correctly?
