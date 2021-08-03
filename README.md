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
