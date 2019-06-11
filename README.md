# Chronic Kidney Disease diagnoser using an SVM and a bagged decision tree Random Forest

The code prepares and processes the dataset from a csv file, it performs feature analysis and ranking of features then optimizes the SVM model using a bayesian optimizer. The model is evaluated using K-fold cross-validation, theres is no hold out validation performed. A confusion matrix is also generated to produce the recall,precision and F1 score of the model.

The code is properly structured, well commented and symbolic variables used allowing beginners too have an easy time understanding it.

Please also do cite or share my work if you like it :-)

# Dataset Brief
This dataset features diagnostic data of early stage chronic kidney disease collected from an Indian population. The dataset has a total of 400 samples each with 24 features. Of the features, 11 are numeric values while 13 are categorical values and certain samples do have certain missing data. Each sample is then classed in one of two classes, having CKD or not having CKD. This data set can then be used in the creation of model to diagnose a patient with chronic Kidney with the given features.

# References and Dataset source  
L.Jerlin (2015). UCI Machine Learning Repository [ https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease ]. Irvine, CA: University of California, School of Information and Computer Science
