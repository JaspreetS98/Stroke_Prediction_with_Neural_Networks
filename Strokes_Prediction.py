#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jaspreet singh
"""

#import libraries
import os
import numpy as np 
import pandas as pd 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler 
from sklearn.impute import KNNImputer 
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.regularizers import l2
from sklearn.model_selection import train_test_split 
from tensorflow.keras import layers, models 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.metrics import confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
import tensorflow as tf 
from sklearn.metrics import accuracy_score, recall_score ,precision_score, f1_score 
from tensorflow.keras import optimizers

#libraries for data visualisation
import missingno as msn
import matplotlib.pyplot as plt 
import seaborn as sns


#function containing NN model
def ann_class_model(optimizer = 'adam',learning_rate=0.0001):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units= 64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    tf.keras.layers.Dropout(0.6)
    ann.add(tf.keras.layers.Dense(units= 32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    tf.keras.layers.Dropout(0.6)
    ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
    ann.compile(optimizer= optimizer, loss= 'binary_crossentropy', metrics= ['accuracy'])
    return ann


#set display columns settings, read and show head of dataset  
pd.set_option('display.max_columns', None)
strokes = pd.read_csv('healthcare-dataset-stroke-data.csv', index_col='id')
print(strokes.head())

#show number of unique values for each feature
print(strokes.nunique())

#show stats of the numerical and catagorical features
print(strokes.describe())
print(strokes.describe(exclude = ['float', 'int64']))

#show number of null values in each column
print(strokes.isnull().sum())

#drop "Other" value in gender rows as too little values
print(strokes.gender.value_counts())
strokes = strokes.drop(strokes[strokes.gender == "Other"].index)
print(strokes.gender.value_counts())

#show some data visualization

#plot bar chart for missing values in columns
colors = []
for col in strokes.columns:
    if strokes[col].isna().sum() != 0:
        colors.append('red')
    else:
        colors.append('blue')
msn.bar(strokes, color=colors, figsize=(10,5), fontsize=10)
plt.title('Missing values', size=20, y=1)


#create subplot
sns.set_style(style = 'darkgrid')
subp, axes = plt.subplots(1,2, figsize = (15,5))
subp.suptitle('Various Features Frequency',fontsize=20)

#plot bar chart for people's smoking status
p = plt.figure(figsize=(8, 6))
p = gender_plot = sns.countplot(y=strokes.smoking_status, palette=("Red","Blue"),ax=axes[0])
p = gender_plot.set_title('smoking status', fontsize=20, y=1)

#plot bar chart for work types
p = plt.figure(figsize=(8, 6))
p = work_plot = sns.countplot(y=strokes.work_type, palette=("Red","Blue"),ax=axes[1])
p = work_plot.set_title('Work types', fontsize=20, y=1)

#create subplot 
sns.set_style(style = 'darkgrid')
subp, axes = plt.subplots(1,2, figsize = (15,5))

#plot bar chart for genders
p = plt.figure(figsize=(8, 6))
p = gender_plot = sns.countplot(y=strokes.gender, palette=("Red","Blue"),ax=axes[0])
p = gender_plot.set_title('Genders', fontsize=20, y=1)

#plot bar chart for residence types
p = plt.figure(figsize=(8, 6))
p = residence_plot = sns.countplot(y=strokes.Residence_type, palette=("Red","Blue"),ax=axes[1])
p = residence_plot.set_title('Residence types', fontsize=20, y=1)

#plot pie chart for genders
p_col = ("Red","Green")
stroke_count = strokes.stroke.value_counts()
plt.figure(figsize=(15, 8))
plt.pie(stroke_count,labels=['no', 'yes'],colors=p_col,autopct='%1.1f%%')
plt.title('Patient had a stroke', size=20)

#plot pie chart for heart disease
p_col = ("Red","Green")
h_dis_count = strokes.heart_disease.value_counts()
plt.figure(figsize=(15, 8))
plt.pie(h_dis_count,labels=['no', 'yes'],colors=p_col,autopct='%1.1f%%')
plt.title('Persons that had a heart disease', size=20)

#plot pie chart for people ever married
p_col = ("Red","Green")
married_count = strokes.ever_married.value_counts()
plt.figure(figsize=(15, 8))
plt.pie(married_count,labels=['no', 'yes'],colors=p_col,autopct='%1.1f%%')
plt.title('People ever married', size=20)

#plot pie chart for people hypertension
p_col = ("Red","Green")
married_count = strokes.hypertension.value_counts()
plt.figure(figsize=(15, 8))
plt.pie(married_count,labels=['no', 'yes'],colors=p_col,autopct='%1.1f%%')
plt.title('People with hypertension', size=20)

#create subplot schema and style
sns.set_style(style = 'darkgrid')
subp, axes = plt.subplots(1,3, figsize = (15,5))
subp.suptitle('Numerical Features Frequency',fontsize=20)

#plot frequency of Age
p = sns.kdeplot(strokes['age'][(strokes["stroke"] == 0)] , color="Red", shade = True ,ax=axes[0])
p = sns.kdeplot(strokes['age'][(strokes["stroke"] == 1)], ax = p, color="Blue", shade= True)
p.set_xlabel('Age')
p.set_ylabel("Frequency")
p = p.legend(["No","Yes"],fontsize = 10)

#plot frequency of Glucose level
p = sns.kdeplot(strokes['avg_glucose_level'][(strokes["stroke"] == 0)] , color="Red", shade = True,ax=axes[1])
p = sns.kdeplot(strokes['avg_glucose_level'][(strokes["stroke"] == 1)], ax = p, color="Blue", shade= True)
p.set_xlabel('Glucose Level')
p.set_ylabel("Frequency")
p = p.legend(["No","Yes"],fontsize = 10)

#plot frequency of BMI
p = sns.kdeplot(strokes['bmi'][(strokes["stroke"] == 0)] , color="Red", shade = True,ax=axes[2])
p = sns.kdeplot(strokes['bmi'][(strokes["stroke"] == 1)], ax =p, color="Blue", shade= True)
p.set_xlabel('BMI')
p.set_ylabel("Frequency")
p = p.legend(["No,","Yes"],fontsize = 10)

#plot correlation matrix as heatmap for numeric values pre encoding
corr = strokes.corr()
p = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, vmax=.3, center=0, square=True, linewidths=.5)
plt.xlim(0,6)
plt.ylim(6,0)
plt.show()

#show ratio of strokes in dataset
stroke_percent = strokes['stroke'].sum() / len(strokes)
print('Percent of patients in dataset with stroke:', round(stroke_percent, 3))

#encoding catagoric features using LabelEncoder
strokes["gender"] = LabelEncoder().fit_transform(strokes["gender"]) 
strokes["ever_married"] = LabelEncoder().fit_transform(strokes["ever_married"]) 
strokes["Residence_type"] = LabelEncoder().fit_transform(strokes["Residence_type"]) 

#remove labels from dataset
X = strokes.iloc[:, 0:-1].values
y = strokes.iloc[:, -1].values
print(X)
print(y)

#encoding catagoric features using OneHotEncoder
cat_transformed = ColumnTransformer([('categoric_val', OneHotEncoder(), [5,9])], remainder='passthrough')
enc_X = cat_transformed.fit_transform(X)
print(enc_X[0])

#create KNN imputer to impute missing values
knn_imp = KNNImputer(n_neighbors=6)
X_imputed = knn_imp.fit_transform(enc_X)
print(X_imputed[0])

#scale numeric features with StandardScaler/MinMaxScaler 
num_transformed = ColumnTransformer([('numeric_val', StandardScaler(), [10,15,16])], remainder='passthrough')
#num_transformed = ColumnTransformer([('numeric_val', MinMaxScaler(), [10,15,16])], remainder='passthrough')
X_scaled = num_transformed.fit_transform(X_imputed)

#show scaled data
print('First line of the scaled Data:')
print(X_scaled[0])

#splits the data into train, test and validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#show amount of samples in each dataset
print('Shape of Train Set:', X_train.shape)
print('Lenght of Test Set:', len(X_test))
print('Lenght of Validation Set:', len(X_val))

#show count of label 1 and 0 before oversampling
print("Counts of label '1' before: {}".format(sum(y_train==1)))
print("Counts of label '0' before: {}".format(sum(y_train==0)))

#SMOTE for class balancing
sm = SMOTE()

#create a new balanced training set using SMOTE
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

#show ratio of strokes in dataset before and after balancing 
print('Quantity of strokes in dataset:', round(y_train.sum()/len(y_train),3))
print('Quantity of strokes in balanced dataset:', y_train_balanced.sum()/len(y_train_balanced))

#show count of label 1 and 0 after oversampling
print('Shape of train_X after OverSampling: {}'.format(X_train_balanced.shape))
print('Shape of train_X after OverSampling: {}\n'.format(y_train_balanced.shape))
print("Counts of label '1' after OverSampling: {}".format(sum(y_train_balanced==1)))
print("Counts of label '0' after OverSampling: {}".format(sum(y_train_balanced==0)))

#send parameters to KerasClassifier to build NN model with specified parameters
ann = KerasClassifier(build_fn = ann_class_model, batch_size = 64, epochs = 100, optimizer = "adam")

#use 5 fold cross validation on the NN of the balanced train data
cross_val_acc = cross_val_score(estimator = ann, X = X_train_balanced, y = y_train_balanced, cv = 5)

#checking and printing the mean and standard deviation of the 5 fold cross validation accuracies
mean = cross_val_acc.mean()
std_dev = cross_val_acc.std()
print("Cross validation's mean Accuracy: {:.2f} %".format(mean*100))
print("Cross validation's Standard Deviation: {:.2f} %".format(std_dev*100))

"""
#utilize GridSearchCV to get best parameters
parameters = {'batch_size': [16, 32, 64],
             'epochs': [50, 70, 100],
             'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = ann, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search.fit(X_train_balanced, y_train_balanced)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
"""
#add early stop to the model
early_stopper = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

#run train model for history 
ann_history = ann.fit(X_train_balanced, y_train_balanced, batch_size= 64, epochs= 100, callbacks=[early_stopper], validation_data=(X_val, y_val))

#create and plot training and validation loss graph
loss_train = ann_history.history['loss']
loss_val = ann_history.history['val_loss']
epochs = len(loss_train)
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#create and plot training and validation accuracy graph
acc_train = ann_history.history['accuracy']
acc_val = ann_history.history['val_accuracy']
epochs = len(acc_train)
plt.plot( acc_train, 'g', label='Training accuracy')
plt.plot( acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#show the metrics of the NN on the test set
min_val_loss = round(ann_history.history['val_loss'][-30], 2)
min_train_loss = round(ann_history.history['loss'][-30], 2)
max_val_acc = round(ann_history.history['val_accuracy'][-30], 2)
max_train_acc = round(ann_history.history['accuracy'][-30], 2)

#print min and max for loss and accuracy of the trainning and validation set
print('\nValidation set min value for loss: ', min_val_loss)
print('Training set min value for loss: ', min_train_loss)
print('Validation set max value for accuracy: ', max_val_acc)
print('Training set max value for accuracy: ', max_train_acc)

#show the metrics of the NN on the test set
y_pred = ann.predict(X_test)
threshold = 0.20
y_pred = [1. if i > threshold else 0. for i in y_pred]
acc_test = round(accuracy_score(y_test, y_pred),2)
rec_test = round(recall_score(y_test, y_pred),2)
pre_test = round(precision_score(y_test, y_pred),2)
f1_test = round(f1_score(y_test, y_pred),2)

#print metrics of NN on the test set
print('\nModel Score For Accuracy:', acc_test)
print('Model Score For Recall:', rec_test)
print('Mode Score For Precision:', pre_test)
print('Model Score For F1 Score:', f1_test)

# create the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n',cm)

#plot Confusion Matrix
plt.figure(figsize = (5, 5))
sns.heatmap(cm, cmap = 'Reds', annot = True, fmt = 'd', linewidths = 3, cbar = False, annot_kws = {'fontsize': 20}, 
            yticklabels = ['No stroke', 'Stroke'], xticklabels = ['Predicted no stroke', 'Predicted stroke'])
plt.yticks(rotation = 0)
plt.xlim(0, len(np.unique(y)))
plt.ylim(len(np.unique(y)), 0)
plt.show()

