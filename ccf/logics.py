import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

def eda(df):

    classes = df['Class'].value_counts()
    normal_share = round((classes[0]/df['Class'].count()*100),2)
    fraud_share = round((classes[1]/df['Class'].count()*100),2)

    images = []

    buffer_1 = io.BytesIO()
    sns.countplot(x='Class', data=df)
    plt.title('Number of fraudulent vs non-fraudulent transcations')
    plt.savefig(buffer_1, format='png')
    plt.close()
    buffer_1.seek(0)
    images.append(base64.b64encode(buffer_1.read()).decode('utf-8'))

    buffer_2 = io.BytesIO()
    fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
    df_fraud_percentage = pd.DataFrame(fraud_percentage) 
    sns.barplot(x='Class',y='Percentage', data=df_fraud_percentage)
    plt.title('Percentage of fraudulent vs non-fraudulent transcations')
    plt.savefig(buffer_2, format='png')
    plt.close()
    buffer_2.seek(0)
    images.append(base64.b64encode(buffer_2.read()).decode('utf-8'))

    buffer_3 = io.BytesIO()
    data_fraud = df[df['Class'] == 1]
    data_non_fraud = df[df['Class'] == 0] 
    plt.figure(figsize=(8,5))
    ax = sns.distplot(data_fraud['Time'],label='fraudulent',hist=False)
    ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
    ax.set(xlabel='Seconds elapsed between the transction and the first transction')
    plt.savefig(buffer_3, format='png')
    plt.close()
    buffer_3.seek(0)
    images.append(base64.b64encode(buffer_3.read()).decode('utf-8')) 

    return images

def preprocess_data(df):
    # print(df)
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
    scaler = StandardScaler()
    X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['Amount'] = scaler.transform(X_test[['Amount']])
    return X_train, X_test, y_train, y_test

def draw_roc(actual, probs):
    buffer = io.BytesIO()
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(buffer, format='png')
    plt.close()
    # plt.show()
    buffer.seek(0)

    return buffer

def logistic_regression(X_train, X_test, y_train, y_test):    
    
    logistic_imb = LogisticRegression(C=0.01)
    logistic_imb_model = logistic_imb.fit(X_train, y_train)
    y_test_pred = logistic_imb_model.predict(X_test)
    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    print(confusion)
    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
   # Accuracy
    print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

    # Sensitivity
    print("Sensitivity:-",TP / float(TP+FN))

    # Specificity
    print("Specificity:-", TN / float(TN+FP))

    # F1 score
    print("F1-Score:-", f1_score(y_test, y_test_pred))

    print(classification_report(y_test, y_test_pred))
    y_test_pred_proba = logistic_imb_model.predict_proba(X_test)[:,1]
    return draw_roc(y_test, y_test_pred_proba)

def xg_boost(X_train, X_test, y_train, y_test):
    params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}
    # fit model on training data
    xgb_imb_model = XGBClassifier(params = params)
    xgb_imb_model.fit(X_train, y_train)
    y_test_pred = xgb_imb_model.predict(X_test)
    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    print(confusion)
    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
    print(classification_report(y_test, y_test_pred))
    y_test_pred_proba = xgb_imb_model.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
    return draw_roc(y_test, y_test_pred_proba)

def decision_tree(X_train, X_test, y_train, y_test):
    dt_imb_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=100,
                                  min_samples_split=100)

    dt_imb_model.fit(X_train, y_train)
    y_test_pred = dt_imb_model.predict(X_test)
    confusion = metrics.confusion_matrix(y_test, y_test_pred)
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    print(confusion)
    # Accuracy
    print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

    # Sensitivity
    print("Sensitivity:-",TP / float(TP+FN))

    # Specificity
    print("Specificity:-", TN / float(TN+FP))

    # F1 score
    # print("F1-Score:-", f1_score(y_train, y_test_pred))

    print(classification_report(y_test, y_test_pred))

    y_test_pred_proba = dt_imb_model.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
    return draw_roc(y_test, y_test_pred_proba)

def plot_learningCurve(history, epoch):
    # Plot training & validation accuracy values
    buffer_1 = io.BytesIO()
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(buffer_1, format='png')
    # plt.show()
    plt.close()
    buffer_1.seek(0)

    # Plot training & validation loss values
    buffer_2 = io.BytesIO()
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(buffer_2, format='png')
    # plt.show()
    plt.close()
    buffer_2.seek(0)
    return buffer_1, buffer_2

def cnn(X_train, X_test, y_train, y_test, epochs=20):
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape = X_train[0].shape))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss = 'binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)
    
    img1, img2 = plot_learningCurve(history, epochs)
    img3 = draw_roc(y_test, model.predict(X_test))

    return img1, img2, img3

def run_algorithms(df):
    images = []
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    buffer_log_reg = logistic_regression(X_train, X_test, y_train, y_test)
    images.append(base64.b64encode(buffer_log_reg.read()).decode('utf-8'))

    buffer_xg_boost = xg_boost(X_train, X_test, y_train, y_test)
    images.append(base64.b64encode(buffer_xg_boost.read()).decode('utf-8'))

    buffer_decison_tree = decision_tree(X_train, X_test, y_train, y_test)
    images.append(base64.b64encode(buffer_decison_tree.read()).decode('utf-8'))
    
    # for cnn we first need to balance the dataset
    non_fraud = df[df['Class']==0]
    fraud = df[df['Class']==1]
    non_fraud = non_fraud.sample(fraud.shape[0])
    data = pd.concat([fraud, non_fraud], ignore_index=True)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    buffer_1_cnn, buffer_2_cnn, buffer_3_cnn = cnn(X_train, X_test, y_train, y_test)
    images.append(base64.b64encode(buffer_1_cnn.read()).decode('utf-8'))
    images.append(base64.b64encode(buffer_2_cnn.read()).decode('utf-8'))
    images.append(base64.b64encode(buffer_3_cnn.read()).decode('utf-8'))

    return images