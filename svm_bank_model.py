import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report



# Read a csv using the pandas read_csv function 
data = pd.read_csv('bank.csv',sep=',',header='infer')
# Remove columns named day, poutcome, contact
data = data.drop(['day','poutcome','contact'],axis=1)

def normalize(data):
    # Before we can feed the data to train
    # and test the classifier, we need to normalize
    # the data to acceptable and convenient values
    # for cross validation and prediction purposes later on 
    data.y.replace(('yes', 'no'), (1, 0), inplace=True)
    data.default.replace(('yes','no'),(1,0),inplace=True)
    data.housing.replace(('yes','no'),(1,0),inplace=True)
    data.loan.replace(('yes','no'),(1,0),inplace=True)
    data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)
    data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
    data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)
    data.job.replace(('technician','services','retired','blue-collar','entrepreneur','admin.',
                      'housemaid','student','self-employed','management',
                      'unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True )
    return data


def experiment_generator(train_feats, train_class):
    # Initialize the plotting sets for later use
    accuracy, penalties = [], []
    # Set your G & C parameters
    G = .00000001 # Should be really small to allow better curviture values
    penalty = 1 # Use large value to minimize the prediction error 
    N = 10 # Number of experiments

    
    for item in range(N):
        # Create a new classifier using the G & C parameters 
        clf = SVC(kernel='rbf', random_state = 0, gamma = G, C = penalty, probability=True)
        # Train the rbf classifier using training features and training class
        clf.fit(train_feats, train_class.values.ravel())
        # Make prediction using training features 
        pred_train = clf.predict(train_feats)
        # Accuracy score
        s_train = accuracy_score(train_class, pred_train)
        # Store values for plotting
        penalties.append(penalty)
        accuracy.append(s_train)
        # Increase experiment parameters
        penalty += 1
        G += .00000001
    # Initialize plot for accuracy and penalty (C)
    plt.scatter(penalties, accuracy)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Penalty - C Parameter')
    plt.show()

data = normalize(data)
plt.hist((data.duration),bins=100)
plt.ylabel('Occurences (Frequency)')
plt.xlabel('Client Call Duration')
plt.show()
plt.hist((data.job),bins=10)
plt.ylabel('Occurences (Frequency)')
plt.xlabel('Client Job Indices')
plt.show()
plt.hist((data.balance),bins=10)
plt.ylabel('Occurences (Frequency)')
plt.xlabel('Client Balance')
plt.show()

# Create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, data.y, test_size=0.2)

# Debugging 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Define your initial features
df_train = X_train 
df_test = X_test

# Initialize a dataframe using the target training y column 
df_train_class = pd.DataFrame(df_train['y'])

# Set the features to the rest of the columns (columns that are not 'y')
df_train_features = df_train.loc[:, df_train.columns != 'y']

# Initialize a dataframe using the target test y column
df_test_class = pd.DataFrame(df_test['y'])

# Set the features to the rest of the columns (columns that are not 'y')
df_test_features = df_test.loc[:, df_test.columns != 'y']

# Set gamma and c parameters for the svm rbf kernel
g = .0001 # gamma (curviture)
c = 1 # penalty - prediction marginal error

# Create an svm classifier with an rbf kernel using gamma and c parameters
mlp_classifier = SVC(kernel='rbf', random_state = 0, gamma = g, C = c, probability=True)
# Training the classifier
mlp_classifier.fit(df_train_features, df_train_class.values.ravel())

# Make a prediction using the training features                      
predicted_train = mlp_classifier.predict(df_train_features)

# Make a prediction using the testing features
predicted_test = mlp_classifier.predict(df_test_features)

# Accuracy score for training 
score_train = accuracy_score(df_train_class, predicted_train)

# Accuracy score for testing
score_test = accuracy_score(df_test_class, predicted_test)

# Display corresponding accuracies for training and testing
print('Training Accuracy Score: {}'.format(score_train))
print('Testing Accuracy Score: {}'.format(score_test))
   
# Precision, Recall  
precision_train = precision_score(df_train_class, predicted_train)
precision_test = precision_score(df_test_class, predicted_test)
print('Training Precision: {}'.format(precision_train))
print('Testing Precision: {}'.format(precision_test))

recall_train = recall_score(df_train_class, predicted_train)
recall_test = recall_score(df_test_class, predicted_test)

print('Training Recall: {}'.format(recall_train))
print('Testing Recall: {}'.format(recall_test))

# Classification Report
print('Training Classification Report: ')
print(classification_report(df_train_class, predicted_train))
print('Testing Classification Report: ')
print(classification_report(df_test_class, predicted_test))

# F1 Score
f1score_train = f1_score(df_train_class, predicted_train)
f1score_test = f1_score(df_test_class, predicted_test)

print("Training F1score: {}".format(f1score_train))
print("Testing F1score: {}".format(f1score_test))

f1score_train = f1_score(df_train_class, predicted_train, average='weighted')
f1score_test = f1_score(df_test_class, predicted_test, average='weighted')

print("Training Weighted F1score: {}".format(f1score_train))
print("Testing Weighted F1score: {}".format(f1score_test))

predicted_prob_train = mlp_classifier.predict_proba(df_train_features)
predicted_prob_test = mlp_classifier.predict_proba(df_test_features)    
    
# ROC-AUC
roc_y_n = 'y'
fpr, tpr, threshold = roc_curve(df_train_class, predicted_prob_train[:,1])
roc_auc_train = auc(fpr, tpr)

print("Training AUC for ROC: {}".format(roc_auc_train))

plt.figure()
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_train)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Training - Receiver Operating Characteristic')
plt.show()

fpr, tpr, threshold = roc_curve(df_test_class, predicted_prob_test[:,1])
roc_auc_test = auc(fpr, tpr)

print("Testing AUC for ROC: {}".format(roc_auc_test))

plt.figure()
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_test)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.title('Testing - Receiver Operating Characteristic')
plt.show()
# Create new experiments 
experiment_generator(df_train_features, df_train_class)



