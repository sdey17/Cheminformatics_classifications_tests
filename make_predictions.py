'''
This script loads the previously created training and test sets, uses different ML models to run n-fold cross-validations and make predictions on the external test sets.
'''

import pickle
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem, PandasTools
from statistics import mean, stdev
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#clf = BernoulliNB()
#clf = RandomForestClassifier()
clf = xgb.XGBClassifier()

model = 'XG'

'''
Define the range (in μM) to be used for active/inactive classification
'''

active_range=1
inactive_range=10

def computeMorganFP(mol, depth=2, nBits=1024):
    a = np.zeros(nBits)
    try:
      DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,depth,nBits),a)
    except:
      return None
    return a

'''
Load the training and test sets 
'''

training_set = pd.read_csv('training_set_{}_{}.csv'.format(active_range, inactive_range), index_col=None)
print('Training set',training_set['Activity'].value_counts())
PandasTools.AddMoleculeColumnToFrame(frame=training_set, smilesCol='SMILES', molCol='Molecule')
training_set['Morgan2FP'] = training_set['Molecule'].map(computeMorganFP)
X_train_ext = training_set.Morgan2FP
y_train_ext = training_set.Activity

ext_test_set = pd.read_csv('ext_test_set_{}_{}.csv'.format(active_range, inactive_range), index_col=None)
print('External Test Set', ext_test_set['Activity'].value_counts())
PandasTools.AddMoleculeColumnToFrame(frame=ext_test_set, smilesCol='SMILES', molCol='Molecule')
ext_test_set['Morgan2FP'] = ext_test_set['Molecule'].map(computeMorganFP)
X_test_ext = ext_test_set.Morgan2FP
y_test_ext = ext_test_set.Activity

folds = []

accuracy = []
sensitivity = []
specificity = []
mcc=[]
roc_auc = []

accuracy_ext = []
sensitivity_ext = []
specificity_ext = []
mcc_ext = []
roc_auc_ext = []

'''
Model building and testing on different external test sets
'''

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=125)
labels = [0,1]

for i, (train_index, test_index) in enumerate(skf.split(X_train_ext, y_train_ext)):
    folds.append({'train':train_index,'test':test_index})

for i in range (len(folds)):
    X_train = pd.DataFrame(X_train_ext).iloc[folds[i]['train']].to_numpy().ravel()
    X_test = pd.DataFrame(X_train_ext).iloc[folds[i]['test']].to_numpy().ravel()
    y_train = pd.DataFrame(y_train_ext).iloc[folds[i]['train']].to_numpy().ravel()
    y_test = pd.DataFrame(y_train_ext).iloc[folds[i]['test']].to_numpy().ravel()

    clf.fit(list(X_train), y_train)

    y_pred = clf.predict(list(X_test))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=labels).ravel()
    accuracy.append((tp+tn)/(tp+tn+fp+fn))   
    sensitivity.append(tp/(tp+fn))
    specificity.append(tn/(tn+fp))
    roc_auc.append(roc_auc_score(y_pred, y_test))
    mcc.append(matthews_corrcoef(y_test, y_pred))

    y_pred_ext = clf.predict(list(X_test_ext))
    tn, fp, fn, tp = confusion_matrix(y_test_ext, y_pred_ext, labels=labels).ravel()
    accuracy_ext.append((tp+tn)/(tp+tn+fp+fn))   
    sensitivity_ext.append(tp/(tp+fn))
    specificity_ext.append(tn/(tn+fp))
    roc_auc_ext.append(roc_auc_score(y_test_ext, y_pred_ext))
    mcc_ext.append(matthews_corrcoef(y_test_ext, y_pred_ext))

metrics = pd.DataFrame(index=['Accuracy','Sensitivity','Specificity','ROC-AUC','MCC'])

metrics['Five-fold CV'] = [mean(accuracy), mean(sensitivity), mean(specificity), mean(roc_auc), mean(mcc)]
metrics['Five-fold CV SD'] = [stdev(accuracy), stdev(sensitivity), stdev(specificity), stdev(roc_auc), stdev(mcc)]

metrics['Ext Test Set'] = [mean(accuracy_ext), mean(sensitivity_ext), mean(specificity_ext), mean(roc_auc_ext), mean(mcc_ext)]
metrics['Ext Test Set SD'] = [stdev(accuracy_ext), stdev(sensitivity_ext), stdev(specificity_ext), stdev(roc_auc_ext), stdev(mcc_ext)]

metrics.to_csv('Metrics_{}_{}_{}.csv'.format(model, active_range, inactive_range))

with open('{}_{}_{}.pkl'.format(model, active_range, inactive_range),'wb') as f:
    pickle.dump(clf,f)

