import pandas as pd
import numpy as np
import re
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

with open('only_viruses.pkl','rb') as f:
    only_viruses = pickle.load(f)

with open('only_clear.pkl','rb') as f:
    only_clear = pickle.load(f)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

class ExtensionCountTransformer():
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    df = pd.DataFrame(data=[0]*X.shape[0], columns=['count_of_exe'])
    df['count_of_drv'] = 0
    for index in range(X.shape[0]):
      lst = list(X.iloc[index].split(','))
      format = list(chain.from_iterable(list(map(lambda x: re.findall(r'\.[a-z]{3}', x), lst))))
      for i in format:
        if i == '.exe':
          df.loc[index, 'count_of_exe'] += 1
        elif i == '.drv':
          df.loc[index, 'count_of_drv'] += 1
    return df

class FeatureExtractionTransformer():
    def __init__(self, viruses, clear):
      self.only_viruses = viruses
      self.only_clear = clear


    def fit(self, X, y=None):
      return self
    

    def transform(self, X, y=None):
      count_viruses = []
      count_clear = []
      count_libs = []
      average_len_libs = []

      for index in range(X.shape[0]):
        libs_preproc = list(map(lambda x: re.sub(r'\.(.+)$', r'', x).strip(), list(X.iloc[index].split(','))))
        virus_libs = 0
        clear_libs = 0
        len_libs = []
        for elem_libs in libs_preproc:
          if elem_libs in self.only_viruses:
            virus_libs += 1
          if elem_libs in self.only_clear:
            clear_libs += 1
          len_libs.append(len(elem_libs))
        average_len_libs.append(np.mean(len_libs))
        count_viruses.append(virus_libs)
        count_clear.append(clear_libs)
        count_libs.append(len(libs_preproc))

      df = pd.DataFrame(data=count_viruses, columns=['count_viruses'])
      df['count_clear'] = count_clear
      df['count_libs'] = count_libs
      df['average_len_libs'] = average_len_libs
      return df

valid = pd.read_csv('val.tsv', sep='\t')
X = valid.drop(['is_virus'], axis=1)
y = valid.is_virus

feature_pipe = Pipeline(
    steps=[
        ('Feature_extraction', FeatureExtractionTransformer(viruses=only_viruses, clear=only_clear)),
        ('scaler', StandardScaler())
    ]
)

ext_pipe = Pipeline(
    steps=[
        ('Count_extensions', ExtensionCountTransformer()),
        ('scaler', StandardScaler())
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('feature_ext', feature_pipe, 'libs'),
        ('extensions', ext_pipe, 'libs')
    ]
)

prep = preprocessor
X_trans = prep.fit_transform(X)
y_pred = model.predict(X_trans)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)


with open('validation.txt','w') as file:
    file.write('True positive: {}\n'.format(tp))
    file.write('False positive: {}\n'.format(fp))
    file.write('False negative: {}\n'.format(fn))
    file.write('True negative: {}\n'.format(tn))
    file.write('Accuracy: {}\n'.format(acc))
    file.write('Precision: {}\n'.format(prec))
    file.write('Recall: {}\n'.format(rec))
    file.write('F1: {}\n'.format(f1))
    