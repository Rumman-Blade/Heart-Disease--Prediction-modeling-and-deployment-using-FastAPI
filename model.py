import numpy as np # Linear Algebra
import pandas as pd # Handling DataFrame
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sns # Data Visualization
plt.style.use(style='fivethirtyeight')
#%matplotlib inline

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from tabulate import tabulate
import pickle

from scipy import stats # Statistical Analysis
import re
import warnings # To mitigate any unwanted warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('https://raw.githubusercontent.com/ds-mahbub/24MLE01_Machine-Learning-Engineer/KNN/Classification/data/heart_disease.csv')

def rename_columns(data):
    new_columns = []
    for column in data.columns:
        new_column_name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', column).lower()
        new_columns.append(new_column_name)
    data.columns = new_columns
rename_columns(data)

num_cols = data.select_dtypes(include = ['number']).columns.tolist()
cat_cols = data.select_dtypes(include = ['object', 'category']).columns.tolist()

df1= data.copy()

df1 = df1.replace({'No': 0, 'Yes': 1})  # this will replace all the "No" with 0 and 'Yes' with 1
df1["sex"] = df1["sex"].replace({'Female': 0, 'Male': 1}) # 'Female' with zero and male  with 1

# diabetic mapping
diabetic_mapping = {
    'No': 0,
    'No, borderline diabetes': 0,
    'Yes': 1,
    'Yes (during pregnancy)': 1
}

race_mapping = {
    'American Indian/Alaskan Native': 0,
    'Asian': 1,
    'Black': 2,
    'Hispanic': 3,
    'Other': 4,
    'White': 5
}

gen_health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

age_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80 or older': 12
}

# Executing all the mappings

df1['diabetic'] = df1['diabetic'].replace(diabetic_mapping).astype(int)
df1['race'] = df1['race'].map(race_mapping).astype(int)
df1['gen_health'] = df1['gen_health'].replace(gen_health_mapping).astype(int)
df1['age_category'] = df1['age_category'].map(age_mapping).astype(int)

pearson_corr = df1.corr(method='pearson')
df2 = df1.copy(deep = True)

X = df2.drop(['heart_disease'], axis=1)
y = df2['heart_disease']

scaler = MinMaxScaler(feature_range = (0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)


smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

decom = PCA(svd_solver='auto') 
X_pca = decom.fit_transform(X_scaled)
ex_var = np.cumsum(np.round(decom.explained_variance_ratio_, 2) * 100)

rf = RandomForestClassifier(n_estimators= 200, 
                            min_samples_split=5, 
                            min_samples_leaf= 1, 
                            max_depth=30, 
                            criterion= 'gini',
                            bootstrap= False)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

feature_importance_values = rf.feature_importances_


feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance_values})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

y_pred_prob = rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = auc(fpr, tpr)

with open('rf.pkl', 'wb') as f:
    pickle.dump(rf, f)

