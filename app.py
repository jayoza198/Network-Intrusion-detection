# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page title
st.title('Network Intrusion Detection System') 

# Sidebar for uploading CSV data file
st.sidebar.header('Upload Input CSV data')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier', ('KNN', 'Logistic Regression', 'Naive Bayes', 'Decision Tree'))

if st.sidebar.button('Classify'):
    # Get the predictions
    if classifier == 'KNN':
        pred = knn_model.predict(input_df)
    elif classifier == 'Logistic Regression':
        pred = logreg_model.predict(input_df)
    elif classifier == 'Naive Bayes': 
        pred = nb_model.predict(input_df)
    else:
        pred = dt_model.predict(input_df)
        
    st.write('Prediction: ', pred)

# Load the data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info('Upload input CSV file')
    st.stop()

# Data preprocessing 
# Separate num & cat columns
num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Scale numerical cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encode categorical cols
le = LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform) 

# Feature selection
X = df.drop('class', axis=1) 
y = df['class']

rfc = RandomForestClassifier()
rfc.fit(X, y)

# Select top 15 features
rf_feat_imp = pd.DataFrame({'feature': X.columns, 'importance': rfc.feature_importances_})
top_feats = rf_feat_imp.sort_values('importance', ascending=False).head(15)['feature']
X = df[top_feats]

# Split data into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the models
knn_model = KNeighborsClassifier(n_jobs=-1).fit(X_train, y_train) 
logreg_model = LogisticRegression(n_jobs=-1, random_state=0).fit(X_train, y_train)
nb_model = BernoulliNB().fit(X_train, y_train)
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
