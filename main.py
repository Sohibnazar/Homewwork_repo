import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
 
@st.cache_data
def load_data():
    df = pd.read_csv("bank.csv", sep=';')
    return df

df = load_data()
st.title("Прогноз банковской кампании (KNN)")
st.subheader("Исходные данные")
st.dataframe(df.head())
 
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
target_col = 'y'

st.sidebar.header("Настройка признаков")
selected_features = st.sidebar.multiselect(
    "Выберите признаки для обучения модели",
    options=numeric_cols + categorical_cols,
    default=numeric_cols + categorical_cols
)
 
X = df[selected_features]
y = df[target_col].str.strip().str.lower().map({'yes':1, 'no':0})
 
X_le = X.copy()
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_le[col] = le.fit_transform(X[col])

scalers =StandardScaler()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_le[num_cols] = scaler.fit_transform(X_le[num_cols])
 
X_train, X_test, y_train, y_test = train_test_split(
    X_le, y, test_size=0.2, stratify=y, random_state=42
)
 
st.sidebar.header("Гиперпараметры KNN")
n_neighbors = st.sidebar.slider("Количество соседей (n_neighbors)", 1, 15, 5)
weights = st.sidebar.selectbox("Вес соседей (weights)", ["uniform", "distance"])
metric = st.sidebar.selectbox("Метрика расстояния (metric)", ["euclidean", "manhattan"])
 
knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]
 
st.subheader("Метрики модели на тестовой выборке")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
 
st.subheader("Прогноз для нового пользователя")

input_data = {}
for col in selected_features:
    if col in numeric_cols:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    else:
        val = st.selectbox(f"{col}", df[col].unique())
    input_data[col] = val

if st.button("Сделать прогноз"):
    input_df = pd.DataFrame([input_data]) 
    for col in input_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        le.fit(df[col])
        input_df[col] = le.transform(input_df[col])
    for col in input_df.select_dtypes(include=[np.number]).columns:
        input_df[col] = scaler.transform(input_df[[col]])
    
    prediction = knn.predict(input_df)[0]
    probability = knn.predict_proba(input_df)[0,1]
    
    st.write(f"Прогноз: {'YES' if prediction==1 else 'NO'}")
    st.write(f"Вероятность: {probability:.4f}")

