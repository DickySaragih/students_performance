# -*- coding: utf-8 -*-
"""student_performance"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Pengaturan Tampilan Plot ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- Fungsi Memuat dan Membersihkan Data ---
@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("'", '').str.replace('(', '').str.replace(')', '')
    df.rename(columns={'target': 'dropout_status'}, inplace=True)
    df = df.dropna()
    return df

# --- Fungsi Preprocessing ---
@st.cache_data
def preprocess_data(df):
    X = df.drop('dropout_status', axis=1)
    y = df['dropout_status']

    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=np.number).columns

    encoders = {}
    for col in categorical_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col])
        encoders[col] = le_col

    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    return X_train, X_test, y_train, y_test, X.columns, le_y, scaler, encoders, categorical_cols, numeric_cols

# --- Fungsi Pelatihan Model ---
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Fungsi Visualisasi ---
def plot_distribution(df, col, title, xlabel, ylabel, hue=None, palette='Set2', order=None):
    plt.figure(figsize=(10, 6))
    if df[col].dtype == 'object' or df[col].dtype == 'category':
        sns.countplot(data=df, x=col, hue=hue, palette=palette, order=order)
        plt.xticks(rotation=45, ha='right')
    else:
        sns.histplot(data=df, x=col, hue=hue, kde=True, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hue:
        plt.legend(title='Status')
    plt.tight_layout()
    st.pyplot(plt)

def plot_boxplot(df, x_col, y_col, title, xlabel, ylabel, palette='Set2'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x_col, y=y_col, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(plt)

def display_classification_report(y_test, y_pred, label_encoder, present_labels):
    try:
        target_names = label_encoder.inverse_transform(present_labels)
    except ValueError:
        target_names = [str(lbl) for lbl in present_labels]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    st.subheader("Top 10 Most Important Features")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Most Important Features for Dropout Prediction')
    plt.tight_layout()
    st.pyplot(plt)

# --- Streamlit App ---
st.title("Prediksi Status Dropout Mahasiswa")
st.markdown("""
Aplikasi ini memuat data performa siswa, melakukan preprocessing, melatih model Random Forest,
dan menampilkan hasil analisis serta prediksi status dropout.
""")

# URL CSV atau path lokal
url = 'https://raw.githubusercontent.com/DickySaragih/data_science_02/main/Students_Performance.csv'
df = load_and_clean_data(url)

if st.sidebar.checkbox("Tampilkan Data Mentah"):
    st.subheader("Data Mentah (5 Baris Pertama)")
    st.dataframe(df.head())
    st.subheader("Distribusi Target")
    st.write(df['dropout_status'].value_counts())

st.header("Eksplorasi Data (EDA)")
st.subheader("1. Distribusi Status Dropout Mahasiswa")
plot_distribution(df, 'dropout_status', "Distribusi Status Dropout Mahasiswa", "Status", "Jumlah Mahasiswa")

st.subheader("2. Distribusi Dropout Berdasarkan Gender")
plot_distribution(df, 'gender', "Distribusi Dropout Berdasarkan Gender", "Gender", "Jumlah Mahasiswa", hue='dropout_status')

st.subheader("3. Distribusi Dropout Berdasarkan Jenis Kursus")
order = df['course'].value_counts().index
plot_distribution(df, 'course', "Distribusi Dropout Berdasarkan Jenis Kursus", "Kursus", "Jumlah Mahasiswa", hue='dropout_status', order=order)

st.subheader("4. Dropout Berdasarkan Status Pernikahan")
plot_distribution(df, 'marital_status', "Dropout Berdasarkan Status Pernikahan", "Status Pernikahan", "Jumlah Mahasiswa", hue='dropout_status')

st.subheader("5. Usia saat Masuk Kuliah vs Status Dropout")
plot_boxplot(df, 'dropout_status', 'age_at_enrollment', "Usia saat Masuk Kuliah vs Status Dropout", "Status", "Usia Saat Enroll")

# --- Model dan Evaluasi ---
st.header("Model dan Evaluasi")
X_train, X_test, y_train, y_test, feature_names, le_y, scaler, encoders, cat_cols, num_cols = preprocess_data(df)
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
present_labels = np.unique(y_test)
try:
    display_labels = le_y.inverse_transform(present_labels)
except ValueError:
    display_labels = [str(i) for i in present_labels]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
fig, ax = plt.subplots()
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
st.pyplot(fig)

# --- Classification Report dan Feature Importance ---
display_classification_report(y_test, y_pred, le_y, present_labels)
plot_feature_importance(model, feature_names)

# --- Prediksi Individu ---
st.header("Prediksi Individu")
st.markdown("Masukkan data siswa untuk memprediksi apakah berpotensi dropout:")

# --- FORM INPUT SESUAI FITUR ---
input_data = {}

# Tampilkan input untuk semua kolom fitur model agar tidak error
for col in feature_names:
    if col in cat_cols:
        unique_vals = df[col].unique()
        input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", unique_vals)
    elif col in num_cols:
        min_val = int(df[col].min()) if pd.api.types.is_integer_dtype(df[col]) else float(df[col].min())
        max_val = int(df[col].max()) if pd.api.types.is_integer_dtype(df[col]) else float(df[col].max())
        default_val = int(df[col].median()) if pd.api.types.is_integer_dtype(df[col]) else float(df[col].median())
        input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=min_val, max_value=max_val, value=default_val)

input_df = pd.DataFrame([input_data])

# --- Encoding ---
for col in cat_cols:
    le = encoders[col]
    # Jika nilai input tidak ada di label encoder classes, tambahkan pengecekan agar tidak error
    if input_df.at[0, col] not in le.classes_:
        st.error(f"Nilai '{input_df.at[0, col]}' untuk fitur '{col}' tidak dikenali.")
        st.stop()
    input_df[col] = le.transform(input_df[col])

# --- Scaling ---
if len(num_cols) > 0:
    input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Urutkan kolom sesuai model ---
input_df = input_df[feature_names]

if st.button("Prediksi Dropout/Graduate"):
    try:
        pred = model.predict(input_df)
        result = le_y.inverse_transform(pred)
        st.success(f"Prediksi Status: {result[0]}")
    except Exception as e:
        st.error(f"Terjadi error saat memproses input: {e}")

st.markdown("---")
st.write("Aplikasi dibuat oleh Dicky Candid Saragih")
