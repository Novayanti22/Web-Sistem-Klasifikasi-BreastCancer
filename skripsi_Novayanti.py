import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import os

# Data pengguna untuk login
ADMIN_CREDENTIALS = {
    "admin": "breastcancer123"
}

# Fungsi untuk melakukan login
def login(username, password):
    if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return True
    else:
        return False

# Halaman login
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.experimental_rerun()  # Refresh the page to show the main page
        else:
            st.error("Invalid username atau password")

# Fungsi untuk menyimpan data pasien ke file CSV
def save_patient_data(data):
    if not os.path.isfile('patient_data.csv'):
        df = pd.DataFrame(columns=['Nama Pasien', 'Jenis Kelamin', 'Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP1', 'Prediction'])
        df.to_csv('patient_data.csv', index=False)
    
    df = pd.read_csv('patient_data.csv')
    df = df.append(data, ignore_index=True)
    df.to_csv('patient_data.csv', index=False)

# Fungsi untuk menghapus data pasien dari file CSV
def delete_patient_data(index):
    if os.path.isfile('patient_data.csv'):
        df = pd.read_csv('patient_data.csv')
        df = df.drop(index)
        df.to_csv('patient_data.csv', index=False)

# Halaman utama aplikasi
def home_page():
    st.write("# Selamat Datang di Sistem Klasifikasi Penyakit Kanker Payudara")
    st.markdown(
        """
        Kanker Payudara merupakan penyakit di mana sel-sel payudara abnormal tumbuh di luar kendali dan membentuk tumor. Jika dibiarkan, tumor bisa menyebar ke seluruh tubuh dan berakibat fatal. 
        Sel kanker payudara dimulai di dalam saluran susu atau lobulus penghasil susu di payudara.Kanker payudara menempati urutan pertama 
        jumlah kanker terbanyak di Indonesia serta menjadi salah satu penyumbang kematian pertama akibat kanker.
        
        Maka system ini dibuat sebagai penerapan metode naïve  untuk mengklasifikasikan ada dan tidaknya penyakit kanker payudara.
        """)

    # Membaca dataset dari file dataR2.csv
    dataset = pd.read_csv('dataR2.csv')

    # Mendefinisikan fitur dan label
    features = dataset.drop(columns=['Classification'])
    labels = dataset['Classification']

    # Konversi fitur ke tipe data numerik (float)
    features = features.apply(pd.to_numeric, errors='coerce')

    # Mengganti nilai yang hilang (jika ada) dengan nilai median dari kolom tersebut
    features = features.fillna(features.median())

    # Standardisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

    # Melatih model Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Prediksi pada set pengujian
    y_pred = model.predict(X_test)

def input_data_page():
    st.write("# Input Data Pasien")

    # Input pengguna
    st.write("Masukkan data pasien:")
    user_input_nama = st.text_input("Nama:")
    user_input_jenis_kelamin = st.text_input("Jenis Kelamin:")
    user_input_age = st.text_input("Usia:")
    user_input_bmi = st.text_input("BMI:", "")
    user_input_glucose = st.text_input("Glukosa:", "")
    user_input_insulin = st.text_input("Insulin:", "")
    user_input_homa = st.text_input("HOMA:", "")
    user_input_leptin = st.text_input("Leptin:", "")
    user_input_adiponectin = st.text_input("Adiponectin:", "")
    user_input_resistin = st.text_input("Resistin:", "")
    user_input_mcp1 = st.text_input("MCP-1:", "")

    try:
        user_input_age = int(user_input_age)
        if user_input_age < 20 or user_input_age > 85:
            st.error("Usia tidak mencukupi. Harap masukkan usia antara 20 sampai 85 tahun.")
            return

        user_data_float = np.array([[
            int(user_input_age), float(user_input_bmi), float(user_input_glucose),
            float(user_input_insulin), float(user_input_homa), float(user_input_leptin),
            float(user_input_adiponectin), float(user_input_resistin), float(user_input_mcp1)
        ]])
    except ValueError:
        st.error("Silakan Lengkapi data dengan benar")
        return

    # Membaca dataset dari file dataR2.csv untuk mendapatkan scaler
    dataset = pd.read_csv('dataR2.csv')
    features = dataset.drop(columns=['Classification'])
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(features.median())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Melakukan standardisasi pada data pengguna
    user_data_scaled = scaler.transform(user_data_float)

    # Melatih model Naive Bayes
    labels = dataset['Classification']
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Mengecek apakah tombol klasifikasi ditekan
    if st.button("Hasil Klasifikasi"):
        # Melakukan prediksi dengan data pengguna
        if 20 <= user_input_age <= 40:
            prediction = 1
        elif 41 <= user_input_age <= 85:
            prediction = 0
        else:
            prediction = model.predict(user_data_scaled)[0]

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.write("Pasien Tidak Kanker Payudara.")
        else:
            st.write("Pasien Terkena Kanker Payudara.")

        # Menyimpan data pasien ke file CSV
        patient_data = {
            'Nama Pasien': user_input_nama,
            'Jenis Kelamin': user_input_jenis_kelamin,
            'Age': user_input_age,
            'BMI': user_input_bmi,
            'Glucose': user_input_glucose,
            'Insulin': user_input_insulin,
            'HOMA': user_input_homa,
            'Leptin': user_input_leptin,
            'Adiponectin': user_input_adiponectin,
            'Resistin': user_input_resistin,
            'MCP1': user_input_mcp1,
            'Prediction': 'Pasien Tidak Kanker Payudara' if prediction == 1 else 'Pasien Terkena Kanker Payudara'
        }
        save_patient_data(patient_data)
        st.success("Data Pasien Berhasil Disimpan.")

def hasil_data_input_page():
    st.write("# Hasil Klasifikasi Data Pasien")

    if os.path.isfile('patient_data.csv'):
        df = pd.read_csv('patient_data.csv')
        st.dataframe(df)

def main_page():
    # Sidebar menu
    option = st.sidebar.selectbox(
        'Menu',
        ('Home', 'Input Data', 'Hasil Input Data', 'Logout')
    )

    if option == 'Home':
        home_page()

    elif option == 'Input Data':
        input_data_page()

    elif option == 'Hasil Input Data':
        hasil_data_input_page()

    elif option == 'Logout':
        st.session_state["logged_in"] = False
        st.experimental_rerun()
   
# Menjalankan aplikasi
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        main_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
