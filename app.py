import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Harga Mobil Listrik", layout="wide")

# =========================
# 🔧 Load Model & Feature Info
# =========================
@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_best_model_with_make.pkl")
    make_columns = joblib.load("make_columns.pkl")  # List of make used during training
    return model, make_columns

model, make_columns = load_resources()

# =========================
# 🎯 App Title & Deskripsi
# =========================
st.title("🚗 Prediksi Harga Mobil Listrik (Base MSRP)")
st.markdown("""
Model ini menggunakan algoritma **Random Forest Regressor** untuk memprediksi harga mobil listrik berdasarkan:
- Jarak tempuh listrik (*Electric Range*)
- Tahun kendaraan (*Model Year*)
- Umur kendaraan (*Vehicle Age*)
- Merk kendaraan (*Make*)
""")
# Penjelasan tambahan tentang Base MSRP
st.info("""
💡 **Apa itu Base MSRP?**
Base MSRP (**Manufacturer’s Suggested Retail Price**) adalah harga yang disarankan produsen saat kendaraan **pertama kali dirilis** — bukan harga pasar saat ini.
Artinya:
- Mobil lama bisa memiliki harga MSRP tinggi jika dulu termasuk mobil mewah.
- Mobil baru bisa lebih murah jika dirancang untuk pasar massal.

Model ini memprediksi harga **berdasarkan saat peluncuran**, bukan harga bekas atau pasar saat ini.
""")

# =========================
#   Input Form
# =========================
st.header("📋 Input Spesifikasi Kendaraan")

col1, col2 = st.columns(2)
with col1:
    electric_range = st.slider("🔋 Electric Range (miles)", 0, 500, 150, step=10)
with col2:
    model_year = st.slider("📅 Model Year", 2000, 2025, 2022)

vehicle_age = 2025 - model_year

make_selected = st.selectbox("🏷️ Pilih Merk Kendaraan (Make)", make_columns)

# =========================
#   Prediksi
# =========================
if st.button("🔍 Prediksi Harga"):
    # Buat vektor one-hot encoding untuk Make
    make_encoded = [1 if make == make_selected else 0 for make in make_columns]

    # Gabungkan semua fitur sesuai urutan saat training
    input_data = np.array([[electric_range, model_year, vehicle_age] + make_encoded])

    # Lakukan prediksi
    prediction = model.predict(input_data)[0]

    # =========================
    # 💰 Tampilkan Hasil
    # =========================
    st.success(f"💰 Estimasi Harga Kendaraan: **${prediction:,.2f}**")

    st.markdown(f"""
    **Detail Input:**
    - 🔋 Electric Range: `{electric_range} miles`
    - 📅 Model Year: `{model_year}`
    - ⏳ Vehicle Age: `{vehicle_age} tahun`
    - 🏷️ Merk: `{make_selected}`
    """)

# =========================
#   Footer
# =========================
st.markdown("---")
st.markdown("📊 Model dilatih dengan data kendaraan listrik di AS dengan rentang tahun dan harga bervariasi.")
