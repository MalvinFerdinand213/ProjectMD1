import streamlit as st
import pandas as pd
import pickle

# Load model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Prediksi Produktivitas Pekerja Garmen")

st.markdown("Masukkan detail produksi di bawah ini untuk memprediksi actual productivity:")

# Input Form
with st.form("prediction_form"):
    date = st.date_input("Tanggal").strftime('%Y-%m-%d')
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    department = st.selectbox("Departemen", ['sewing', 'finishing'])
    day = st.selectbox("Hari", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday'])
    team = st.number_input("Team", min_value=1, max_value=100, value=1)
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.75)
    smv = st.number_input("SMV", min_value=0.0)
    wip = st.number_input("WIP", min_value=0)
    over_time = st.number_input("Over Time (menit)", min_value=0)
    incentive = st.number_input("Incentive", min_value=0.0)
    idle_time = st.number_input("Idle Time", min_value=0.0)
    idle_men = st.number_input("Idle Men", min_value=0)
    no_of_style_change = st.number_input("Style Changes", min_value=0)
    no_of_workers = st.number_input("Jumlah Pekerja", min_value=1)

    submit = st.form_submit_button("Prediksi")

# Prediksi
if submit:
    input_data = pd.DataFrame([[
        date, quarter, department, day, team, targeted_productivity,
        smv, wip, over_time, incentive, idle_time, idle_men,
        no_of_style_change, no_of_workers
    ]], columns=[
        'date', 'quarter', 'department', 'day', 'team', 'targeted_productivity',
        'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
        'no_of_style_change', 'no_of_workers'
    ])

    prediction = model.predict(input_data)
    st.success(f"Prediksi Produktivitas Aktual: *{prediction[0]:.4f}*")
