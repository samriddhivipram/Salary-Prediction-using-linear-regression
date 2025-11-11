import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="₹",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header{
        font-size:48px;
        font-weight:bold;
        color:#1f77b4;
        text-align:center;
        margin-bottom:10px;
    }
    .sub-header{
        font-size:20px;
        color:#666;
        text-align:center;
        margin-bottom:30px;
    }
    .prediction-box{
        background-color:#f0f8ff;
        padding:30px;
        border-radius:10px;
        border:2px solid #1f77b4;
        text-align:center;
        margin:20px 0;
    }
    .salary-amount{
        font-size:48px;
        font-weight:bold;
        color:#1f77b4;
    }
    .info-box{
        background-color:#fff3cd;
        padding:15px;
        border-radius:5px;
        border-left:5px solid #ffc107;
        margin:20px 0;
    }
    </style>
""",unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained model and encoders"""
    try:
        model=joblib.load('salary_model.pkl')
        encoders=joblib.load('label_encoders.pkl')
        return model,encoders
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter notebook 'model_training.ipynb' first to train the model.")
        st.stop()

@st.cache_data
def load_data():
    """Load the salary dataset"""
    try:
        df = pd.read_csv('salary_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found! Please ensure 'salary_data.csv' exists.")
        st.stop()

def get_unique_values(df):
    """Extract unique values for dropdowns"""
    job_titles = sorted(df['Job_Title'].unique())
    education_levels = sorted(df['Education_Level'].unique())
    locations = sorted(df['Location'].unique())
    industries = sorted(df['Industry'].unique())
    return job_titles,education_levels,locations,industries

def predict_salary(model,encoders,job_title,experience,education,location,industry):
    """Make salary prediction"""
    input_data = pd.DataFrame({
        'Job_Title':[job_title],
        'Years_of_Experience':[experience],
        'Education_Level':[education],
        'Location':[location],
        'Industry':[industry]
    })
    
    for col in ['Job_Title','Education_Level','Location','Industry']:
        input_data[col] = encoders[col].transform(input_data[col])
    
    prediction = model.predict(input_data)[0]
    return prediction

st.markdown('<div class="main-header">Salary Predictor</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict your expected salary based on job profile and experience</div>',unsafe_allow_html=True)

model, encoders = load_models()
df = load_data()
job_titles,education_levels,locations,industries = get_unique_values(df)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Enter Your Details")
    
    selected_job = st.selectbox(
        "Job Title",
        options=job_titles,
        help="Select your job role or the position you're applying for"
    )
    
    years_exp = st.slider(
        "Years of Experience",
        min_value=0,
        max_value=20,
        value=3,
        help="Total years of professional experience"
    )
    
    selected_education = st.selectbox(
        "Education Level",
        options=education_levels,
        help="Highest degree or qualification obtained"
    )

with col2:
    st.subheader("🌍 Location & Industry")
    
    selected_location = st.selectbox(
        "Location",
        options=locations,
        help="City where you work or plan to work"
    )
    
    selected_industry = st.selectbox(
        "Industry",
        options=industries,
        help="Industry sector of your job"
    )
    
    st.write("")
    st.write("")
    st.write("")

st.markdown("---")

if st.button("🔍 Predict Salary", use_container_width=True):
    with st.spinner("Calculating your estimated salary..."):
        predicted_salary = predict_salary(
            model,encoders,selected_job,years_exp,selected_education,
            selected_location,selected_industry
        )
        
        st.markdown('<div class="prediction-box">',unsafe_allow_html=True)
        st.markdown("### Your Estimated Annual Salary")
        st.markdown(f'<div class="salary-amount">₹ {predicted_salary:,.0f}</div>', unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        
        monthly_salary = predicted_salary/12
        st.success(f"💵 **Monthly Salary:** ₹ {monthly_salary:,.0f}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Annual Package",f"₹ {predicted_salary/100000:.2f} LPA")
        
        with col2:
            st.metric("Monthly Salary",f"₹ {monthly_salary:,.0f}")
        
        with col3:
            daily_salary = predicted_salary/365
            st.metric("Daily Rate",f"₹ {daily_salary:,.0f}")
        
        similar_jobs = df[
            (df['Job_Title'] == selected_job) & 
            (df['Location'] == selected_location)
        ]['Salary_INR']
        
        if len(similar_jobs)>0:
            avg_salary = similar_jobs.mean()
            min_salary = similar_jobs.min()
            max_salary = similar_jobs.max()

st.markdown("---")
st.markdown('<div class="info-box">',unsafe_allow_html=True)
st.markdown("""
**💡 Note:** This prediction is based on Sample Data. 
Actual salaries may vary based on company size, specific skills, certifications, and market conditions.
""")
st.markdown('</div>',unsafe_allow_html=True)
