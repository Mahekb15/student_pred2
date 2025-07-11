import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load('placement_model.pkl')

# Streamlit page settings
st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🎓 Student Placement Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter student details to predict placement and visualize your profile.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input sliders
cgpa = st.slider("📊 CGPA (0 - 10)", 0.0, 10.0, 7.0, 0.1)
internships = st.slider("💼 Number of Internships", 0, 5, 1)
projects = st.slider("📁 Number of Projects", 0, 5, 2)
backlogs = st.slider("🚫 Number of Backlogs", 0, 5, 0)
communication = st.slider("🗣️ Communication Skill (0 - 10)", 0, 10, 7)
certifications = st.slider("📜 Number of Certifications", 0, 5, 2)

# Predict button
if st.button("🔍 Predict Placement"):
    # Prepare input data
    columns = ['CGPA', 'Internships', 'Projects', 'Backlogs', 'Communication', 'Certifications']
    new_data = pd.DataFrame([[cgpa, internships, projects, backlogs, communication, certifications]], columns=columns)

    # Predict placement
    prediction = model.predict(new_data)[0]

    # Show result
    st.markdown("## 🎯 Prediction Result:")
    if prediction == 1:
        st.success("✅ The student is likely to be **Placed**")
    else:
        st.error("❌ The student is likely **Not Placed**")

    # Show pie chart and bar chart
    st.markdown("## 📊 Statistical Profile Visualizations")

    # Prepare data
    values = [cgpa, internships, projects, backlogs, communication, certifications]
    labels = ['CGPA', 'Internships', 'Projects', 'Backlogs', 'Communication', 'Certifications']
    colors = sns.color_palette("pastel")

    # Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig1)

    # Bar Charts
    st.markdown("### 📈 Feature Scores Bar Chart")
    fig2, axs = plt.subplots(2, 3, figsize=(14, 8))

    sns.barplot(x=["CGPA"], y=[cgpa], ax=axs[0, 0], palette="Blues_d")
    axs[0, 0].set_title("CGPA")

    sns.barplot(x=["Internships"], y=[internships], ax=axs[0, 1], palette="Greens_d")
    axs[0, 1].set_title("Internships")

    sns.barplot(x=["Projects"], y=[projects], ax=axs[0, 2], palette="Oranges_d")
    axs[0, 2].set_title("Projects")

    sns.barplot(x=["Backlogs"], y=[backlogs], ax=axs[1, 0], palette="Reds_d")
    axs[1, 0].set_title("Backlogs")

    sns.barplot(x=["Communication"], y=[communication], ax=axs[1, 1], palette="Purples_d")
    axs[1, 1].set_title("Communication Skill")

    sns.barplot(x=["Certifications"], y=[certifications], ax=axs[1, 2], palette="Greys_d")
    axs[1, 2].set_title("Certifications")

    for ax in axs.flat:
        ax.set_ylabel("Score")
        ax.set_ylim(0, 10)

    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9rem;'>Made with ❤️ by Mahek Bhathawala</p>", unsafe_allow_html=True)
