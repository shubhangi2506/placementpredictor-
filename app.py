import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------
# LOAD DATASET
# -----------------------

df = pd.read_csv("placement-dataset.csv")

# Remove unnecessary column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# -----------------------
# DECISION BOUNDARY FUNCTION
# -----------------------

def plot_decision_boundary(model, df):

    x_min = df["cgpa"].min() - 1
    x_max = df["cgpa"].max() + 1

    y_min = df["iq"].min() - 10
    y_max = df["iq"].max() + 10

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 1)
    )

    mesh_data = pd.DataFrame({
        "cgpa": xx.ravel(),
        "iq": yy.ravel()
    })

    Z = model.predict(mesh_data)

    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3
    )

    scatter = ax.scatter(
        df["cgpa"],
        df["iq"],
        c=df["placement"],
        s=70
    )

    ax.set_xlabel("CGPA")
    ax.set_ylabel("IQ")
    ax.set_title("Decision Boundary")

    return fig

# -----------------------
# HEADER
# -----------------------

st.title("🎓 AI Student Placement Predictor")

st.markdown("""
Predict placement chances using a Machine Learning model trained on student data.
""")

# -----------------------
# INPUT SECTION
# -----------------------

st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

with col2:
    iq = st.slider(
        "IQ Score",
        min_value=50,
        max_value=200,
        value=100
    )

input_data = pd.DataFrame({
    "cgpa": [cgpa],
    "iq": [iq]
})

# -----------------------
# PREDICTION
# -----------------------

if st.button("🚀 Predict Placement"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    placement_probability = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is likely NOT to be Placed")

    st.metric(
        "Placement Probability",
        f"{placement_probability:.2f}%"
    )

# -----------------------
# DATASET STATISTICS
# -----------------------

st.subheader("📊 Dataset Statistics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Students",
    len(df)
)

c2.metric(
    "Placed",
    len(df[df["placement"] == 1])
)

c3.metric(
    "Not Placed",
    len(df[df["placement"] == 0])
)

# -----------------------
# DATASET PREVIEW
# -----------------------

with st.expander("📄 View Dataset"):

    st.dataframe(df)

# -----------------------
# SCATTER PLOT
# -----------------------

st.subheader("📈 CGPA vs IQ Distribution")

fig1, ax1 = plt.subplots(figsize=(8, 5))

placed = df[df["placement"] == 1]
not_placed = df[df["placement"] == 0]

ax1.scatter(
    placed["cgpa"],
    placed["iq"],
    label="Placed",
    s=70
)

ax1.scatter(
    not_placed["cgpa"],
    not_placed["iq"],
    label="Not Placed",
    s=70
)

ax1.set_xlabel("CGPA")
ax1.set_ylabel("IQ")

ax1.legend()

st.pyplot(fig1)

# -----------------------
# DECISION BOUNDARY
# -----------------------

st.subheader("🧠 Machine Learning Decision Boundary")

boundary_fig = plot_decision_boundary(
    model,
    df
)

st.pyplot(boundary_fig)

# -----------------------
# SUMMARY
# -----------------------

with st.expander("📚 About This Project"):

    st.write("""
    This application uses Logistic Regression to predict student placement
    based on CGPA and IQ scores.

    Technologies Used:
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    """)