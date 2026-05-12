import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from streamlit_option_menu import option_menu

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD CUSTOM CSS
# =========================================================

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Clean TotalCharges column
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    return df

df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

#    st.image(
#       "assets/logo.png",
#       width=180
#   )

    st.markdown("## Customer Churn")

    selected = option_menu(
        menu_title="Menu",
        options=[
            "Dashboard",
            "Prediction",
            "Analytics",
            "About Model"
        ],
        icons=[
            "house",
            "activity",
            "bar-chart",
            "info-circle"
        ],
        menu_icon="cast",
        default_index=0
    )

# =========================================================
# DASHBOARD PAGE
# =========================================================

if selected == "Dashboard":

    st.title("📊 Customer Churn Intelligence Dashboard")

    st.markdown("""
    Analyze customer behavior, predict churn risk,
    and generate business insights using Machine Learning.
    """)

    st.markdown("---")

    # METRICS

    total_customers = len(df)

    churn_customers = len(
        df[df["Churn"] == "Yes"]
    )

    churn_rate = round(
        (churn_customers / total_customers) * 100,
        2
    )

    avg_monthly = round(
        df["MonthlyCharges"].mean(),
        2
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers",
            total_customers
        )

    with col2:
        st.metric(
            "Churn Rate",
            f"{churn_rate}%"
        )

    with col3:
        st.metric(
            "Avg Monthly Charges",
            f"${avg_monthly}"
        )

    st.markdown("---")

    # PIE CHART

    pie_chart = px.pie(
        df,
        names="Churn",
        title="Customer Churn Distribution",
        color_discrete_sequence=px.colors.sequential.Blues
    )

    st.plotly_chart(
        pie_chart,
        use_container_width=True
    )

    # TENURE HISTOGRAM

    hist_chart = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Tenure Distribution by Churn"
    )

    st.plotly_chart(
        hist_chart,
        use_container_width=True
    )

# =========================================================
# PREDICTION PAGE
# =========================================================

elif selected == "Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown("""
    Enter customer details below to predict customer churn probability.
    """)

    st.markdown("---")

    # LOAD MODEL COLUMNS
    model_columns = joblib.load("models/model_columns.pkl")

    # LOAD SCALER
    scaler = joblib.load("models/scaler.pkl")

    # =========================
    # USER INPUTS
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider("Tenure", 0, 72, 12)

        monthly_charges = st.slider(
            "Monthly Charges",
            0,
            200,
            70
        )

        total_charges = st.slider(
            "Total Charges",
            0,
            10000,
            2500
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button("Predict Churn"):

        input_dict = {
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # CREATE EMPTY DATAFRAME
        input_df = pd.DataFrame(
            np.zeros((1, len(model_columns))),
            columns=model_columns
        )

        # NUMERIC VALUES
        for key, value in input_dict.items():
            if key in input_df.columns:
                input_df[key] = value

        # ONE HOT ENCODING
        categorical_inputs = {
            f"Partner_Yes": partner == "Yes",
            f"Dependents_Yes": dependents == "Yes",
            f"PhoneService_Yes": phone_service == "Yes",
            f"MultipleLines_Yes": multiple_lines == "Yes",
            f"InternetService_Fiber optic": internet_service == "Fiber optic",
            f"InternetService_No": internet_service == "No",
            f"Contract_One year": contract == "One year",
            f"Contract_Two year": contract == "Two year",
            f"PaperlessBilling_Yes": paperless == "Yes",
            f"PaymentMethod_Credit card (automatic)": payment_method == "Credit card (automatic)",
            f"PaymentMethod_Electronic check": payment_method == "Electronic check",
            f"PaymentMethod_Mailed check": payment_method == "Mailed check"
        }

        for col, val in categorical_inputs.items():
            if col in input_df.columns:
                input_df[col] = int(val)

        # SCALE INPUT
        input_scaled = scaler.transform(input_df)

        # PREDICTION
        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠ High Churn Risk "
                f"({probability:.2%} probability)"
            )

            st.info("""
            ### Recommended Retention Strategies

            - Offer loyalty discounts
            - Provide premium support
            - Recommend annual plans
            - Send personalized offers
            """)

        else:

            st.success(
                f"✅ Customer likely to stay "
                f"({1 - probability:.2%} confidence)"
            )

# =========================================================
# ANALYTICS PAGE
# =========================================================

elif selected == "Analytics":

    st.title("📈 Customer Churn Analytics")

    st.markdown("""
    Explore customer trends and churn behavior
    using interactive visualizations.
    """)

    st.markdown("---")

    # CONTRACT TYPE ANALYSIS

    if "Contract" in df.columns:

        contract_chart = px.histogram(
            df,
            x="Contract",
            color="Churn",
            barmode="group",
            title="Contract Type vs Churn"
        )

        st.plotly_chart(
            contract_chart,
            use_container_width=True
        )

    # PAYMENT METHOD ANALYSIS

    if "PaymentMethod" in df.columns:

        payment_chart = px.histogram(
            df,
            x="PaymentMethod",
            color="Churn",
            title="Payment Method Analysis"
        )

        st.plotly_chart(
            payment_chart,
            use_container_width=True
        )

    # MONTHLY CHARGES

    monthly_chart = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn"
    )

    st.plotly_chart(
        monthly_chart,
        use_container_width=True
    )

# =========================================================
# ABOUT MODEL PAGE
# =========================================================

elif selected == "About Model":

    st.title("🤖 About the Machine Learning Model")

    st.markdown("""
    ## Model Information

    This project uses Machine Learning techniques
    to predict customer churn.

    ### Algorithms
    - Random Forest
    - XGBoost
    - Logistic Regression

    ### Features Used
    - Tenure
    - Monthly Charges
    - Total Charges
    - Senior Citizen
    - Contract Type
    - Payment Method
    - Internet Service

    ### Goal
    Help businesses reduce customer churn
    through predictive analytics.

    ### Tech Stack
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - Pandas
    """)

    st.markdown("---")

    st.success("Project successfully deployed using Streamlit Cloud 🚀")
