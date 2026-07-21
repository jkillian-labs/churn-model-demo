import streamlit as st
import pandas as pd
import pickle

# ── Load model and encoder ───────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    with open("churn_hgb_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)

    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


model, encoder = load_artifacts()

# ── Training Columns ─────────────────────────────────────────────────────────

TRAINING_COLUMNS = [
    'TOTAL_NUM_SESSIONS',
    'GROSS_TOTAL_SESSION_LENGTH',
    'ACTIVE_DAYS',
    'ACTIVE_QUARTERS',
    'ACTIVE_MONTHS',
    'ACTIVE_WEEKS',
    'SESSIONS_Q1',
    'SESSIONS_Q2',
    'SESSIONS_Q3',
    'SESSIONS_Q4',
    'AVG_SESSIONS_PER_ACTIVE_DAY',
    'AVG_MINUTES_PER_ACTIVE_DAY',
    'AVG_SESSION_LENGTH',
    'AVG_SESSIONS_PER_ACTIVE_QUARTER',
    'AVG_SESSIONS_PER_ACTIVE_MONTH',
    'ACTIVE_DAY_RATIO',
    'ACTIVE_QUARTER_RATIO',
    'ACTIVE_MONTH_RATIO',
    'DAYS_SINCE_LAST_ACTIVITY',
    'SESSION_GROWTH',
    'SESSION_GROWTH_RATIO',
    'RECENCY_WEIGHTED_SESSIONS',
    'AGE',
    'TECH_COMFORT_SCORE',
    'INCOME_SCORE',
    'EDUCATION_SCORE',
    'AFFLUENCE_SCORE',
    'PREMIUM_DIGITAL_CUSTOMER',
    'INCOME_LEVEL_Low',
    'INCOME_LEVEL_Medium',
    'INCOME_LEVEL_Very High',
    'EDUCATION_High School',
    'EDUCATION_Other',
    'EDUCATION_Post-Graduate',
    'DEVICE_TYPE_Mobile-only',
    'DEVICE_TYPE_Multi-device',
    'AGE_INCOME_SEGMENT_older_low_income',
    'AGE_INCOME_SEGMENT_young_high_income',
    'AGE_INCOME_SEGMENT_young_low_income',
    'AGE_TECH_SEGMENT_younger_low_tech',
    'INCOME_DEVICE_COMBO_High_Mobile-only',
    'INCOME_DEVICE_COMBO_High_Multi-device',
    'INCOME_DEVICE_COMBO_Low_Desktop-only',
    'INCOME_DEVICE_COMBO_Low_Mobile-only',
    'INCOME_DEVICE_COMBO_Low_Multi-device',
    'INCOME_DEVICE_COMBO_Medium_Desktop-only',
    'INCOME_DEVICE_COMBO_Medium_Mobile-only',
    'INCOME_DEVICE_COMBO_Medium_Multi-device',
    'INCOME_DEVICE_COMBO_Very High_Desktop-only',
    'INCOME_DEVICE_COMBO_Very High_Mobile-only',
    'INCOME_DEVICE_COMBO_Very High_Multi-device'
]

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("Healthy Meals Churn Prediction")
st.write(
    "Enter customer activity and demographic data to estimate renewal and churn probability."
)

age = st.number_input("Age", min_value=18, max_value=100, value=35)

income_level = st.selectbox(
    "Income Level",
    ["Low", "Medium", "High", "Very High"]
)

education = st.selectbox(
    "Education",
    ["High School", "Other", "Graduate", "Post-Graduate"]
)

device_type = st.selectbox(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"]
)

tech_comfort_score = st.slider(
    "Tech Comfort Score",
    1,
    10,
    5
)

st.subheader("Customer Activity")

total_sessions = st.number_input(
    "Total Sessions",
    min_value=0,
    value=100
)

gross_session_length = st.number_input(
    "Gross Session Length",
    min_value=0,
    value=2000
)

active_days = st.number_input(
    "Active Days",
    min_value=1,
    value=30
)

active_quarters = st.number_input(
    "Active Quarters",
    min_value=1,
    max_value=4,
    value=4
)

active_months = st.number_input(
    "Active Months",
    min_value=1,
    max_value=12,
    value=12
)

active_weeks = st.number_input(
    "Active Weeks",
    min_value=0,
    value=40
)

sessions_q1 = st.number_input(
    "Sessions Q1",
    min_value=0,
    value=25
)

sessions_q2 = st.number_input(
    "Sessions Q2",
    min_value=0,
    value=25
)

sessions_q3 = st.number_input(
    "Sessions Q3",
    min_value=0,
    value=25
)

sessions_q4 = st.number_input(
    "Sessions Q4",
    min_value=0,
    value=25
)

days_since_last_activity = st.number_input(
    "Days Since Last Activity",
    min_value=1,
    value=30
)

# ── Prediction ───────────────────────────────────────────────────────────────

if st.button("Predict"):

    active_days = max(active_days, 1)
    active_quarters = max(active_quarters, 1)
    active_months = max(active_months, 1)

    avg_sessions_per_active_day = total_sessions / active_days
    avg_minutes_per_active_day = gross_session_length / active_days

    avg_session_length = (
        gross_session_length / total_sessions
        if total_sessions > 0 else 0
    )

    avg_sessions_per_active_quarter = (
        total_sessions / active_quarters
    )

    avg_sessions_per_active_month = (
        total_sessions / active_months
    )

    active_day_ratio = active_days / 365
    active_quarter_ratio = active_quarters / 4
    active_month_ratio = active_months / 12

    session_growth = sessions_q4 - sessions_q1

    session_growth_ratio = (
        sessions_q4 / sessions_q1
        if sessions_q1 > 0 else 0
    )

    recency_weighted_sessions = (
        total_sessions / days_since_last_activity
        if days_since_last_activity > 0 else 0
    )

    income_score_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }

    education_score_map = {
        "High School": 1,
        "Other": 2,
        "Graduate": 3,
        "Post-Graduate": 4
    }

    income_score = income_score_map[income_level]
    education_score = education_score_map[education]
    affluence_score = income_score + education_score

    if age < 35 and income_level in ["High", "Very High"]:
        age_income_segment = "young_high_income"
    elif age < 35:
        age_income_segment = "young_low_income"
    elif income_level in ["High", "Very High"]:
        age_income_segment = "older_high_income"
    else:
        age_income_segment = "older_low_income"

    if age >= 50 and tech_comfort_score >= 8:
        age_tech_segment = "older_tech_savvy"
    elif age >= 50:
        age_tech_segment = "older_low_tech"
    elif tech_comfort_score >= 8:
        age_tech_segment = "younger_tech_savvy"
    else:
        age_tech_segment = "younger_low_tech"

    income_device_combo = f"{income_level}_{device_type}"

    premium_digital_customer = int(
        income_level in ["High", "Very High"]
        and tech_comfort_score >= 8
        and active_months >= 9
    )

    encoded_base = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type,
        "AGE_INCOME_SEGMENT": age_income_segment,
        "AGE_TECH_SEGMENT": age_tech_segment,
        "INCOME_DEVICE_COMBO": income_device_combo
    }])

    encoded = encoder.transform(encoded_base)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )

    feature_row = pd.DataFrame([{
        'TOTAL_NUM_SESSIONS': total_sessions,
        'GROSS_TOTAL_SESSION_LENGTH': gross_session_length,
        'ACTIVE_DAYS': active_days,
        'ACTIVE_QUARTERS': active_quarters,
        'ACTIVE_MONTHS': active_months,
        'ACTIVE_WEEKS': active_weeks,
        'SESSIONS_Q1': sessions_q1,
        'SESSIONS_Q2': sessions_q2,
        'SESSIONS_Q3': sessions_q3,
        'SESSIONS_Q4': sessions_q4,
        'AVG_SESSIONS_PER_ACTIVE_DAY': avg_sessions_per_active_day,
        'AVG_MINUTES_PER_ACTIVE_DAY': avg_minutes_per_active_day,
        'AVG_SESSION_LENGTH': avg_session_length,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
        'AVG_SESSIONS_PER_ACTIVE_MONTH': avg_sessions_per_active_month,
        'ACTIVE_DAY_RATIO': active_day_ratio,
        'ACTIVE_QUARTER_RATIO': active_quarter_ratio,
        'ACTIVE_MONTH_RATIO': active_month_ratio,
        'DAYS_SINCE_LAST_ACTIVITY': days_since_last_activity,
        'SESSION_GROWTH': session_growth,
        'SESSION_GROWTH_RATIO': session_growth_ratio,
        'RECENCY_WEIGHTED_SESSIONS': recency_weighted_sessions,
        'AGE': age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
        'INCOME_SCORE': income_score,
        'EDUCATION_SCORE': education_score,
        'AFFLUENCE_SCORE': affluence_score,
        'PREMIUM_DIGITAL_CUSTOMER': premium_digital_customer
    }])

    final_df = pd.concat(
        [feature_row.reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    for col in TRAINING_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df = final_df[TRAINING_COLUMNS]

    probability = model.predict_proba(final_df)[0][1]
    churn_probability = 1 - probability

    st.metric(
        "Renewal Probability",
        f"{probability:.1%}"
    )

    st.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )

    if churn_probability >= 0.60:
        st.error("High Churn Risk")
    elif churn_probability >= 0.40:
        st.warning("Medium Churn Risk")
    else:
        st.success("Low Churn Risk")
