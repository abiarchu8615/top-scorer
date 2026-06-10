import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import plotly.express as px
import streamlit as st
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# =====================================================
# LOAD ERP DATA
# =====================================================

st.set_page_config(
    page_title="AI ERP Command Center",
    layout="wide"
)

st.title("🚀 AI ERP Command Center")

uploaded_file = st.file_uploader(
    "Upload ERP Excel File",
    type=["xls", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload your ERP Excel file.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

st.success("ERP file loaded successfully!")

# ==========================
# DEPARTMENT FILTER
# ==========================

department_cols = [
    col for col in df.columns
    if "department" in col.lower()
]

if len(department_cols) > 0:

    DEPT_COL = department_cols[0]

    department_list = sorted(
        df[DEPT_COL]
        .dropna()
        .unique()
        .tolist()
    )

    selected_department = st.sidebar.selectbox(
        "Select Department",
        ["ALL"] + department_list
    )

    if selected_department != "ALL":

        df = df[
            df[DEPT_COL]
            == selected_department
        ]

else:

    selected_department = "ALL"
    
# =====================================================
# MENU FILTER
# =====================================================

menu_list = sorted(
    df["Menu Option Name"]
    .dropna()
    .unique()
    .tolist()
)

selected_menu = st.sidebar.selectbox(
    "Select Menu Option",
    ["ALL"] + menu_list
)

if selected_menu != "ALL":

    df = df[
        df["Menu Option Name"]
        == selected_menu
    ]

st.sidebar.write(
    f"Records: {len(df)}"
)

# ==========================
# PROCESS FILTER
# ==========================

process_list = sorted(
    df["Process Name"]
    .dropna()
    .unique()
    .tolist()
)

selected_process = st.sidebar.selectbox(
    "Select Process",
    ["ALL"] + process_list
)

if selected_process != "ALL":
    df = df[df["Process Name"] == selected_process]



    
# ==========================
# USER FILTER
# ==========================

user_list = sorted(
    df["User Name"]
    .dropna()
    .unique()
    .tolist()
)

selected_user = st.sidebar.selectbox(
    "Select User",
    ["ALL"] + user_list
)

if selected_user != "ALL":
    df = df[df["User Name"] == selected_user]
    
    st.sidebar.markdown("---")

   
# =====================================================
# CONVERT ERP DURATION TO MINUTES
# =====================================================

def convert_duration(value):

    try:

        value = str(value).replace("\n", "").strip()

        parts = value.split(":")

        if len(parts) == 2:

            hh = int(parts[0].strip())
            mm = int(parts[1].strip())

            return hh * 60 + mm

        return 0

    except:
        return 0

duration_column = "Duration\n(HH : MI)"

df["Duration_Minutes"] = (
    df[duration_column]
    .astype(str)
    .apply(convert_duration)
)

# =====================================================
# SHOW COLUMN NAMES
# =====================================================

st.subheader("Detected ERP Columns")

st.write(df.columns.tolist())

department_cols = [
    col for col in df.columns
    if "department" in col.lower()
]

st.write("Detected Department Columns:", department_cols)

# =====================================================
# AUTO DETECT USER COLUMN
# =====================================================

possible_user_cols = [
    col for col in df.columns
    if "user" in col.lower()
]

if len(possible_user_cols) == 0:
    st.error(
        "Cannot find User column. Please rename manually."
    )
    st.stop()

USER_COL = "User Name"

# =====================================================
# AUTO DETECT DURATION
# =====================================================

DURATION_COL = "Duration_Minutes"

# =====================================================
# MENU OPTION
# =====================================================

menu_cols = [
    col for col in df.columns
    if "menu" in col.lower()
]

if len(menu_cols) > 0:

    MENU_COL = "Menu Option Name"

else:

    MENU_COL = USER_COL

# =====================================================
# PROCESS COLUMN
# =====================================================

process_cols = [
    col for col in df.columns
    if "process" in col.lower()
]

if len(process_cols) > 0:

   PROCESS_COL = "Process Name"

else:

    PROCESS_COL = MENU_COL

# =====================================================
# FEATURE ENGINEERING
# =====================================================

user_features = df.groupby(USER_COL).agg({

    MENU_COL: "nunique",
    PROCESS_COL: "nunique",
    DURATION_COL: "sum"

}).reset_index()

user_features.columns = [

    "User",
    "Unique_Menus",
    "Unique_Processes",
    "Total_Duration"

]

# =====================================================
# EFFICIENCY SCORE
# =====================================================

user_features["Efficiency_Raw"] = (

    user_features["Unique_Menus"] +
    user_features["Unique_Processes"]

) / (

    user_features["Total_Duration"] + 1

)

eff_scaler = MinMaxScaler()

user_features["Efficiency_Score"] = (

    eff_scaler.fit_transform(

        user_features[
            ["Efficiency_Raw"]
        ]

    ) * 100

)

def efficiency_level(score):

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Average"

    else:
        return "Low"

user_features["Efficiency_Level"] = (

    user_features["Efficiency_Score"]
    .apply(efficiency_level)

)


# =====================================================
# PRODUCTIVITY SCORE
# =====================================================

scaler = MinMaxScaler()

st.subheader("Debug Data")

st.write(user_features.head())

st.write(user_features.dtypes)

scaled = scaler.fit_transform(

    user_features[
        [
            "Unique_Menus",
            "Unique_Processes",
            "Total_Duration"
        ]
    ]

)

user_features["Menus_Scaled"] = scaled[:,0]
user_features["Processes_Scaled"] = scaled[:,1]
user_features["Duration_Scaled"] = scaled[:,2]

user_features["Productivity_Score"] = (

    0.4 * user_features["Menus_Scaled"]
    + 0.3 * user_features["Processes_Scaled"]
    + 0.3 * user_features["Duration_Scaled"]

) * 100

# =====================================
# DEPARTMENT PRODUCTIVITY
# =====================================

department_productivity = None

if len(department_cols) > 0:

    DEPT_COL = department_cols[0]

    department_productivity = (

        df.groupby(DEPT_COL)
        .agg({

            "User Name": "nunique",
            "Duration_Minutes": "sum",
            "Menu Option Name": "nunique",
            "Process Name": "nunique"

        })
        .reset_index()

    )

    department_productivity["Productivity"] = (

        department_productivity["Menu Option Name"] * 0.4 +

        department_productivity["Process Name"] * 0.3 +

        department_productivity["Duration_Minutes"] * 0.3

    )

# =====================================================
# RISK SCORE
# =====================================================

user_features["Risk_Score"] = (

    (1 - user_features["Menus_Scaled"]) * 30 +

    (1 - user_features["Processes_Scaled"]) * 20 +

    (1 - user_features["Duration_Scaled"]) * 25 +

    (100 - user_features["Productivity_Score"]) * 0.25

)

user_features["Risk_Score"] = (
    user_features["Risk_Score"]
    .clip(0,100)
)

def risk_level(score):

    if score >= 70:
        return "High Risk"

    elif score >= 40:
        return "Medium Risk"

    else:
        return "Low Risk"

user_features["Risk_Level"] = (
    user_features["Risk_Score"]
    .apply(risk_level)
)

user_features["Executive_Score"] = (

    user_features["Productivity_Score"] * 0.7 +

    (100 - user_features["Risk_Score"]) * 0.3

)
# =====================================================
# PERFORMANCE LABEL
# =====================================================

def performance(score):

    if score >= 80:
        return "High"

    elif score >= 50:
        return "Medium"

    else:
        return "Low"

user_features["Performance"] = (
    user_features["Productivity_Score"]
    .apply(performance)
)

# =====================================================
# MACHINE LEARNING
# =====================================================

X = user_features[
    [
        "Unique_Menus",
        "Unique_Processes",
        "Total_Duration"
    ]
]

# =====================================
# USER CLUSTERING
# =====================================

from sklearn.cluster import KMeans

n_clusters = min(
    4,
    len(user_features)
)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

user_features["Cluster"] = (
    kmeans.fit_predict(X)
)

cluster_names = {
    0: "Power Users",
    1: "Normal Users",
    2: "Casual Users",
    3: "Special Cases"
}

user_features["Cluster_Name"] = (
    user_features["Cluster"]
    .map(cluster_names)
)

y = user_features["Performance"]

# =====================================================
# ANOMALY DETECTION
# =====================================================

iso = IsolationForest(

    contamination=0.05,
    random_state=42

)

user_features["Anomaly"] = (
    iso.fit_predict(X)
)

# =====================================
# LOF DETECTION
# =====================================

lof = LocalOutlierFactor(
    n_neighbors=5,
    contamination=0.05
)

user_features["LOF_Anomaly"] = (
    lof.fit_predict(X)
)

user_features.loc[
    user_features["LOF_Anomaly"] == -1,
    "Risk_Score"
] += 15

user_features["Risk_Score"] = (
    user_features["Risk_Score"]
    .clip(0,100)
)
user_features.loc[
    user_features["Anomaly"] == -1,
    "Risk_Score"
] += 20

user_features["Risk_Score"] = (
    user_features["Risk_Score"]
    .clip(0,100)
)


# =====================================
# ERP HEALTH SCORE
# =====================================

erp_health = (

    user_features["Productivity_Score"].mean() * 0.4 +

    user_features["Efficiency_Score"].mean() * 0.3 +

    (100 -
     user_features["Risk_Score"].mean()) * 0.3

)
# =====================================================
# EXECUTIVE INSIGHTS
# =====================================================

top_user = user_features.loc[
    user_features["Productivity_Score"].idxmax(),
    "User"
]

risk_user = user_features.loc[
    user_features["Risk_Score"].idxmax(),
    "User"
]

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🤖 ERP AI Assistant"
)

question = st.sidebar.text_input(
    "Ask ERP AI"
)

if question:

    q = question.lower()

    if "highest risk" in q:

        st.sidebar.success(
            f"Highest Risk: {risk_user}"
        )

    elif "top performer" in q:

        st.sidebar.success(
            f"Top Performer: {top_user}"
        )

    elif "anomaly" in q:

        anomaly_count = len(
            user_features[
                user_features["Anomaly"] == -1
            ]
        )

        st.sidebar.success(
            f"Anomalies: {anomaly_count}"
        )

    else:

        st.sidebar.info(
            "Try: highest risk, top performer, anomaly"
        )
        
# =====================================================
# KPI SECTION
# =====================================================

col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8)

col1.metric("Users", len(user_features))

col2.metric(
    "Avg Productivity",
    round(
        user_features["Productivity_Score"].mean(),
        2
    )
)

col3.metric(
    "Top Performer",
    top_user
)

col4.metric(
    "Highest Risk",
    risk_user
)

col5.metric(
    "Top Score",
    round(
        user_features["Productivity_Score"].max(),
        2
    )
)

col6.metric(
    "Anomalies",
    len(
        user_features[
            user_features["Anomaly"] == -1
        ]
    )
)

col7.metric(
    "Avg Risk",
    round(
        user_features["Risk_Score"].mean(),
        2
    )
)

col8.metric(
    "Avg Efficiency",
    round(
        user_features["Efficiency_Score"].mean(),
        2
    )
)

# =====================================================
# AI INSIGHTS
# =====================================================

st.subheader("🧠 AI Insights")

risk_user = user_features.loc[
    user_features["Risk_Score"].idxmax(),
    "User"
]

top_user = user_features.loc[
    user_features["Productivity_Score"].idxmax(),
    "User"
]

st.success(
    f"🏆 Top performer: {top_user}"
)

if user_features["Risk_Score"].max() > 80:

    st.warning(
        f"⚠ User {risk_user} shows unusually high risk behavior."
    )

if user_features["Productivity_Score"].mean() < 50:

    st.error(
        "📉 Overall productivity is below target."
    )

if user_features["Efficiency_Score"].mean() > 70:

    st.success(
        "⚡ ERP users demonstrate strong efficiency."
    )

anomaly_count = len(
    user_features[
        user_features["Anomaly"] == -1
    ]
)

if anomaly_count > 0:

    st.warning(
        f"🔍 AI detected {anomaly_count} suspicious users."
    )
# =====================================
# DEPARTMENT COMPARISON
# =====================================

if department_productivity is not None:
  st.subheader("🏢 Department Productivity Comparison")
  fig_dept = px.bar(

    department_productivity,

    x=DEPT_COL,

    y="Productivity",

    title="Department Productivity",

    text_auto=True

)

st.plotly_chart(
    fig_dept,
    use_container_width=True
)

st.subheader("🏆 Department Ranking")

dept_rank = (

    department_productivity
    .sort_values(
        "Productivity",
        ascending=False
    )

)

st.dataframe(
    dept_rank,
    use_container_width=True
)

col1,col2,col3,col4 = st.columns(4)

top_dept = dept_rank.iloc[0][DEPT_COL]

col1.metric(
    "Best Department",
    top_dept
)

col2.metric(
    "Departments",
    len(dept_rank)
)

col3.metric(
    "Avg Productivity",
    round(
        dept_rank["Productivity"].mean(),
        2
    )
)

col4.metric(
    "Top Score",
    round(
        dept_rank["Productivity"].max(),
        2
    )
)

st.subheader("📊 Most Used ERP Menus")

menu_usage = (

    df.groupby(
        "Menu Option Name"
    )
    .size()
    .reset_index(name="Count")
    .sort_values(
        "Count",
        ascending=False
    )
    .head(15)

)

fig_menu = px.bar(

    menu_usage,

    x="Menu Option Name",

    y="Count",

    title="Top ERP Menu Usage"

)

st.plotly_chart(
    fig_menu,
    use_container_width=True
)


# =====================================================
# TOP USERS
# =====================================================

st.subheader("🏆 Top Employees")

top10 = (
    user_features
    .sort_values(
        "Productivity_Score",
        ascending=False
    )
    .head(10)
)

fig = px.bar(

    top10,

    x="User",
    y="Productivity_Score",

    title="Top Productivity"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("🚨 Highest Risk Users")

risk_top = (
    user_features
    .sort_values(
        "Risk_Score",
        ascending=False
    )
    .head(10)
)

fig_risk = px.bar(
    risk_top,
    x="User",
    y="Risk_Score",
    color="Risk_Level",
    title="Highest Risk Users"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)
st.subheader("⚡ Most Efficient Users")

efficient_users = (

    user_features
    .sort_values(
        "Efficiency_Score",
        ascending=False
    )
    .head(10)

)

fig_eff = px.bar(

    efficient_users,

    x="User",
    y="Efficiency_Score",

    color="Efficiency_Level",

    title="Most Efficient ERP Users"

)

st.plotly_chart(
    fig_eff,
    use_container_width=True
)

st.subheader("🚨 High Risk Menu Usage")

menu_risk = (

    df.groupby(
        "Menu Option Name"
    )
    .agg({

        "Duration_Minutes":"sum",

        "User Name":"nunique"

    })

    .reset_index()

)

fig_risk_menu = px.treemap(

    menu_risk,

    path=["Menu Option Name"],

    values="Duration_Minutes",

    color="User Name"

)

st.plotly_chart(
    fig_risk_menu,
    use_container_width=True
)


# =====================================================
# PERFORMANCE DISTRIBUTION
# =====================================================

st.subheader("📊 Performance Levels")

fig2 = px.pie(

    user_features,

    names="Performance",

    title="Employee Performance"

)

st.plotly_chart(
    fig2,
    use_container_width=True
)


import plotly.graph_objects as go

st.subheader("🏥 ERP Health Score")

fig_gauge = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=erp_health,

        title={"text":"ERP Health"},

        gauge={
            "axis":{
                "range":[0,100]
            }
        }

    )

)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

# =====================================================
# ANOMALY USERS
# =====================================================

st.subheader(
    "⚠️ Suspicious Users"
)

anomalies = (
    user_features[
        user_features["Anomaly"]==-1
    ]
)

st.dataframe(anomalies)

# =====================================================
# ALL USERS
# =====================================================
st.subheader("🥇 ERP Leaderboard")

leaderboard = (

    user_features
    .sort_values(
        "Productivity_Score",
        ascending=False
    )
    .head(3)

)

medals = ["🥇", "🥈", "🥉"]

for i, row in enumerate(
    leaderboard.itertuples()
):

    st.write(
        f"{medals[i]} "
        f"{row.User} "
        f"({row.Productivity_Score:.2f})"
    )
    
st.subheader("📋 User Ranking")

ranking = (
    user_features
    .sort_values(
        "Productivity_Score",
        ascending=False
    )
)

st.dataframe(ranking)
st.subheader("🛡 Risk Assessment")

st.dataframe(

    user_features[
        [
            "User",
            "Productivity_Score",
            "Risk_Score",
            "Risk_Level",
            "Executive_Score",
            "Anomaly"
        ]
    ]
    .sort_values(
        "Risk_Score",
        ascending=False
    )

)
# =====================================================
# EXPORT RESULTS
# =====================================================

from datetime import datetime

filename = (
    f"ERP_User_Ranking_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

from io import BytesIO

output = BytesIO()

with pd.ExcelWriter(
    output,
    engine="openpyxl"
) as writer:

    ranking.to_excel(
        writer,
        index=False
    )

st.download_button(
    "📥 Download Ranking",
    data=output.getvalue(),
    file_name="ERP_User_Ranking.xlsx"
)

st.success(f"Exported: {filename}")