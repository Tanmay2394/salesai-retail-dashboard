import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="SalesAI - Retail Analytics",
    page_icon="📊",
    layout="centered"
)

st.title("📊 SalesAI – GenAI Powered Retail Analytics Assistant")
st.caption("Conversational Analytics Engine for ABC Retail Store")
st.markdown("---")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("fmcg_mult_prod_transactions_lifecycle_sample - Copy.xlsx")
    return df

df = load_data()

df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce")
df["transaction_month"] = pd.to_datetime(df["transaction_month"])

# ---------------------------
# SALES OVERVIEW
# ---------------------------
st.subheader("📈 Sales Overview")

total_revenue = df["sales_amount"].sum()
total_transactions = len(df)
unique_customers = df["customer_number"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"₹ {int(total_revenue):,}")
col2.metric("🧾 Transactions", total_transactions)
col3.metric("👥 Customers", unique_customers)

st.markdown("#### 📌 Quick Business Snapshot")
st.markdown("""
- Revenue performance overview  
- Customer lifecycle segmentation  
- Product contribution analysis  
- AI-powered strategic insights  
""")

st.markdown("---")

# ---------------------------
# RFM ANALYSIS
# ---------------------------
st.subheader("📊 Customer Lifecycle (RFM Analysis)")

latest_date = df["transaction_month"].max()

rfm = df.groupby("customer_number").agg({
    "transaction_month": lambda x: (latest_date - x.max()).days,
    "transaction_id": "count",
    "sales_amount": "sum"
}).reset_index()

rfm.columns = ["customer_number", "Recency", "Frequency", "Monetary"]

rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4]).astype(int)

rfm["RFM_Score"] = (
    rfm["R_score"].astype(str) +
    rfm["F_score"].astype(str) +
    rfm["M_score"].astype(str)
)

def segment_customer(row):
    if row["RFM_Score"] in ["444","443","434","344"]:
        return "Champions"
    elif row["F_score"] >= 3 and row["M_score"] >= 3:
        return "Loyal Customers"
    elif row["R_score"] <= 2 and row["F_score"] >= 3:
        return "At Risk"
    elif row["R_score"] == 4:
        return "New Customers"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)
segment_counts = rfm["Segment"].value_counts()

fig2, ax2 = plt.subplots()
segment_counts.plot(kind="bar", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

st.markdown("---")

# ---------------------------
# GEMINI SAFE FUNCTION
# ---------------------------
def get_gemini_response(prompt, context):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"""
You are a senior retail data analyst at ABC Retail.

STRICT RULES:
- Use ONLY the provided dataset context.
- Do NOT hallucinate.
- Be data-driven.
- Keep answers concise and professional.

FORMAT YOUR RESPONSE AS:

### 📊 Key Insight
(What does the data show?)

### 📈 Business Interpretation
(Why is this happening?)

### 🎯 Recommendation
(What should the business do?)

DATA CONTEXT:
{context}

USER QUESTION:
{prompt}
"""

        response = models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# SESSION STATE
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "report_data" not in st.session_state:
    st.session_state.report_data = []

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# ---------------------------
# CHAT INPUT
# ---------------------------
if prompt := st.chat_input("Ask a business question about the sales data..."):

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.history.append(("user", prompt))

    # ---------------------------
    # DYNAMIC VISUALS
    # ---------------------------
    if "top" in prompt.lower() and "product" in prompt.lower():
        top_products = df.groupby("product_id")["sales_amount"].sum().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots()
        top_products.plot(kind="bar", ax=ax)
        ax.set_title("Top 5 Products by Revenue")
        st.pyplot(fig)

    elif "trend" in prompt.lower() or "monthly" in prompt.lower():
        monthly_sales = df.groupby(df["transaction_month"].dt.to_period("M"))["sales_amount"].sum()
        fig, ax = plt.subplots()
        monthly_sales.plot(marker='o', linewidth=2)
        ax.set_title("Monthly Revenue Trend")
        ax.grid(True)
        st.pyplot(fig)

    # ---------------------------
    # DATA CONTEXT
    # ---------------------------
    product_summary = df.groupby("product_id")["sales_amount"].sum().sort_values(ascending=False).head()

    context = f"""
Columns: {list(df.columns)}

Total Revenue: {total_revenue}
Total Transactions: {total_transactions}
Unique Customers: {unique_customers}

Top Products:
{product_summary.to_string()}

Customer Segments:
{segment_counts.to_string()}
"""

    reply = get_gemini_response(prompt, context)

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    final_reply = f"""
{reply}

---
🕒 Timestamp: {timestamp}  
⏱ Processing Time: {processing_time} seconds
"""

    with st.chat_message("assistant"):
        st.markdown(final_reply)

    st.session_state.history.append(("assistant", final_reply))

    st.session_state.report_data.append({
        "timestamp": timestamp,
        "question": prompt,
        "answer": reply,
        "processing_time_seconds": processing_time
    })

# ---------------------------
# DOWNLOAD REPORT
# ---------------------------
if st.session_state.report_data:
    report_df = pd.DataFrame(st.session_state.report_data)
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Chat Report",
        data=csv,
        file_name="salesai_chat_report.csv",
        mime="text/csv"
    )
