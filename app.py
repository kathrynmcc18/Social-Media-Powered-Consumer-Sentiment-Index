# app.py
# -*- coding: utf-8 -*-
"""
Social Media Powered Consumer Sentiment Index
Streamlit App by Kathryn McCarthy
"""

import streamlit as st
import pandas as pd
import os
import time
from mcsi_model import run_mcsi_model

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Social Media Powered Consumer Sentiment Index (SMPCSI)",
    page_icon="📊",
    layout="wide",
)
# ---------------------------
# Custom CSS for consistent dark mode styling
# ---------------------------
st.markdown(
    """
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #111827; /* dark navy */
        }

        /* Headings, labels, and text */
        .stMarkdown, .stTextInput label, .stSelectbox label,
        .stMultiSelect label, .stSlider label, .stNumberInput label {
            color: #FAFAFA !important;
        }

        /* Progress bar color */
        .stProgress > div > div > div > div {
            background-color: #ff7f0e;
        }

        /* Tables */
        .stDataFrame, .stTable {
            color: #E5E7EB !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("📊 Social Media Powered Consumer Sentiment Index")
st.markdown("""
This interactive app lets you upload social-media data, select economic topics,  
and generate your own **Social Media Powered Consumer Sentiment Index**.
""")

st.divider()


# ---------------------------
# Step 1: Upload Dataset
# ---------------------------
st.header("Step 1: Upload Your Social Media Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file (must include `snippet_text` and `company` columns):",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext == ".xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success(f"✅ File uploaded successfully! ({len(df):,} rows loaded)")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
else:
    st.info("Please upload your dataset to continue.")
    st.stop()

st.divider()


# ---------------------------
# Step 2: Select Topics
# ---------------------------
st.header("Step 2: Select Topics to Include in the Model")

if "company" not in df.columns:
    st.error("Column 'company' not found in the uploaded dataset.")
    st.stop()

topics = sorted(df["company"].dropna().unique())

st.markdown("""
Select one or more **economic topics** below.  
These determine which subset of your uploaded dataset is analyzed in the model.
""")

selected_topics = st.multiselect(
    "Choose topics:",
    topics,
    default=topics[:3]
)

if not selected_topics:
    st.warning("Please select at least one topic to continue.")
    st.stop()

# Filter dataset
filtered_df = df[df["company"].isin(selected_topics)].copy()
#filtered_df = filtered_df.sample(min(len(filtered_df), 100000), random_state=42)
st.success(f"✅ Filtered dataset includes {len(filtered_df):,} social media snippets across {len(selected_topics)} topic(s).")
st.dataframe(filtered_df.head(10), use_container_width=True)
st.caption("💡 This filtered dataset will be sent to the model for consumer sentiment analysis.")


# ---------------------------
# Optional: Limit Sample Size for Faster Runs
# ---------------------------
st.sidebar.markdown("### ⚙️ Model Run Settings")

# Only show if the dataset is large enough
max_rows = len(filtered_df)
if max_rows > 1000:
    sample_size = st.sidebar.slider(
        "Select sample size for testing (# of social media snippets to analyze):",
        min_value=500,
        max_value=max_rows,
        value=min(5000, max_rows),
        step=500,
        help="Reduce sample size for faster runs during testing."
    )
    filtered_df = filtered_df.sample(n=sample_size, random_state=42)
    st.info(f"⚙️ Using a random sample of **{sample_size:,}** rows out of {max_rows:,}.")
else:
    st.sidebar.write(f"✅ Full dataset ({max_rows:,} rows) will be used.")


st.divider()


# ---------------------------
# Step 3: Run Model (with smart progress bar + ETA)
# ---------------------------
st.header("Step 3: Run the Social Media Powered Consumer Sentiment Index Model")

st.markdown("""
Click below to run the **Social Media Powered Consumer Sentiment Index (SMPCSI)** model.  
This will:
- Load a sentiment model (HuggingFace CardiffNLP Twitter RoBERTa Sentiment Model or VADER)
- Score all posts
- Aggregate daily sentiment
- Fit the regression vs University of Michigan CSI
- Produce a comparison plot between University of Michigan CSI and Social Media Powered CSI
""")

if st.button("🚀 Run Social Media Sentiment Model"):
    n_rows = len(filtered_df)
    est_seconds = max(30, n_rows / 25)  # Rough estimate
    est_minutes = round(est_seconds / 60, 1)

    st.info(f"⏳ Estimated runtime: ~{est_minutes} minute{'s' if est_minutes != 1 else ''}. "
            "The model is scoring each post sequentially.")

    progress_bar = st.progress(0, text="Starting model run...")
    status_placeholder = st.empty()
    start_time = time.time()

    try:
        # --- Step 1: Save filtered dataset ---
        progress_bar.progress(5, text="Preparing data...")
        filtered_path = "./Filtered_Social_File.xlsx"
        filtered_df.to_excel(filtered_path, index=False)
        time.sleep(0.5)

        # --- Step 2: Initialize model ---
        progress_bar.progress(20, text="Loading sentiment model (this may take 1–2 minutes)...")
        status_placeholder.info("🔄 Initializing HuggingFace/VADER sentiment scorer...")
        time.sleep(1)

        # --- Step 3: Simulate scoring progress ---
        progress_bar.progress(40, text="Scoring posts and computing sentiment indices...")
        status_placeholder.info(f"🧠 Scoring {n_rows:,} posts... please wait.")

        est_update_interval = 5
        total_updates = int(est_seconds / est_update_interval)
        progress_value = 40

        for i in range(total_updates):
            time.sleep(est_update_interval)
            progress_value = min(40 + int((i / total_updates) * 40), 80)
            elapsed = time.time() - start_time
            remaining = max(0, est_seconds - elapsed)
            mins_rem = round(remaining / 60, 1)
            progress_bar.progress(
                progress_value,
                text=f"Scoring in progress... (~{mins_rem} min remaining)"
            )

        # --- Step 4: Run model ---
        artifacts = run_mcsi_model(filtered_path, selected_topics)
        progress_bar.progress(85, text="Aggregating results and fitting regression model...")
        status_placeholder.info("📈 Aggregating results and fitting regression model...")
        time.sleep(0.5)

        # --- Step 5: Display results ---
        progress_bar.progress(100, text="Done!")
        status_placeholder.empty()
        st.success("✅ SMPCSI Model completed successfully!")

        # ---------------------------
        # Display Results
        # ---------------------------
        #st.subheader("📊 Model Performance Metrics")
        #st.json(artifacts["metrics"])

        st.subheader("📅 Daily Social Sentiment Index")
        st.dataframe(artifacts["daily"].head(20), use_container_width=True)

        st.subheader("📈 Monthly Nowcast (Social Media Powered CSI vs. UMich CSI)")
        st.dataframe(artifacts["window_join"].head(20), use_container_width=True)


        plot_path = artifacts.get("plot_path")
        if plot_path and os.path.exists(plot_path):
            st.subheader("📈 Interactive Comparison Plot - MCSI vs. SMPCSI")
            with open(plot_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=600, scrolling=True)
        else:
            st.warning("⚠️ No plot found. The model may not have generated the output HTML.")

        total_time = round((time.time() - start_time) / 60, 2)
        st.caption(f"🕓 Model finished in {total_time} minutes.")

    except Exception as e:
        progress_bar.empty()
        status_placeholder.empty()
        st.error(f"❌ Error running the model: {e}")


