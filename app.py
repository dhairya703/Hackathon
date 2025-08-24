import streamlit as st
import pandas as pd
from model import FlightAIAnalyzer, NLPScheduleAssistant  # your classes

st.set_page_config(page_title="Flight Schedule AI", layout="wide")
st.title("✈️ Flight Schedule Optimization & NLP Assistant")

# -----------------------------
# Initialize session_state
# -----------------------------
if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# -----------------------------
# Sidebar: Upload CSV
# -----------------------------
st.sidebar.header("Upload Flight Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    if st.session_state.analyzer is None:
        analyzer = FlightAIAnalyzer()
        if analyzer.load_and_preprocess_data(uploaded_file):
            st.success("Data loaded successfully!")
            st.session_state.analyzer = analyzer
            st.session_state.assistant = NLPScheduleAssistant(analyzer)
        else:
            st.error("Failed to load data.")
            st.stop()
    else:
        analyzer = st.session_state.analyzer
        assistant = st.session_state.assistant

    # -----------------------------
    # Train Model Button
    # -----------------------------
    if st.button("Train Delay Prediction Model"):
        mae = analyzer.build_delay_prediction_model()
        st.session_state.model_trained = True
        st.success(f"Model trained. MAE: {mae:.2f} minutes.")

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Peak & Busiest Slots", "Cascading Disruptions",
        "High-Impact Flights", "NLP Assistant"
    ])

    # -----------------------------
    # Tab 1: Overview
    # -----------------------------
    with tab1:
        st.subheader("Dataset Overview")
        st.write(f"Total Flights: {len(analyzer.data)}")
        st.write(f"Airlines: {analyzer.data['Airline'].nunique()}")
        st.write(f"Destinations: {analyzer.data['Destination'].nunique()}")
        st.dataframe(analyzer.data.head(10))

    # -----------------------------
    # Tab 2: Peak & Busiest Slots
    # -----------------------------
    with tab2:
        st.subheader("Peak-Time Delay Analysis")
        df_peak = analyzer.peak_time_delay_analysis()
        st.dataframe(df_peak)
        st.bar_chart(df_peak[["Flight_Count","Avg_Dep_Delay"]])

        st.subheader("Busiest Slots")
        counts = analyzer.identify_busiest_slots()
        st.dataframe(counts)

    # -----------------------------
    # Tab 3: Cascading Disruptions
    # -----------------------------
    with tab3:
        st.subheader("Cascading Disruption Analysis")
        df_cascade = analyzer.detect_cascading_disruptions()
        st.dataframe(df_cascade)

    # -----------------------------
    # Tab 4: High-Impact Flights
    # -----------------------------
    with tab4:
        st.subheader("High-Impact Flights")
        top_n = st.slider("Select Top N Flights", 5, 50, 10)
        df_impact = analyzer.high_impact_flights(top_n=top_n)
        st.dataframe(df_impact)

    # -----------------------------
    # Tab 5: NLP Assistant
    # -----------------------------
# -----------------------------
# Tab 5: NLP Assistant
# -----------------------------
    with tab5:
        st.subheader("Ask about your flights (NLP)")

        # Display all available commands
        st.markdown(
            """
            **Available Commands:**
            - `train model` – Build and train the delay prediction model
            - `show busiest slots (top N)` – Show hours with the most flights
            - `peak delay` / `peak time` – Analyze peak-time delays
            - `cascading disruptions` – Detect cascading delays for airlines
            - `optimal hours (top N)` – Show hours with lowest predicted delays
            - `high impact flights (top N)` – Identify flights likely to cause disruptions
            - `most delayed` – Identify flights that are delayed the most
            - `least delayed` – Identify flights that are least delayed
            - `simulate flight <Flight_Number>` – Get reschedule suggestion for a flight
            """
        )

        query = st.text_input("Enter your question or command:")
        if st.button("Submit Query"):
            if query:
                if st.session_state.model_trained:
                    response = st.session_state.assistant.handle(query)
                    st.text(response)
                else:
                    st.warning("Please train the model first!")

