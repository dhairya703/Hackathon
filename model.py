
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ==============================
# Flight AI Analyzer
# ==============================
class FlightAIAnalyzer:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.delay_model: Optional[RandomForestRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.features: List[str] = []

    # -----------------------------
    # Load & preprocess
    # -----------------------------
    def load_and_preprocess_data(self, file_path: str) -> bool:
        try:
            self.data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: file not found: {file_path}")
            return False

        df = self.data
        df.columns = df.columns.str.strip()

        # Convert time/date
        df["Sched_Departure_Time_dt"] = pd.to_datetime(
            df["Sched_Departure_Time"].str.replace(" IST",""), format="%H:%M", errors="coerce"
        )
        df["Hour_Slot"] = df["Sched_Departure_Time_dt"].dt.hour
        df["Day_of_Week"] = pd.to_datetime(
            df["Sched_Departure_Date"].astype(str) + f" {datetime.now().year}",
            format="%d %b %Y", errors="coerce"
        ).dt.dayofweek

        df["Departure_Delay_mins"] = pd.to_numeric(df.get("Departure_Delay_mins", 0), errors="coerce").fillna(0)
        df.dropna(subset=["Hour_Slot", "Day_of_Week"], inplace=True)
        df["Hour_Slot"] = df["Hour_Slot"].astype(int)
        df["Day_of_Week"] = df["Day_of_Week"].astype(int)

        # Feature engineering
        peak_hours = {6,7,8,17,18,19,20,21}
        df["Is_Peak_Hour"] = df["Hour_Slot"].apply(lambda x: 1 if x in peak_hours else 0)
        df["Is_Weekend"] = df["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)

        # Encode categorical features
        for col in ["Airline", "Direction", "Origin", "Destination"]:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))
                self.label_encoders[col] = le

        # Aggregated features
        df["Avg_Hourly_Delay"] = df.groupby("Hour_Slot")["Departure_Delay_mins"].transform("mean")
        df["Avg_Airline_Delay"] = df.groupby("Airline")["Departure_Delay_mins"].transform("mean")
        df["Avg_Destination_Delay"] = df.groupby("Destination")["Departure_Delay_mins"].transform("mean")

        self.data = df
        return True

    # -----------------------------
    # Analytics
    # -----------------------------
    def identify_busiest_slots(self, top_n: int = 5) -> pd.Series:
        return self.data.groupby("Hour_Slot")["Flight_Number"].count().sort_values(ascending=False).head(top_n)

    def peak_time_delay_analysis(self) -> pd.DataFrame:
        df = self.data
        return df.groupby("Hour_Slot").agg(
            Flight_Count=("Flight_Number","count"),
            Avg_Dep_Delay=("Departure_Delay_mins","mean"),
            Max_Dep_Delay=("Departure_Delay_mins","max")
        ).sort_values("Flight_Count", ascending=False)

    def detect_cascading_disruptions(self) -> pd.DataFrame:
        df = self.data.copy()
        result = []
        for airline in df["Airline"].dropna().unique():
            sub = df[df["Airline"]==airline].sort_values("Sched_Departure_Time_dt")
            cascades = 0
            prev_delay = None
            for d in sub["Departure_Delay_mins"]:
                if prev_delay is not None and d > prev_delay and d > 15:
                    cascades += 1
                prev_delay = d
            total_flights = len(sub)
            result.append({
                "Airline": airline,
                "Total_Flights": total_flights,
                "Cascade_Events": cascades,
                "Cascade_Rate": cascades/total_flights if total_flights else 0
            })
        return pd.DataFrame(result).sort_values("Cascade_Rate", ascending=False)

    # -----------------------------
    # Delay Prediction
    # -----------------------------
    def build_delay_prediction_model(self) -> float:
        self.features = [
            "Hour_Slot", "Day_of_Week", "Is_Peak_Hour", "Is_Weekend",
            "Airline_encoded", "Direction_encoded", "Origin_encoded", "Destination_encoded",
            "Avg_Hourly_Delay", "Avg_Airline_Delay", "Avg_Destination_Delay"
        ]
        avail = [f for f in self.features if f in self.data.columns]
        X = self.data[avail]
        y = self.data["Departure_Delay_mins"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.delay_model = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=15
        )
        self.delay_model.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, self.delay_model.predict(X_te))
        return mae

    def optimal_hours(self, k: int = 5) -> pd.DataFrame:
        if self.delay_model is None:
            raise RuntimeError("Model not trained.")
        avail = [f for f in self.features if f in self.data.columns]
        reps = self.data.groupby("Hour_Slot")[avail].mean().reset_index()
        reps["predicted_delay"] = self.delay_model.predict(reps[avail])
        return reps.sort_values("predicted_delay").head(k)[["Hour_Slot","predicted_delay"]]

    def high_impact_flights(self, top_n: int = 10) -> pd.DataFrame:
        df = self.data.copy()
        def impact(row):
            delay_score = row["Departure_Delay_mins"]
            peak_mult = 1.5 if row["Is_Peak_Hour"] else 1.0
            airline_factor = row["Avg_Airline_Delay"] / (df["Avg_Airline_Delay"].max() + 1)
            dest_factor = row["Avg_Destination_Delay"] / (df["Avg_Destination_Delay"].max() + 1)
            return (delay_score * peak_mult) + 10 * airline_factor + 10 * dest_factor
        df["Impact_Score"] = df.apply(impact, axis=1)
        cols = ["Flight_Number","Airline","Departure_Delay_mins","Hour_Slot","Impact_Score"]
        existing = [c for c in cols if c in df.columns]
        return df.sort_values("Impact_Score", ascending=False).head(top_n)[existing]

    # -----------------------------
    # Simulation & Recommendation
    # -----------------------------
    def simulate_and_recommend_time(self, flight_number: str) -> Optional[Dict[str, Any]]:
        if self.delay_model is None:
            raise RuntimeError("Model not trained.")
        df = self.data
        sub = df[df["Flight_Number"]==flight_number]
        if sub.empty:
            combined = df["Flight_Number"] + " " + df["Airline"]
            mask = combined.str.contains(flight_number)
            sub = df[mask]
        if sub.empty:
            return None

        orig = sub.iloc[0].copy()
        sim = pd.DataFrame([orig.to_dict()]*24)
        sim["Hour_Slot"] = range(24)
        peak_hours = {6,7,8,17,18,19,20,21}
        sim["Is_Peak_Hour"] = sim["Hour_Slot"].apply(lambda x: 1 if x in peak_hours else 0)
        sim["Is_Weekend"] = orig["Is_Weekend"]

        avail = [f for f in self.features if f in sim.columns]
        sim["predicted_delay"] = self.delay_model.predict(sim[avail])
        sim["improvement"] = orig["Departure_Delay_mins"] - sim["predicted_delay"]

        # Choose hour with max improvement
        best = sim.loc[sim["improvement"].idxmax()]
        recommended_hour = best["Hour_Slot"] if best["improvement"] > 0 else orig["Hour_Slot"]

        return {
            "flight": flight_number,
            "current_hour": int(orig["Hour_Slot"]),
            "current_delay": float(orig["Departure_Delay_mins"]),
            "recommended_hour": int(recommended_hour),
            "predicted_delay_at_recommendation": float(best["predicted_delay"]),
            "minutes_saved": float(max(0.0, best["improvement"]))
        }

class NLPScheduleAssistant:
    def __init__(self, analyzer: FlightAIAnalyzer):
        self.analyzer = analyzer

    def parse(self, text: str) -> Tuple[str, Dict[str, Any]]:
        t = text.strip().lower()
        # Help
        if "help" in t or "commands" in t:
            return "help", {}
        if any(k in t for k in ["train model","build model"]):
            return "train_model", {}
        if "busiest" in t or "peak slots" in t:
            top_n = self._extract_topn(t) or 5
            return "busiest_slots", {"top_n": top_n}
        if "peak delay" in t or "peak time" in t:
            return "peak_delay", {}
        if "cascading" in t or "cascade" in t or "disruption" in t:
            return "cascading_disruptions", {}
        if "optimal" in t or "lowest delay" in t:
            k = self._extract_topn(t) or 5
            return "optimal_hours", {"k": k}
        if "most delayed" in t:
            top_n = self._extract_topn(t) or 10
            return "most_delayed", {"top_n": top_n}
        if "least delayed" in t:
            top_n = self._extract_topn(t) or 10
            return "least_delayed", {"top_n": top_n}
        if "high impact" in t or "disruption" in t:
            top_n = self._extract_topn(t) or 10
            return "high_impact", {"top_n": top_n}
        # Flight simulation
        fn = self._extract_flight_number(text)
        if fn:
            return "simulate", {"flight_number": fn}
        return "help", {}

    def _extract_topn(self, t: str) -> Optional[int]:
        m = re.search(r"\btop\s*(\d{1,2})\b", t)
        if m: return int(m.group(1))
        return None

    def _extract_flight_number(self, text: str) -> Optional[str]:
        df = self.analyzer.data
        if df is None or df.empty:
            return None

        # Match typical flight number patterns
        candidates = re.findall(r"\b[A-Za-z]{1,3}\d{2,4}[A-Za-z]?\b", text)
        if not candidates:
            return None

        candidates = [c.strip().upper() for c in candidates]

        # Try exact match (case-insensitive)
        flight_numbers = df["Flight_Number"].dropna().astype(str).str.upper()
        for c in candidates:
            if c in flight_numbers.values:
                return flight_numbers[flight_numbers == c].iloc[0]

        # Try partial match (for combined numbers like "6E2162 IGO209W")
        for c in candidates:
            matches = [f for f in flight_numbers if c in f]
            if len(matches) == 1:
                return matches[0]  # unique partial match

        # Fallback: return first candidate even if not found
        return candidates[0]


    def handle(self, text: str) -> str:
        intent, params = self.parse(text)
        try:
            if intent == "help":
                return ("I can help with:\n"
                        "• “train model” – build delay predictor\n"
                        "• “show busiest slots (top 5)” – busiest hours by movements\n"
                        "• “optimal hours (top 5)” – lowest predicted delay hours\n"
                        "• “high impact flights (top 10)” – likely disruption sources\n"
                        "• “simulate flight <Flight_Number>” – reschedule suggestion\n")
            if intent == "train_model":
                mae = self.analyzer.build_delay_prediction_model()
                return f"Model trained. Mean Absolute Error: {mae:.2f} minutes."
            if intent == "busiest_slots":
                counts = self.analyzer.identify_busiest_slots(params.get("top_n",5))
                return "Busiest hours (flight count):\n" + counts.to_string()
            if intent == "optimal_hours":
                df = self.analyzer.optimal_hours(params.get("k",5))
                return "Lowest predicted-delay hours:\n" + df.round(2).to_string(index=False)
            if intent == "most_delayed":
                top_n = params.get("top_n", 10)
                df_top = self.analyzer.data.sort_values("Departure_Delay_mins", ascending=False).head(top_n)
                return f"Most Delayed Flights (Top {top_n}):\n" + df_top[["Flight_Number","Airline","Departure_Delay_mins"]].to_string(index=False)

            if intent == "least_delayed":
                top_n = params.get("top_n", 10)
                df_top = self.analyzer.data.sort_values("Departure_Delay_mins", ascending=True).head(top_n)
                return f"Least Delayed Flights (Top {top_n}):\n" + df_top[["Flight_Number","Airline","Departure_Delay_mins"]].to_string(index=False)

            if intent == "high_impact":
                df = self.analyzer.high_impact_flights(params.get("top_n",10))
                return "High-impact flights:\n" + df.round(2).to_string(index=False)
            if intent == "peak_delay":
                df = self.analyzer.peak_time_delay_analysis()
                return "Peak-time delay analysis:\n" + df.round(2).to_string()
            if intent == "cascading_disruptions":
                df = self.analyzer.detect_cascading_disruptions()
                return "Cascading disruption analysis:\n" + df.round(2).to_string(index=False)
            if intent == "simulate":
                fn = params.get("flight_number")
                res = self.analyzer.simulate_and_recommend_time(fn)
                if res is None:
                    return f"Flight not found: {fn}"
                if res["minutes_saved"] > 0:
                    return (f"Reschedule recommendation for {res['flight']}:\n"
                            f"- Current hour: {res['current_hour']}, delay: {res['current_delay']:.0f}m\n"
                            f"- Recommended hour: {res['recommended_hour']} "
                            f"(predicted delay {res['predicted_delay_at_recommendation']:.1f}m)\n"
                            f"- Potential time saved: {res['minutes_saved']:.1f} minutes")
                else:
                    return (f"{res['flight']} is already near optimal.\n"
                            f"- Current hour: {res['current_hour']}, delay: {res['current_delay']:.0f}m\n"
                            f"- Best hour found: {res['recommended_hour']} "
                            f"(predicted {res['predicted_delay_at_recommendation']:.1f}m)")
        except Exception as e:
            return f"Error: {str(e)}"


# ==============================
# CLI
# ==============================
def main():
    csv_file = "flights_deduplicated_by_subset.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    analyzer = FlightAIAnalyzer()
    if not analyzer.load_and_preprocess_data(csv_file):
        sys.exit(1)
    assistant = NLPScheduleAssistant(analyzer)

    print("\n✈️ Flight NLP Assistant ready. Type 'help'. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if q.lower() in {"quit","exit","bye"}:
            print("Goodbye!")
            break
        print(assistant.handle(q))


if __name__ == "__main__":
    main()
