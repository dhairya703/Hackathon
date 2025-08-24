import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, using only sklearn models")

# Optimization
from scipy.optimize import minimize
import pulp

# NLP for interface
import re
from collections import Counter

class FlightScheduleOptimizer:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess flight data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(file_path)
        
        # Clean column names
        self.data.columns = self.data.columns.str.strip()
        
        # Convert time columns and handle errors by coercing to NaT (Not a Time)
        self.data['Sched_Departure_Time_dt'] = pd.to_datetime(
            self.data['Sched_Departure_Time'], format='%H:%M IST', errors='coerce'
        )
        
        # Extract hour from the new datetime column
        self.data['Hour_Slot'] = self.data['Sched_Departure_Time_dt'].dt.hour
        
        # Handle missing values in key numeric and categorical columns
        self.data['Departure_Delay_mins'] = pd.to_numeric(self.data['Departure_Delay_mins'], errors='coerce').fillna(0)
        self.data['Arrival_Delay_mins'] = pd.to_numeric(self.data['Arrival_Delay_mins'], errors='coerce').fillna(0)
        self.data.dropna(subset=['Hour_Slot'], inplace=True) # Drop rows where time was invalid
        self.data['Hour_Slot'] = self.data['Hour_Slot'].astype(int) # Convert hour to integer
        
        # Create additional features
        self.data['Is_Peak_Hour'] = self.data['Hour_Slot'].apply(
            lambda x: 1 if x in [6, 7, 8, 17, 18, 19, 20, 21] else 0
        )
        
        # Correctly parse the date and extract the day of the week
        self.data['Day_of_Week'] = pd.to_datetime(self.data['Sched_Departure_Date'], format='%d %b', errors='coerce').dt.dayofweek
        
        # Encode categorical variables
        categorical_cols = ['Airline', 'Direction', 'Origin', 'Destination']
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].fillna('Unknown'))
                self.label_encoders[col] = le
        
        print(f"Data loaded: {len(self.data)} flights")
        return self.data.head()

    def analyze_peak_times_and_delays(self):
        """Analyze peak-time delays and patterns"""
        print("\n=== PEAK TIME AND DELAY ANALYSIS ===")
        
        # Hourly traffic and delays
        hourly_stats = self.data.groupby('Hour_Slot').agg({
            'Flight_Number': 'count',
            'Departure_Delay_mins': ['mean', 'std', 'max'],
            'Arrival_Delay_mins': ['mean', 'std', 'max']
        }).round(2)
        
        hourly_stats.columns = ['Flight_Count', 'Avg_Dep_Delay', 'Std_Dep_Delay', 
                                'Max_Dep_Delay', 'Avg_Arr_Delay', 'Std_Arr_Delay', 'Max_Arr_Delay']
        
        print("\nHourly Traffic and Delay Statistics:")
        print(hourly_stats.sort_values('Flight_Count', ascending=False))
        
        # Peak hours identification
        peak_hours = hourly_stats.nlargest(5, 'Flight_Count').index.tolist()
        print(f"\nPeak Hours: {peak_hours}")
        
        # Delay distribution analysis
        delay_ranges = pd.cut(self.data['Departure_Delay_mins'], 
                              bins=[-np.inf, 0, 15, 30, 60, np.inf], 
                              labels=['Early', 'On-Time', 'Minor-Delay', 'Major-Delay', 'Severe-Delay'])
        
        delay_dist = delay_ranges.value_counts()
        print(f"\nDelay Distribution:")
        for category, count in delay_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"{category}: {count} flights ({percentage:.1f}%)")
        
        return hourly_stats, peak_hours

    def analyze_runway_constraints(self):
        """Analyze runway capacity constraints"""
        print("\n=== RUNWAY CONSTRAINT ANALYSIS ===")
        
        # Estimate runway capacity based on flight frequency
        runway_capacity = {}
        
        for hour in range(24):
            hour_data = self.data[self.data['Hour_Slot'] == hour]
            if len(hour_data) > 0:
                # Estimate movements per hour (departures + arrivals)
                movements = len(hour_data)
                avg_delay = hour_data['Departure_Delay_mins'].mean()
                
                # Simple capacity model: high delays indicate capacity constraints
                if avg_delay > 30:
                    capacity_utilization = "Over-Capacity"
                elif avg_delay > 15:
                    capacity_utilization = "Near-Capacity"
                else:
                    capacity_utilization = "Under-Capacity"
                
                runway_capacity[hour] = {
                    'movements': movements,
                    'avg_delay': avg_delay,
                    'status': capacity_utilization
                }
        
        # Convert to DataFrame for easier analysis
        capacity_df = pd.DataFrame(runway_capacity).T
        print("\nRunway Capacity Analysis by Hour:")
        print(capacity_df.sort_values('avg_delay', ascending=False))
        
        return capacity_df

    def detect_cascading_disruptions(self):
        """Detect cascading disruption patterns"""
        print("\n=== CASCADING DISRUPTION ANALYSIS ===")
        
        # Group by airline to track aircraft rotations
        cascade_analysis = {}
        
        for airline in self.data['Airline'].unique():
            # Use the datetime object for accurate sorting
            airline_data = self.data[self.data['Airline'] == airline].sort_values('Sched_Departure_Time_dt')
            
            # Simple cascade detection: consecutive flights with increasing delays
            cascades = 0
            for i in range(1, len(airline_data)):
                prev_delay = airline_data.iloc[i-1]['Departure_Delay_mins']
                curr_delay = airline_data.iloc[i]['Departure_Delay_mins']
                
                if curr_delay > prev_delay and curr_delay > 15:
                    cascades += 1
            
            cascade_analysis[airline] = {
                'total_flights': len(airline_data),
                'cascade_events': cascades,
                'cascade_rate': cascades / len(airline_data) if len(airline_data) > 0 else 0
            }
        
        cascade_df = pd.DataFrame(cascade_analysis).T.sort_values('cascade_rate', ascending=False)
        print("Cascading Disruption Analysis by Airline:")
        print(cascade_df.head(10))
        
        return cascade_df

    def build_delay_prediction_models(self):
        """Build AI models to predict delays"""
        print("\n=== BUILDING DELAY PREDICTION MODELS ===")
        
        # Prepare features - only use columns that actually exist
        feature_cols = [
            'Hour_Slot', 'Is_Peak_Hour', 'Day_of_Week', 
            'Airline_encoded', 'Direction_encoded', 'Origin_encoded', 'Destination_encoded'
        ]
        
        # Ensure all feature columns exist in the dataframe before using them
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        print(f"Using features: {available_features}")
        
        # Remove rows with missing target values or features
        model_data = self.data.dropna(subset=available_features + ['Departure_Delay_mins'])
        
        if model_data.empty:
            print("Error: No data available for modeling after dropping missing values.")
            return {}
            
        X = model_data[available_features]
        y = model_data['Departure_Delay_mins']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        model_results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Linear Regression
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else: # Use original data for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {'model': model, 'mae': mae, 'r2': r2}
            print(f"MAE: {mae:.2f}, R²: {r2:.3f}")
            self.models[name] = model
        
        # Neural Network model if TensorFlow is available
        if TF_AVAILABLE and len(X_train) > 100:
            print("\nTraining Neural Network...")
            nn_model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            nn_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
            
            nn_pred = nn_model.predict(X_test_scaled).flatten()
            nn_mae = mean_absolute_error(y_test, nn_pred)
            nn_r2 = r2_score(y_test, nn_pred)
            
            model_results['Neural Network'] = {'model': nn_model, 'mae': nn_mae, 'r2': nn_r2}
            print(f"Neural Network - MAE: {nn_mae:.2f}, R²: {nn_r2:.3f}")
            self.models['Neural Network'] = nn_model
        
        # Select best model based on MAE
        if model_results:
            best_model_name = min(model_results, key=lambda x: model_results[x]['mae'])
            print(f"\nBest Model: {best_model_name}")
        
        return model_results

    def optimize_flight_schedules(self):
        """Optimize flight schedules using mathematical optimization"""
        print("\n=== OPTIMIZING FLIGHT SCHEDULES ===")
        
        # Get hourly capacity constraints
        hourly_capacity = self.data.groupby('Hour_Slot').size().to_dict()
        if not hourly_capacity:
            print("Cannot optimize, no hourly data available.")
            return None
        max_hourly_capacity = max(hourly_capacity.values()) * 1.2  # 20% buffer
        
        # Simple slot optimization using PuLP
        prob = pulp.LpProblem("Flight_Schedule_Optimization", pulp.LpMinimize)
        
        # Decision variables: flight assignments to time slots
        flights = self.data['Flight_Number'].unique()[:50]  # Limit for demo
        time_slots = range(24)
        
        # Binary variables: x[i,t] = 1 if flight i is assigned to slot t
        x = pulp.LpVariable.dicts("x", (flights, time_slots), cat='Binary')
        
        # Objective: Minimize total expected delay
        expected_delays = self.data.groupby('Hour_Slot')['Departure_Delay_mins'].mean().to_dict()
        
        prob += pulp.lpSum([x[f][t] * expected_delays.get(t, 0) for f in flights for t in time_slots])
        
        # Constraints
        # Each flight must be assigned to exactly one slot
        for f in flights:
            prob += pulp.lpSum([x[f][t] for t in time_slots]) == 1
        
        # Capacity constraints
        for t in time_slots:
            prob += pulp.lpSum([x[f][t] for f in flights]) <= max_hourly_capacity
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        optimized_schedule = {}
        total_delay_reduction = 0
        
        if prob.status == pulp.LpStatusOptimal:
            flight_to_slot = self.data.set_index('Flight_Number')['Hour_Slot'].to_dict()
            for f in flights:
                for t in time_slots:
                    if x[f][t].value() == 1:
                        original_slot = flight_to_slot.get(f)
                        if original_slot is not None:
                            original_delay = expected_delays.get(original_slot, 0)
                            new_delay = expected_delays.get(t, 0)
                            delay_reduction = original_delay - new_delay
                            
                            optimized_schedule[f] = {
                                'original_slot': original_slot,
                                'optimized_slot': t,
                                'delay_reduction': delay_reduction
                            }
                            total_delay_reduction += delay_reduction
            
            print(f"Optimization completed. Potential delay reduction: {total_delay_reduction:.2f} minutes")
            rescheduled_count = len([flight for flight in optimized_schedule.values() if flight['original_slot'] != flight['optimized_slot']])
            print(f"Flights rescheduled: {rescheduled_count}")
        else:
            print("Optimization failed or was not feasible.")
            
        return optimized_schedule

    def identify_high_impact_flights(self):
        """Identify high-impact flights that cause maximum disruption"""
        print("\n=== IDENTIFYING HIGH-IMPACT FLIGHTS ===")
        
        # Pre-calculate counts to speed up the loop
        airline_counts = self.data['Airline'].value_counts()
        destination_counts = self.data['Destination'].value_counts()
        total_flights = len(self.data)

        # Use .apply() for a more efficient calculation
        def calculate_impact_score(row):
            score = 0
            # Factor 1: Delay magnitude
            score += row['Departure_Delay_mins'] * 0.3
            # Factor 2: Peak hour multiplier
            if row['Is_Peak_Hour']:
                score *= 1.5
            # Factor 3: Airline connectivity
            score += (airline_counts.get(row['Airline'], 0) / total_flights) * 100
            # Factor 4: Route popularity
            score += (destination_counts.get(row['Destination'], 0) / total_flights) * 50
            return score

        self.data['Impact_Score'] = self.data.apply(calculate_impact_score, axis=1)

        # Sort and display the top impactful flights
        high_impact_flights_df = self.data.sort_values('Impact_Score', ascending=False).head(20)

        print("Top 20 High-Impact Flights:")
        for i, row in high_impact_flights_df.iterrows():
            print(f"{i+1:2}. {row['Flight_Number']}: Score={row['Impact_Score']:.1f}, "
                  f"Delay={row['Departure_Delay_mins']}min, {row['Airline']}, "
                  f"Hour={row['Hour_Slot']}")
                  
        return high_impact_flights_df

    def nlp_interface(self, query):
        """Natural Language Processing interface for querying insights"""
        query_lower = query.lower()
        
        # Simple NLP pattern matching
        if any(word in query_lower for word in ['peak', 'busy', 'busiest']):
            hourly_stats = self.data.groupby('Hour_Slot')['Flight_Number'].count()
            busiest_hour = hourly_stats.idxmax()
            busiest_count = hourly_stats.max()
            return f"The busiest hour is {busiest_hour}:00 with {busiest_count} flights."
        
        elif any(word in query_lower for word in ['delay', 'delayed', 'late']):
            avg_delay = self.data['Departure_Delay_mins'].mean()
            delayed_flights = len(self.data[self.data['Departure_Delay_mins'] > 15])
            total_flights = len(self.data)
            delay_rate = (delayed_flights / total_flights) * 100
            return f"Average delay is {avg_delay:.1f} minutes. {delayed_flights} flights ({delay_rate:.1f}%) are delayed >15min."
        
        elif any(word in query_lower for word in ['airline', 'carrier']):
            airline_delays = self.data.groupby('Airline')['Departure_Delay_mins'].mean().sort_values(ascending=False)
            best_airline = airline_delays.idxmin()
            worst_airline = airline_delays.idxmax()
            return f"Best on-time performance: {best_airline} ({airline_delays[best_airline]:.1f}min avg delay). Worst: {worst_airline} ({airline_delays[worst_airline]:.1f}min)."
        
        elif any(word in query_lower for word in ['optimize', 'improve', 'reduce']):
            return "Based on analysis, consider: 1) Redistribute flights from peak hours (e.g., 6-8 AM, 5-9 PM), 2) Prioritize handling for high-impact flights, 3) Implement dynamic scheduling based on real-time conditions."
        
        else:
            return "I can help with: peak times, delays, airline performance, and optimization. Try 'busiest hours' or 'delay patterns'."

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FLIGHT SCHEDULE OPTIMIZATION REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\nDATASET OVERVIEW:")
        print(f"Total flights analyzed: {len(self.data)}")
        print(f"Date range: {self.data['Sched_Departure_Date'].min()} to {self.data['Sched_Departure_Date'].max()}")
        print(f"Airlines covered: {self.data['Airline'].nunique()}")
        print(f"Destinations: {self.data['Destination'].nunique()}")
        
        # Key insights
        avg_delay = self.data['Departure_Delay_mins'].mean()
        on_time_rate = (len(self.data[self.data['Departure_Delay_mins'] <= 15]) / len(self.data)) * 100
        
        print(f"\nKEY PERFORMANCE INDICATORS:")
        print(f"Average departure delay: {avg_delay:.1f} minutes")
        print(f"On-time performance (≤15min): {on_time_rate:.1f}%")
        
        # Peak hour analysis
        hourly_traffic = self.data.groupby('Hour_Slot')['Flight_Number'].count()
        peak_hours = hourly_traffic.nlargest(3).index.tolist()
        print(f"Peak hours: {', '.join([f'{h}:00' for h in peak_hours])}")
        
        # Recommendations
        print(f"\nOPTIMIZATION RECOMMENDATIONS:")
        print("1. Redistribute flights from peak hours to off-peak times to balance load.")
        print("2. Implement predictive delay management for identified high-impact flights.")
        print("3. Establish dynamic slot allocation based on real-time conditions and predictions.")
        print("4. Focus operational improvements on airlines with the highest cascade failure rates.")
        print("5. Consider ground delay programs during predicted high-congestion periods.")

    def create_visualizations(self):
        """Create visualization charts"""
        print("\nGenerating visualizations...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Flight Schedule Analysis Dashboard', fontsize=20)
        
        # 1. Hourly traffic and average delay
        hourly_stats = self.data.groupby('Hour_Slot').agg(
            Flight_Count=('Flight_Number', 'count'),
            Avg_Delay=('Departure_Delay_mins', 'mean')
        ).reset_index()
        
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        ax1.bar(hourly_stats['Hour_Slot'], hourly_stats['Flight_Count'], color='skyblue', label='Flight Count')
        ax2.plot(hourly_stats['Hour_Slot'], hourly_stats['Avg_Delay'], color='red', marker='o', label='Avg Delay (min)')
        ax1.set_title('Hourly Traffic vs. Average Delay')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Flights', color='skyblue')
        ax2.set_ylabel('Average Delay (minutes)', color='red')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_xticks(range(0, 24, 2))

        # 2. Delay distribution pie chart
        delay_bins = [-np.inf, 0, 15, 30, 60, np.inf]
        delay_labels = ['Early/On-time (<0)', 'On-time (0-15)', 'Minor Delay (15-30)', 'Major Delay (30-60)', 'Severe Delay (>60)']
        delay_counts = pd.cut(self.data['Departure_Delay_mins'], bins=delay_bins, labels=delay_labels).value_counts().sort_index()
        colors = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FF9800', '#F44336']
        axes[0, 1].pie(delay_counts, labels=delay_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=140, wedgeprops=dict(width=0.4))
        axes[0, 1].set_title('Flight Delay Distribution')

        # 3. Top 10 Airlines by Average Delay
        airline_perf = self.data.groupby('Airline')['Departure_Delay_mins'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=airline_perf.values, y=airline_perf.index, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Top 10 Airlines by Average Delay')
        axes[1, 0].set_xlabel('Average Delay (minutes)')
        axes[1, 0].set_ylabel('Airline')

        # 4. Impact Score vs. Delay
        impact_data = self.data.sample(n=min(1000, len(self.data)), random_state=42) # Sample for readability
        sns.scatterplot(data=impact_data, x='Departure_Delay_mins', y='Impact_Score', 
                        hue='Is_Peak_Hour', ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Delay vs. Impact Score')
        axes[1, 1].set_xlabel('Departure Delay (minutes)')
        axes[1, 1].set_ylabel('Calculated Impact Score')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def main():
    """Main execution function"""
    optimizer = FlightScheduleOptimizer()
    
    print("Flight Schedule Optimization System")
    print("="*35)
    
    # Centralized data loading and preprocessing
    try:
        optimizer.load_and_preprocess_data("flights_deduplicated_by_subset.csv")
    except FileNotFoundError:
        print("Error: 'flights_deduplicated_by_subset.csv' not found.")
        print("Please make sure the CSV file is in the same directory as the script.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # Run the full analysis pipeline
    try:
        # 1. Core Analyses
        optimizer.analyze_peak_times_and_delays()
        optimizer.analyze_runway_constraints()
        optimizer.detect_cascading_disruptions()
        
        # 2. AI Modeling
        optimizer.build_delay_prediction_models()
        
        # 3. Optimization and Impact Analysis
        optimizer.optimize_flight_schedules()
        optimizer.identify_high_impact_flights()
        
        # 4. Reporting
        optimizer.generate_comprehensive_report()
        
        # 5. NLP Interface Demo
        print("\n" + "="*40)
        print("NLP INTERFACE DEMO")
        print("="*40)
        
        sample_queries = [
            "What are the busiest hours?",
            "Show me delay patterns",
            "How can we improve the schedule?",
            "Which airline performs best?"
        ]
        
        for query in sample_queries:
            response = optimizer.nlp_interface(query)
            print(f"\nQ: {query}")
            print(f"A: {response}")
        
        # 6. Visualizations
        optimizer.create_visualizations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        print("Please ensure your data file has the correct format and columns.")

if __name__ == "__main__":
    main()