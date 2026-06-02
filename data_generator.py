"""
IoT Data Generator for TCSP-PC
Simulates DAR TE-style data: vehicle density, CO2 emissions, traffic patterns
Based on North American 1km grid, 2008-2017 range
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

# Cities with approximate traffic characteristics
CITY_PROFILES = {
    "New York":       {"base_density": 68, "emission_factor": 1.4, "peak_ratio": 2.1},
    "Los Angeles":    {"base_density": 75, "emission_factor": 1.6, "peak_ratio": 2.4},
    "Chicago":        {"base_density": 55, "emission_factor": 1.3, "peak_ratio": 1.9},
    "Houston":        {"base_density": 62, "emission_factor": 1.7, "peak_ratio": 2.0},
    "Detroit":        {"base_density": 48, "emission_factor": 1.5, "peak_ratio": 1.8},
    "San Francisco":  {"base_density": 52, "emission_factor": 1.2, "peak_ratio": 1.7},
    "Boston":         {"base_density": 44, "emission_factor": 1.1, "peak_ratio": 1.6},
    "Rural Midwest":  {"base_density": 12, "emission_factor": 0.9, "peak_ratio": 1.3},
}

VEHICLE_TYPES = {
    "sedan":       {"weight": 0.35, "base_emission": 12.0, "label": 0},  # eco
    "suv":         {"weight": 0.25, "base_emission": 22.0, "label": 0},  # eco
    "hybrid":      {"weight": 0.10, "base_emission": 8.0,  "label": 0},  # eco
    "truck":       {"weight": 0.15, "base_emission": 38.0, "label": 1},  # polluting
    "bus":         {"weight": 0.08, "base_emission": 45.0, "label": 1},  # polluting
    "old_diesel":  {"weight": 0.07, "base_emission": 52.0, "label": 1},  # polluting
}

POLLUTION_THRESHOLD = 30.0  # g/km — above this = polluting


def hour_factor(hour: int) -> float:
    """Traffic volume multiplier by hour of day (0-23)"""
    if 7 <= hour <= 9:    return 2.0   # morning peak
    if 17 <= hour <= 19:  return 2.2   # evening peak
    if 22 <= hour or hour <= 5: return 0.3   # night
    return 1.0


def day_factor(day_of_week: int) -> float:
    """0=Monday ... 6=Sunday"""
    if day_of_week < 5:  return 1.0   # weekday
    if day_of_week == 5: return 0.85  # Saturday
    return 0.65   # Sunday


def generate_emission_record(
    city: str,
    timestamp: datetime,
    distance_km: float,
    vehicle_type: str,
) -> dict:
    """Generate one IoT sensor reading for a vehicle"""
    profile = CITY_PROFILES[city]
    vtype = VEHICLE_TYPES[vehicle_type]

    # Emission rate (g/km) with realistic noise
    base_em = vtype["base_emission"]
    hour_mult = hour_factor(timestamp.hour)
    day_mult  = day_factor(timestamp.weekday())

    # Add road-condition noise and acceleration effects
    noise = np.random.normal(0, base_em * 0.12)
    accel_factor = 1.0 + np.random.exponential(0.08)  # acceleration spikes

    emission_rate = max(1.0, (base_em + noise) * accel_factor)

    # Vehicle density (vehicles/km at this point)
    base_density = profile["base_density"] * hour_mult * day_mult
    density = max(1, int(np.random.normal(base_density, base_density * 0.2)))

    # Pollution level classification
    pollution_label = 1 if emission_rate >= POLLUTION_THRESHOLD else 0

    # Travel time (minutes per km)
    congestion_factor = 1.0 + (density / 100.0) * 0.5
    travel_time = np.random.normal(2.5 * congestion_factor, 0.4)

    return {
        "timestamp":        timestamp.isoformat(),
        "city":             city,
        "hour":             timestamp.hour,
        "day_of_week":      timestamp.weekday(),
        "vehicle_type":     vehicle_type,
        "distance_km":      round(distance_km, 2),
        "emission_rate":    round(emission_rate, 3),      # g/km
        "vehicle_density":  density,                       # vehicles/km
        "travel_time_min":  round(travel_time, 2),
        "emission_label":   pollution_label,               # 0=eco, 1=polluting
        "city_emission_factor": profile["emission_factor"],
    }


def generate_dataset(
    n_samples: int = 15000,
    start_year: int = 2008,
    end_year: int = 2017,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """Generate full synthetic IoT dataset"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n_samples} IoT sensor records...")

    records = []
    cities = list(CITY_PROFILES.keys())
    vehicle_types = list(VEHICLE_TYPES.keys())
    vtype_weights = [VEHICLE_TYPES[v]["weight"] for v in vehicle_types]

    start_dt = datetime(start_year, 1, 1)
    end_dt   = datetime(end_year, 12, 31)
    total_seconds = (end_dt - start_dt).total_seconds()

    for _ in range(n_samples):
        city = np.random.choice(cities)
        vtype = np.random.choice(vehicle_types, p=vtype_weights)
        rand_seconds = np.random.uniform(0, total_seconds)
        ts = start_dt + timedelta(seconds=rand_seconds)
        dist = np.random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        rec = generate_emission_record(city, ts, dist, vtype)
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save raw data
    raw_path = os.path.join(output_dir, "iot_traffic_data.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} records to {raw_path}")

    # Print class distribution
    eco = (df["emission_label"] == 0).sum()
    pol = (df["emission_label"] == 1).sum()
    print(f"Class distribution: Eco-friendly={eco} ({eco/len(df)*100:.1f}%), "
          f"Polluting={pol} ({pol/len(df)*100:.1f}%)")

    return df


def generate_realtime_sample(city: str = "New York") -> dict:
    """Generate a single real-time IoT reading (used by dashboard API)"""
    ts = datetime.now()
    vehicle_types = list(VEHICLE_TYPES.keys())
    vtype_weights  = [VEHICLE_TYPES[v]["weight"] for v in vehicle_types]
    vtype = np.random.choice(vehicle_types, p=vtype_weights)
    dist  = float(np.random.choice([10, 20, 30, 40, 50]))
    return generate_emission_record(city, ts, dist, vtype)


if __name__ == "__main__":
    df = generate_dataset(n_samples=15000)
    print("\nSample records:")
    print(df.head(5).to_string())
    print(f"\nEmission rate stats:\n{df['emission_rate'].describe()}")