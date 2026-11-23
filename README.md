# Flight-Regime Recognition for a Multirotor UAV

---

## 1. Dataset at a Glance

| Item | Value |
|---|---|
| Name | *Data Collected with Package Delivery Quadcopter Drone* (a.k.a. "In-flight positional and energy use data set of a DJI Matrice 100 quadcopter") |
| Original source | Carnegie Mellon University (Dept. of Civil & Environmental Engineering + Robotics Institute), Rodrigues *et al.*, published in *Scientific Data* (Nature), 2021 |
| Flights | 209 recordings: 195 cruise flights + 14 hover / idle recordings |
| Total flight time | ~10 h 45 min, ~65 km flown |
| Samples | ~2.3 × 10⁵ time-stamped rows |
| Features | ~25 numeric sensor channels + 3 per-flight operating parameters |
| Sample rate | Sensors synchronized to ≈5–10 Hz |
| Format | CSV (one file per flight in `flights.zip`, plus `parameters.csv`) |

---

## 2. Original Source and How to Get It

**Original creators:**
Thiago A. Rodrigues, Jay Patrikar, Arnav Choudhry, Jacob Feldgoise, Vaibhav Arcot, Aradhana Gahlaut, Sophia Lau, Brady Moon, Bastian Wagner, H. Scott Matthews, Sebastian Scherer, Constantine Samaras — Carnegie Mellon University.

**Data descriptor paper:**
T. A. Rodrigues *et al.*, "In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery," *Scientific Data*, vol. 8, no. 1, Art. 155, 2021.
DOI: https://doi.org/10.1038/s41597-021-00930-x · Open-access preprint: https://arxiv.org/abs/2103.13313

**Download (official repository: CMU KiltHub, a Figshare instance):**
- DOI: https://doi.org/10.1184/R1/12683453
- Page: https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453
- Alternate link: https://api.figshare.com/v2/articles/12683453
- A copy of the dataset is in the [`data/`](data/) folder: [`flights.csv`](data/flights.csv), [`flights.zip`](data/flights.zip), [`parameters.csv`](data/parameters.csv), [`raw_files.zip`](data/raw_files.zip), and the original [`README.txt`](data/README.txt).
  - `flights.csv` and `raw_files.zip` are over 100 MB and are stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning, or run `git lfs pull` after cloning, to get the real files instead of small pointer files.

---

## 3. What the Dataset Is About

CMU flew an autonomous package-delivery quadcopter along a fixed triangular route over and over, changing three things between flights:

| Operating parameter | Values |
|---|---|
| Commanded cruise ground speed | 4, 6, 8, 10, 12 m/s |
| Commanded cruise altitude | 25, 50, 75, 100 m |
| Payload mass | 0, 250, 500 g (take-off mass 3680 / 3930 / 4180 g) |

Each speed/altitude/payload combination was flown at least three times. Every flight has the same structure, and that structure is what makes it useful for regime recognition:

1. **Vertical take-off / climb** at ≈2.5 m/s up to cruise altitude
2. **Turn** (yaw) toward the first waypoint
3. **Acceleration and straight-line cruise** at the commanded speed
4. **Stop / hover** at each waypoint, then turn toward the next one (three legs)
5. Return to launch point, stop, and **vertical descent** at ≈1.5 m/s
6. Land and disarm (**idle on ground**)

---

## 4. Features

All columns from the per-flight CSV files (Table 1 of the data paper):

### 4.1 Per-flight operating parameters (constant within a flight)
| Column | Unit | Description |
|---|---|---|
| `flight` | – | Flight ID |
| `speed` | m/s | Commanded cruise ground speed |
| `payload` | g | Payload mass |
| `altitude` | m | Commanded cruise altitude |
| `date`, `local_time` | – | Date and start time of the flight |
| `route` | – | Pre-defined path (R1 is the main triangular route) |

### 4.2 Time-series sensor channels
| Group | Columns | Unit | Sensor |
|---|---|---|---|
| Time | `time` | s | Elapsed time |
| Air data | `wind_speed`, `wind_angle` | m/s, deg | Ultrasonic anemometer |
| Power | `battery_voltage`, `battery_current` | V, A | Voltage/current sensor |
| Position | `position_x` (lon), `position_y` (lat), `position_z` (altitude) | deg, deg, m | GNSS/INS |
| Attitude | `orientation_x/y/z/w` | quaternion | IMU (converted to roll/pitch/yaw) |
| Velocity | `velocity_x/y/z` | m/s | GNSS/INS |
| Angular rate | `angular_x/y/z` | rad/s | IMU gyroscope |
| Acceleration | `linear_acceleration_x/y/z` | m/s² | IMU accelerometer |

Reported sensor accuracy: horizontal position ±2 m, vertical position ±5 m, velocity ±0.1 m/s, attitude ±2°, current ±1.2%.

### 4.3 Derived features I may add
- Roll, roll rate, pitch, yaw, and yaw rate
- Horizontal speed `√(vx² + vy²)`, climb rate `vz`, height above take-off point
- Rolling-window statistics (mean, std, min, max, slope) over 1–5 second windows

---

## 5. Labels

**The raw dataset has no regime labels.** The data paper says this directly. I will create labels with transparent physics-based rules, following the five-state scheme used by Palamas *et al.* (2023) on this same dataset:

| Label | Regime | Rule (applied to smoothed GNSS/INS signals) |
|---|---|---|
| 0 | **IDLE / HOVER** | Low horizontal speed, low \|v_z\|, low yaw rate |
| 1 | **ASCEND** | v_z > +threshold |
| 2 | **DESCEND** | v_z < −threshold |
| 3 | **TURN** | \|yaw rate\| > threshold, low translation |
| 4 | **HMSL** (horizontal movement, straight line / cruise) | Horizontal speed > threshold, low yaw rate |

Thresholds will be picked from the commanded values (2.5 m/s climb, 1.5 m/s descent, known cruise speeds) and checked by plotting labeled flights. The classes are heavily imbalanced, where cruise and hover dominate and turns are short, so I will use class weighting and any other methods to make up for this.

---

## 6. Problem I Want to Solve

### 6.1 Main task: flight-regime recognition (multi-class classification)
Given a short window (1–5 s) of onboard sensor data, predict which of the five flight regimes the aircraft is in.

I will reproduce and try to outperform the following papers. 

- **A. Palamas, N. Souli, T. Panayiotou, P. Kolios, G. Ellinas**, "A Multi-task Learning Framework for Drone State Identification and Trajectory Prediction," arXiv:2309.06741, 2023.
  - Same dataset (182 flights on route R1), same five states, LSTM model, 30-step windows.
  - Reported: **98.51% accuracy, macro-F1 = 0.869**. TURN is the weakest class (F1 ≈ 0.59).
- Follow-up: N. Souli *et al.*, "A Multi-task Transformer Architecture for Drone State Identification and Trajectory Prediction," IEEE DCOSS-IoT 2024.

---

## 7. Why This Is an Engineering Problem

For helicopters, Health and Usage Monitoring Systems (HUMS) identify flight regimes so that fatigue damage to life-limited parts (rotor, gearbox, airframe) can be charged to the regimes actually flown, instead of assuming worst-case usage. Getting the regime right directly affects component retirement times, maintenance cost and safety. A quadcopter is a small rotorcraft with the same basic regimes (hover, climb, descent, turn, forward flight), so this dataset is a public, reproducible stand-in for that problem. 

A reliable regime classifier is the first stage of most flight-data monitoring pipelines. "Normal" behaviour is defined per regime, so anomalies can only be flagged once the regime is known.

The data comes from physical sensors flown outdoors in varying wind. It includes real sensor error, a documented anemometer bias, and flights that drifted off the commanded altitude, which are the kinds of problems found in flight-test data.

---

## 8. References

1. T. A. Rodrigues *et al.*, "In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery," *Scientific Data*, 8, 155 (2021). doi:10.1038/s41597-021-00930-x
2. T. A. Rodrigues *et al.*, *Data Collected with Package Delivery Quadcopter Drone*, Carnegie Mellon University KiltHub, dataset (2021). doi:10.1184/R1/12683453
3. A. Palamas, N. Souli, T. Panayiotou, P. Kolios, G. Ellinas, "A Multi-task Learning Framework for Drone State Identification and Trajectory Prediction," arXiv:2309.06741 (2023).
4. N. Souli, A. Palamas, T. Panayiotou, P. Kolios, G. Ellinas, "A Multi-task Transformer Architecture for Drone State Identification and Trajectory Prediction," *IEEE DCOSS-IoT*, pp. 285–291 (2024). doi:10.1109/DCOSS-IoT61029.2024.00051
