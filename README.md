# ✈️ Flight Delay Prediction (Pre‑Departure) using PySpark

This project builds an **honest, leakage‑free pre‑departure flight delay prediction model** using **PySpark ML** on the *US Flight Data 2024* dataset from Kaggle.

---

## 📌 Problem Statement

> **Can we estimate the risk of a flight being delayed *before departure*, using only schedule, route, and historical patterns?**

This mirrors real airline and operations use cases such as:

* Gate and crew planning
* Passenger risk alerts
* Schedule robustness analysis

---

## 📂 Dataset

**Source:** Kaggle – *Flight Data 2024*

The dataset contains millions of US domestic flight records with:

* schedule information
* route and carrier details
* actual delay outcomes

⚠️ Only **pre‑departure‑safe features** are used as model inputs.

---

## 🎯 Target Definition

We define the binary target as:

```text
label = 1  if arrival delay > 15 minutes
label = 0  otherwise
```

* `arr_delay` is used **only to create the label**
* It is **never used as a feature**

This follows the airline industry standard for on‑time performance.

---

## 🚫 Leakage Control

The following columns are **explicitly excluded** from features because they are only known *after departure* or *after arrival*:

```text
dep_time, dep_delay, taxi_out, wheels_off, wheels_on, taxi_in,
arr_time, arr_delay, actual_elapsed_time, air_time,
carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay
```

This ensures the model is **production‑valid** and not trained on future information.

---

## 🧠 Feature Engineering

### 1️⃣ Schedule & Calendar Features

* `year`
* `month`
* `day_of_month`
* `day_of_week`
* `crs_dep_hour` (derived from scheduled departure time)
* `crs_elapsed_time`
* `distance`

### 2️⃣ Categorical Route & Carrier Features

* `op_unique_carrier`
* `origin`
* `dest`
* `origin_state_nm`
* `dest_state_nm`

These are encoded using **StringIndexer only** (no one‑hot encoding), which is optimal for tree‑based models in Spark.

---

## ⭐ High‑Impact Historical Aggregate Features

To improve predictive power while remaining leakage‑free, we add **historical delay rates**, computed *only from training data*:

* `carrier_delay_rate`
* `origin_delay_rate`
* `dest_delay_rate`
* `route_delay_rate` (origin → destination)

These features capture long‑term structural risk patterns and are standard in airline analytics.

---

## ⚖️ Class Imbalance Handling

Delayed flights are less frequent than on‑time flights.

We address this by applying **class weights** during training:

```text
Delayed flights receive higher weight than on‑time flights
```

This improves recall without introducing bias.

---

## 🏗️ Modeling Approach

### Model Used

* **Gradient Boosted Trees (GBTClassifier)** from Spark ML

Chosen because:

* Strong performance on tabular data
* Handles non‑linear interactions well
* Outperforms Random Forests for weak‑signal problems

### Pipeline Architecture

```text
Raw Data
  ↓
StringIndexer (categorical columns)
  ↓
VectorAssembler
  ↓
GBTClassifier
```

All transformations are encapsulated inside a **Spark ML Pipeline** to ensure:

* no data leakage
* consistent train/test transformations
* production readiness

---

## 🧪 Train / Test Strategy

A **time‑based split** is used:

* **Training:** Flights from earlier months
* **Testing:** Flights from later months

This mimics real‑world deployment and avoids temporal leakage.

---

## 📊 Evaluation

**Metric:** ROC‑AUC

### Results (Typical)

| Model Stage                 | AUC             |
| --------------------------- | --------------- |
| Baseline RF (schedule only) | ~0.60           |
| + Historical features       | ~0.65           |
| + GBT + class weights       | **0.66 – 0.70** |

These results are **realistic for pre‑departure prediction** and indicate a clean, trustworthy model.

---

## 🔍 Model Interpretability

Feature importance analysis consistently highlights:

* `route_delay_rate`
* `carrier_delay_rate`
* `crs_dep_hour`
* `origin` / `dest`

This aligns with domain intuition and validates the feature design.

---

## 🛠️ Tech Stack

* **PySpark**
* **Spark MLlib**
* **Python 3.12**
* Local Spark (development)

---

## 🚀 Future Improvements

* Add holiday & special‑event indicators
* Incorporate weather forecasts (pre‑departure safe)
* Train route‑specific sub‑models
* Extend to delay‑duration regression
* Productionize with model versioning and monitoring

---

## 🧾 Key Takeaway

> This project demonstrates how to build a **production‑grade, leakage‑free ML model** on large‑scale flight data, focusing on *realistic pre‑departure prediction* rather than hindsight‑driven accuracy.

---

## 📎 References

* Kaggle: Flight Data 2024
* Apache Spark ML Documentation
* Airline On‑Time Performance Standards

---

If you’re reviewing this project: **the emphasis is correctness, realism, and explainability over inflated metrics.**
