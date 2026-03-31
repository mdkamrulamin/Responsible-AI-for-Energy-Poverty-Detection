# Responsible AI for Energy Poverty Detection

A privacy-preserving proof of concept that uses Ontario electricity demand data and weather data to detect community energy stress using an LSTM autoencoder and weather-linked anomaly interpretation.

## Overview

This project explores whether public electricity demand data and weather stress indicators can be used to generate early warning signals of possible community energy stress in Ontario. The system is designed to be privacy-preserving, explainable, reproducible, and useful for human-in-the-loop decision support.

## Features

- Hourly Ontario electricity demand preprocessing
- Ontario weather proxy creation from multiple stations
- Merged master dataset generation
- Sequence preparation for LSTM modeling
- Unsupervised LSTM autoencoder training
- Reconstruction-error-based anomaly detection
- Weather-linked stress event detection
- Monthly summaries and top event extraction
- Streamlit dashboard for interactive visualization

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

## Data Setup

Place the following files inside the `data/` folder:

* `ontario_electricity_demand.csv`
* `ontario_weather_proxy_2018_2023_heat24.csv`

## Run the Project

To run the backend pipeline:

```bash
python main.py
```

To launch the dashboard:

```bash
streamlit run app.py
```

## Dashboard

The Streamlit dashboard includes:

* **Demand & Anomalies** — hourly demand and anomaly points
* **Weather Stress Context** — temperature, cold stress, heat stress, and stress events
* **Early Warning Summary** — KPI cards, monthly summaries, and top stress events

## Outputs

The project generates outputs such as:

* merged master dataset
* processed sequence artifacts
* trained LSTM autoencoder model
* training history and plots
* anomaly result files
* aggregated stress-event summaries
* top event tables for dashboard use

## Example Visuals

> Replace these filenames with your actual image filenames placed in the project root.

![Dashboard Overview](dashboard_overview.png)

![Demand and Anomalies](demand_anomalies.png)

![Weather Stress Context](weather_stress_context.png)

![Monthly Summary](monthly_summary.png)

## Why This Project Matters

This project demonstrates how AI can be used responsibly in a socially important setting. Instead of monitoring individuals, it uses aggregated public data to create a non-intrusive early-warning framework that can support municipalities, policymakers, utilities, and community organizations.

## Limitations

* Provincial demand is only a proxy for community energy stress
* The system does not measure household hardship directly
* The model is unsupervised, so there is no ground-truth label for true stress events
* The system is intended for decision support, not automated intervention

## Future Improvements

Possible future work includes:

* adding more weather variables
* comparing with baseline models such as Isolation Forest
* dynamic or seasonal anomaly thresholds
* improved event grouping and severity logic
* richer dashboard interactivity
* testing alternative sequence lengths and model architectures

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* TensorFlow / Keras
* matplotlib
* Plotly
* Streamlit

## License

This project is intended for academic and research demonstration purposes.

```
```
