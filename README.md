# Stock Analytics Platform — Setup Guide

## Requirements
- Python 3.11
- Docker Desktop
- Java 21 (Eclipse Temurin)

## Setup Steps

### 1. Clone the repo
git clone https://github.com/YourUsername/stock-analytics.git
cd stock-analytics

### 2. Create virtual environment with Python 3.11
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set environment variables
Add these to your System Environment Variables:
- HADOOP_HOME = C:\hadoop
- JAVA_HOME = C:\Users\YourName\AppData\Local\Programs\Eclipse Adoptium\jdk-21.0.10.7-hotspot

Download winutils.exe and hadoop.dll from:
https://github.com/cdarlint/winutils (hadoop-3.3.5/bin/)
Place both files in C:\hadoop\bin\

### 5. Add the dataset
Download all_stocks_5yr.csv from (https://www.kaggle.com/datasets/camnugent/sandp500)
The dataset is already in the directory. It is in data/raw/all_stocks_5yr.csv

### 6. Generate Parquet files
python ingestion/fetch_historical.py

### 7. Train ML models
python ml/train_model.py

### 8. Start Kafka
docker compose up -d

### 9. Run the pipeline (3 terminals)
Terminal 1: python ingestion/kafka_producer.py
Terminal 2: python processing/stream_processor.py
Terminal 3: streamlit run dashboard/app.py