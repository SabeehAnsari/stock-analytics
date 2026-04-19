import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["JAVA_HOME"]   = "C:\\Users\\Sabeeh\\AppData\\Local\\Programs\\Eclipse Adoptium\\jdk-21.0.10.7-hotspot"
os.environ["PATH"] += ";C:\\hadoop\\bin;C:\\Users\\Sabeeh\\AppData\\Local\\Programs\\Eclipse Adoptium\\jdk-21.0.10.7-hotspot\\bin"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, avg, max, min,
    round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, LongType
)

KAFKA_BROKER           = "localhost:9092"
KAFKA_TOPIC            = "stock-stream"
ALERTS_PATH            = "alerts/alert_log.csv"
PRICE_CHANGE_THRESHOLD = 2.0


def create_spark_session():
    return (
        SparkSession.builder
        .appName("StockStreamProcessor")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def define_schema():
    return StructType([
        StructField("date",   StringType(), True),
        StructField("ticker", StringType(), True),
        StructField("open",   DoubleType(), True),
        StructField("high",   DoubleType(), True),
        StructField("low",    DoubleType(), True),
        StructField("close",  DoubleType(), True),
        StructField("volume", LongType(),   True),
    ])


def run_stream():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    schema = define_schema()

    # Read from Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    # Parse JSON
    parsed = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    # Flag anomaly on raw rows — high/low spread > threshold % of close
    with_anomaly = parsed.withColumn(
        "anomaly",
        ((col("high") - col("low")) / col("close") * 100) > PRICE_CHANGE_THRESHOLD
    )

    # Query 1 — print aggregated summary to console every 10 seconds
    aggregated = (
        with_anomaly
        .groupBy("ticker")
        .agg(
            spark_round(avg("close"), 2).alias("avg_close"),
            spark_round(max("high"),  2).alias("period_high"),
            spark_round(min("low"),   2).alias("period_low"),
            spark_round(avg("volume"),0).alias("avg_volume")
        )
    )

    console_query = (
        aggregated.writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", False)
        .trigger(processingTime="10 seconds")
        .start()
    )

    # Query 2 — write anomalous rows to CSV (append on raw stream, no aggregation)
    anomaly_query = (
        with_anomaly
        .filter(col("anomaly") == True)
        .writeStream
        .outputMode("append")
        .format("csv")
        .option("path", ALERTS_PATH)
        .option("header", True)
        .option("checkpointLocation", "data/checkpoints/anomaly")
        .trigger(processingTime="10 seconds")
        .start()
    )

    print("\nStream processor running — output every 10 seconds.")
    print("Press Ctrl+C to stop.\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_stream()