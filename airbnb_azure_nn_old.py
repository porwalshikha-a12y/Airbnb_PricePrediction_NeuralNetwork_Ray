import os
import pandas as pd
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    abs, atan2, avg, broadcast, col, concat_ws, count, length, lower,
    month, radians, regexp_replace, round, row_number, sin, sqrt, sum,
    to_date, trim, when
)
from pyspark.sql.window import Window

from DataCleansingAndEDA import perform_eda
from logger import get_logger
from NeuralNetworks_Ray import run_nn_pipeline
from NeuralNetwork_Model import run_nn

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

az_storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
az_storage_account_key = os.getenv("AZURE_STORAGE_KEY")

AZ_CONTAINER = "airbnb-data"
AZ_BASE_PATH = f"abfss://{AZ_CONTAINER}@{az_storage_account_name}.dfs.core.windows.net"

# Optional local output (for model CSV)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(BASE_DIR, "Data", "Cleansed")
os.makedirs(output_path, exist_ok=True)


# =========================
# SPARK SESSION
# =========================
def create_spark():
    try:
        SparkSession.getActiveSession().stop()
    except:
        pass

    spark = SparkSession.builder \
        .appName('airbnb ADLS pipeline') \
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-azure:3.3.6",
                "org.apache.hadoop:hadoop-azure-datalake:3.3.6",
                "org.apache.hadoop:hadoop-common:3.3.6",
                "com.azure:azure-storage-blob:12.25.0"
            ])
        ) \
        .getOrCreate()

    spark.conf.set(
        f"fs.azure.account.auth.type.{az_storage_account_name}.dfs.core.windows.net",
        "SharedKey"
    )

    spark.conf.set(
        f"fs.azure.account.key.{az_storage_account_name}.dfs.core.windows.net",
        az_storage_account_key
    )

    spark.conf.set(
        "fs.abfss.impl",
        "org.apache.hadoop.fs.azurebfs.SecureAzureBlobFileSystem"
    )

    return spark


# =========================
# LOAD DATA FROM ADLS
# =========================
def load_data(spark):

    base = f"{AZ_BASE_PATH}/raw"

    airbnb = spark.read.csv(
        f"{base}/listings_london.csv",
        header=True, inferSchema=True, multiLine=True, escape='"'
    )

    calendar = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("compression", "gzip") \
        .csv(f"{base}/london_calendar.csv.gz")

    crime = spark.read.csv(
        f"{base}/london_crime_data/",
        header=True, inferSchema=True, recursiveFileLookup=True
    )

    stations = spark.read.csv(
        f"{base}/tfl_stations.csv",
        header=True, inferSchema=True
    )

    tourist = spark.read.csv(
        f"{base}/london_tourist_spots.csv",
        header=True, inferSchema=True
    )

    income_house = spark.read.csv(
        f"{base}/london_houseprices_income.csv",
        header=True, inferSchema=True
    )

    return airbnb, calendar, crime, stations, tourist, income_house


# =========================
# SAVE TO ADLS
# =========================
def save_dataframe_csv(df):
    try:
        df.coalesce(1).write \
            .mode("overwrite") \
            .option("header", True) \
            .csv(f"{AZ_BASE_PATH}/cleansed/")

        print("✅ Data saved to ADLS (cleansed folder)")

    except Exception as e:
        print(f"❌ Error saving to ADLS: {e}")


# =========================
# FEATURE + CLEANING (UNCHANGED)
# =========================
def clean_airbnb_df(airbnb):
    airbnb_df = airbnb.select(
        col("id"),
        col("latitude").cast("double"),
        col("longitude").cast("double"),
        col("price"),
        col("accommodates"),
        col("bedrooms"),
        col("beds"),
        col("bathrooms"),
        col("room_type"),
        col("property_type"),
        col("neighbourhood_cleansed"),
        col("name"),
        col("description"),
        col("amenities"),
        col("review_scores_rating"),
        col("number_of_reviews"),
        col("reviews_per_month"),
        col("availability_365"),
        col("availability_30"),
        col("host_is_superhost"),
        col("host_response_rate"),
        col("host_acceptance_rate")
    )

    airbnb_df = airbnb_df.withColumn("area", trim(lower(col("neighbourhood_cleansed"))))

    airbnb_df = airbnb_df.fillna({"accommodates": 0, "bathrooms": 0, "bedrooms": 0})

    airbnb_df = airbnb_df.withColumn(
        "price",
        regexp_replace(col("price"), "[$,]", "").cast("double")
    )

    airbnb_df = airbnb_df.dropna(subset=["price"])

    airbnb_df = airbnb_df.filter(
        (col("price") >= 20) & (col("price") <= 500)
    )

    return airbnb_df


# =========================
# MAIN PIPELINE
# =========================
def read_raw_data():
    spark = create_spark()
    logger = get_logger()
    logger.info('Pipeline started...')

    airbnb_raw, calendar, crime, stations, tourist, income_house = load_data(spark)

    airbnb_clean_df = clean_airbnb_df(airbnb_raw)

    print("Rows after cleaning:", airbnb_clean_df.count())

    # Save to ADLS
    save_dataframe_csv(airbnb_clean_df)

    return airbnb_clean_df


# =========================
# RUN FULL PIPELINE
# =========================
def run_full_pipeline():
    final_df = read_raw_data()

    eda_df = final_df.toPandas()
    eda_output = perform_eda(eda_df.copy())

    model_df = eda_output.copy()

    # Save locally for modelling
    model_df.to_csv(f"{output_path}/airbnb_model_input.csv", index=False)

    nn_ray_results = run_nn_pipeline(df=model_df)
    nn_results = run_nn(df=model_df)

    results = pd.DataFrame([
        {"Model": "NN Ray", **nn_ray_results["metrics_real"]},
        {"Model": "NN", **nn_results["metrics_real"]}
    ])

    results.to_csv(f"{output_path}/model_results.csv", index=False)

    print(results)

    return results


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_full_pipeline()