"""
airbnb_azure_nn.py — Full Airbnb London price-prediction pipeline against ADLS.

Reads six raw datasets from Azure Data Lake Storage Gen2, cleans + feature-
engineers the listings data, joins in calendar/crime/stations/tourism/income
features, runs EDA, and trains two MLP models (Ray Tune + sklearn GridSearch).
"""

import os
import pandas as pd
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    abs, atan2, avg, broadcast, col, concat_ws, cos, count, length, lower,
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

# Local output (for model CSV + data dictionary)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(BASE_DIR, "Data", "Cleansed")
os.makedirs(output_path, exist_ok=True)


# =========================
# Avoids data LEAKAGE
# =========================
def get_model_safe_df(df):
    """Drop columns derived from price (or otherwise leaky) before modelling."""
    leakage_cols = [
        "price_per_person",
        "log_price",
        "price_per_guest",
        "price_per_bedroom",
        "price_per_bathroom",
        "value_score",
        "segment",
    ]
    return df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")


# =========================
# BUILD DATA DICTIONARY
# =========================
COLUMN_METADATA = {
    # --- Identifiers / raw listings columns (Inside Airbnb) ---
    "id":                       ("Listings",        "Unique Airbnb listing ID"),
    "latitude":                 ("Listings",        "Listing latitude (WGS84)"),
    "longitude":                ("Listings",        "Listing longitude (WGS84)"),
    "price":                    ("Listings",        "Nightly price in GBP after cleaning, filtered to £20–£500"),
    "accommodates":             ("Listings",        "Maximum number of guests"),
    "bedrooms":                 ("Listings",        "Number of bedrooms"),
    "beds":                     ("Listings",        "Number of beds"),
    "bathrooms":                ("Listings",        "Number of bathrooms"),
    "room_type":                ("Listings",        "Entire home, private room, shared room, hotel room"),
    "property_type":            ("Listings",        "Detailed property type (apartment, townhouse, etc.)"),
    "neighbourhood_cleansed":   ("Listings",        "Borough as published by Inside Airbnb"),
    "name":                     ("Listings",        "Listing title (free text)"),
    "description":              ("Listings",        "Listing description (free text), nulls filled with empty string"),
    "amenities":                ("Listings",        "Comma-separated list of amenities (free text)"),
    "review_scores_rating":     ("Listings",        "Overall guest rating (0–5)"),
    "number_of_reviews":        ("Listings",        "Lifetime review count for the listing"),
    "reviews_per_month":        ("Listings",        "Average reviews per month"),
    "availability_365":         ("Listings",        "Days available to book in next 365 days"),
    "availability_30":          ("Listings",        "Days available to book in next 30 days"),
    "host_is_superhost":        ("Listings",        "Whether host has Superhost status (t/f)"),
    "host_response_rate":       ("Listings",        "Host response rate as percentage string"),
    "host_acceptance_rate":     ("Listings",        "Host acceptance rate as percentage string"),

    # --- Engineered from listings ---
    "area":                     ("Engineered",      "Lower-cased trimmed borough; join key for crime + income"),
    "price_per_person":         ("Engineered",      "price / (accommodates + 0.1); EDA only — leakage risk"),
    "is_central_london":        ("Engineered",      "1 if latitude in (51.50, 51.52), else 0"),
    "desc_length":              ("Engineered",      "Character length of description"),
    "has_luxury_words":         ("Engineered",      "1 if description matches luxury|penthouse|stunning|designer|premium|view|balcony|central"),
    "has_balcony":              ("Engineered",      "1 if amenities mentions 'balcony'"),
    "has_view":                 ("Engineered",      "1 if amenities mentions 'view'"),
    "has_garden":               ("Engineered",      "1 if amenities mentions 'garden'"),
    "has_hot_tub":              ("Engineered",      "1 if amenities mentions 'hot tub' or 'jacuzzi'"),
    "property_room_combo":      ("Engineered",      "property_type + '_' + room_type (interaction)"),
    "luxury_score":             ("Engineered",      "0.4·bathrooms + 0.3·accommodates + 0.2·bedrooms"),

    # --- Cross-source merges ---
    "dominant_season":          ("Calendar",        "Season with most booked days for the listing (winter/spring/summer/autumn)"),
    "area_crime_count":         ("Met Police",      "Number of crimes recorded in the listing's borough"),
    "stations_within_1km":      ("TfL StopPoint",   "Count of tube/DLR/Overground stations within 1 km (Haversine)"),
    "tourist_spots_within_1km": ("Tourist spots",   "Count of curated London tourist spots within 1 km"),
    "house_price":              ("London Datastore","Borough mean median house price (GBP)"),
    "income":                   ("London Datastore","Borough mean household income (GBP)"),
}


def build_data_dictionary(spark_df):
    """Return a pandas DataFrame describing every column in the final dataset."""
    rows = []
    for field in spark_df.schema.fields:
        source, description = COLUMN_METADATA.get(field.name, ("Unknown", ""))
        rows.append({
            "column":      field.name,
            "dtype":       field.dataType.simpleString(),
            "nullable":    field.nullable,
            "source":      source,
            "description": description,
        })
    return pd.DataFrame(rows, columns=["column", "dtype", "nullable", "source", "description"])


# =========================
# SPARK SESSION
# =========================
def create_spark():
    try:
        SparkSession.getActiveSession().stop()
    except Exception:
        pass

    spark = SparkSession.builder \
        .appName('airbnb ADLS pipeline') \
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.hadoop:hadoop-azure:3.3.6",
                "org.apache.hadoop:hadoop-azure-datalake:3.3.6",
                "org.apache.hadoop:hadoop-common:3.3.6",
                "com.azure:azure-storage-blob:12.25.0",
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
        .option("multiLine", False) \
        .option("escape", '"') \
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
# CLEAN AIRBNB DATA + ENGINEER FEATURES
# =========================
def clean_airbnb_df(airbnb):
    """
    Cleans Airbnb listings dataset:
    - Casts latitude/longitude to double
    - Cleans price column
    - Removes nulls
    - Filters realistic price range
    - Adds engineered features (text signals, amenity flags, luxury proxy)
    """
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
        col("host_acceptance_rate"),
    )

    airbnb_df = airbnb_df.withColumn(
        "area", trim(lower(col("neighbourhood_cleansed")))
    )

    airbnb_df = airbnb_df.withColumn("accommodates", col("accommodates").cast("int")) \
                         .withColumn("bedrooms", col("bedrooms").cast("int")) \
                         .withColumn("beds", col("beds").cast("int")) \
                         .withColumn("bathrooms", col("bathrooms").cast("double"))

    airbnb_df = airbnb_df.fillna({
        "accommodates": 0,
        "bathrooms": 0,
        "bedrooms": 0,
    })

    # Clean price column
    airbnb_df = airbnb_df.withColumn(
        "price",
        regexp_replace(col("price"), "[$,]", "").cast("double")
    )

    print(f"Rows before: {airbnb_df.count()}")
    airbnb_df = airbnb_df.dropna(subset=["price"])
    print(f"Rows after dropping nulls for price: {airbnb_df.count()}")

    airbnb_df = airbnb_df.filter(
        (col("price") >= 20) & (col("price") <= 500)
    )
    print(f"Rows after selecting price range 20–500: {airbnb_df.count()}")

    # Engineered features
    airbnb_df = airbnb_df.withColumn(
        "price_per_person",
        round(
            when(
                col("accommodates") > 0,
                col("price") / (col("accommodates") + 0.1)
            ).otherwise(0),
            2
        )
    )

    airbnb_df = airbnb_df.withColumn(
        "is_central_london",
        ((col("latitude") > 51.50) & (col("latitude") < 51.52)).cast("int")
    )

    # Text signals
    airbnb_df = airbnb_df.fillna({"description": "", "amenities": ""})

    airbnb_df = airbnb_df.withColumn("desc_length", length(col("description")))

    airbnb_df = airbnb_df.withColumn(
        "has_luxury_words",
        lower(col("description")).rlike(
            "luxury|penthouse|stunning|designer|premium|view|balcony|central"
        ).cast("int")
    )

    # Amenity flags (type matters, not just count)
    airbnb_df = airbnb_df.withColumn(
        "has_balcony", lower(col("amenities")).rlike("balcony").cast("int")
    ).withColumn(
        "has_view", lower(col("amenities")).rlike("view").cast("int")
    ).withColumn(
        "has_garden", lower(col("amenities")).rlike("garden").cast("int")
    ).withColumn(
        "has_hot_tub", lower(col("amenities")).rlike("hot tub|jacuzzi").cast("int")
    )

    # Property × Room interaction
    airbnb_df = airbnb_df.withColumn(
        "property_room_combo",
        concat_ws("_", col("property_type"), col("room_type"))
    )

    # Simple luxury proxy
    airbnb_df = airbnb_df.withColumn(
        "luxury_score",
        col("bathrooms") * 0.4 +
        col("accommodates") * 0.3 +
        col("bedrooms") * 0.2
    )

    return airbnb_df


# =========================
# CALENDAR → DOMINANT SEASON
# =========================
def compute_dominant_season(calendar_df):
    calendar_df = calendar_df.withColumn("date", to_date(col("date")))
    calendar_df = calendar_df.withColumn(
        "price",
        regexp_replace(col("price"), "[$,]", "").cast("double")
    )
    calendar_df = calendar_df.withColumn("month", month(col("date")))
    calendar_df = calendar_df.withColumn(
        "season",
        when(col("month").isin([12, 1, 2]), "winter")
        .when(col("month").isin([3, 4, 5]), "spring")
        .when(col("month").isin([6, 7, 8]), "summer")
        .otherwise("autumn")
    )
    calendar_df = calendar_df.withColumn(
        "is_booked",
        when(col("available") == "f", 1).otherwise(0)
    )

    season_bookings = calendar_df.groupBy("listing_id", "season") \
        .agg(sum("is_booked").alias("booked_days"))

    window_spec = Window.partitionBy("listing_id").orderBy(col("booked_days").desc())

    return season_bookings.withColumn(
        "rank", row_number().over(window_spec)
    ).filter(col("rank") == 1) \
     .select("listing_id", col("season").alias("dominant_season"))


# =========================
# CRIME → BOROUGH AGGREGATE
# =========================
def clean_crime_df(crime_df):
    """Cleans crime dataset and aggregates crime count per borough (area)."""
    df = crime_df.withColumn(
        "area",
        trim(regexp_replace(col("LSOA name"), r"\s\d+[A-Z]$", ""))
    )
    df = df.withColumn("area", lower(trim(col("area"))))
    return df.groupBy("area").agg(count("*").alias("area_crime_count"))


# =========================
# INCOME + HOUSE PRICES
# =========================
def prepare_income_house(df):
    df = df.withColumnRenamed("Borough", "area") \
           .withColumnRenamed("Median House Price", "house_price") \
           .withColumnRenamed("Mean Household Income", "income")
    df = df.withColumn("area", trim(lower(col("area"))))
    return df.groupBy("area").agg(
        avg("house_price").alias("house_price"),
        avg("income").alias("income")
    )


# =========================
# HAVERSINE DISTANCE
# =========================
def haversine(lat1, lon1, lat2, lon2):
    a = (
        sin((radians(lat2 - lat1)) / 2) ** 2 +
        cos(radians(lat1)) * cos(radians(lat2)) *
        sin((radians(lon2 - lon1)) / 2) ** 2
    )
    return 2 * 6371 * atan2(sqrt(a), sqrt(1 - a))


# =========================
# STATIONS WITHIN 1 KM
# =========================
def compute_station_within_1km(airbnb, stations):
    stations = stations.select(
        col("latitude").alias("station_lat"),
        col("longitude").alias("station_lon")
    )
    print(f"Rows before stations filter: {airbnb.count()}")
    joined = airbnb.crossJoin(broadcast(stations)).filter(
        (abs(col("latitude") - col("station_lat")) < 0.02) &
        (abs(col("longitude") - col("station_lon")) < 0.02)
    )
    print(f"Rows after stations bbox filter: {joined.count()}")

    joined = joined.withColumn(
        "distance_km",
        haversine(col("latitude"), col("longitude"),
                  col("station_lat"), col("station_lon"))
    )

    nearby = joined.filter(col("distance_km") <= 1)
    print(f"Rows after stations dist <= 1 km filter: {nearby.count()}")

    return nearby.groupBy("id").agg(count("*").alias("stations_within_1km"))


# =========================
# TOURIST SPOTS WITHIN 1 KM
# =========================
def compute_tourism_within_1km(airbnb, tourist):
    tourist = tourist.select(
        col("latitude").alias("tourist_lat"),
        col("longitude").alias("tourist_lon")
    )

    joined = airbnb.crossJoin(broadcast(tourist)).filter(
        (abs(col("latitude") - col("tourist_lat")) < 0.02) &
        (abs(col("longitude") - col("tourist_lon")) < 0.02)
    )
    print(f"Rows after tourist bbox filter: {joined.count()}")

    joined = joined.withColumn(
        "distance_km",
        haversine(col("latitude"), col("longitude"),
                  col("tourist_lat"), col("tourist_lon"))
    )

    nearby = joined.filter(col("distance_km") <= 1)
    print(f"Rows after tourist dist <= 1 km filter: {nearby.count()}")

    return nearby.groupBy("id").agg(count("*").alias("tourist_spots_within_1km"))


# =========================
# MERGE HELPERS
# =========================
def merge_airbnb_crime(airbnb_clean, crime_area):
    merged = airbnb_clean.join(crime_area, on="area", how="left")
    return merged.fillna({"area_crime_count": 0})


def merge_season_with_airbnb(airbnb_df, season_df):
    return airbnb_df.join(
        season_df,
        airbnb_df.id == season_df.listing_id,
        "left"
    ).drop("listing_id")


# =========================
# SAVE TO ADLS
# =========================
def save_dataframe_csv(df):
    """Save Spark DataFrame to ADLS cleansed/ folder as a single CSV."""
    try:
        df.coalesce(1).write \
            .mode("overwrite") \
            .option("header", True) \
            .csv(f"{AZ_BASE_PATH}/cleansed/")
        print("✅ Cleansed data saved to ADLS (cleansed/ folder)")
    except Exception as e:
        print(f"❌ Error saving to ADLS: {e}")


# =========================
# READ + CLEAN + JOIN PIPELINE
# =========================
def read_raw_data():
    spark = create_spark()
    logger = get_logger()
    logger.info('Pipeline started...')

    # Load raw datasets
    airbnb_raw, calendar, crime, stations, tourist, income_house = load_data(spark)

    # Clean Airbnb listings
    airbnb_clean_df = clean_airbnb_df(airbnb_raw)
    print(f"Rows after airbnb clean df: {airbnb_clean_df.count()}")

    # Calendar → dominant season per listing
    calendar_agg_clean_df = compute_dominant_season(calendar)
    airbnb_calendar_df = merge_season_with_airbnb(airbnb_clean_df, calendar_agg_clean_df)

    # Crime
    crime_clean_df = clean_crime_df(crime)
    airbnb_crime_merged_df = merge_airbnb_crime(airbnb_calendar_df, crime_clean_df)
    print(f"Rows after airbnb merged with crime df: {airbnb_crime_merged_df.count()}")

    # Stations within 1 km
    stations_features = compute_station_within_1km(airbnb_clean_df, stations) \
        .select("id", "stations_within_1km")
    airbnb_crime_stations_merged_df = airbnb_crime_merged_df.join(
        stations_features, "id", "left"
    )
    print(f"Rows after airbnb-crime-stations merged df: {airbnb_crime_stations_merged_df.count()}")

    # Tourism within 1 km
    tourism_features = compute_tourism_within_1km(airbnb_clean_df, tourist) \
        .select("id", "tourist_spots_within_1km")
    airbnb_crime_stations_tourism_merged_df = airbnb_crime_stations_merged_df.join(
        tourism_features, "id", "left"
    )
    print(f"Rows after airbnb-crime-stations-tourism merged df: {airbnb_crime_stations_tourism_merged_df.count()}")

    # Income + house prices
    income_house_clean_df = prepare_income_house(income_house)
    airbnb_final_cleaned_df = airbnb_crime_stations_tourism_merged_df.join(
        income_house_clean_df, "area", "left"
    )
    print(f"Rows after airbnb merged with income/house df: {airbnb_final_cleaned_df.count()}")

    # Fill missing values and finalise types
    airbnb_final_cleaned_df = airbnb_final_cleaned_df.fillna({
        "house_price": 0,
        "income": 0,
        "stations_within_1km": 0,
        "tourist_spots_within_1km": 0,
        "review_scores_rating": 0,
    })

    airbnb_final_cleaned_df = airbnb_final_cleaned_df.withColumn(
        "house_price", round(col("house_price"), 0).cast("int")
    ).withColumn(
        "income", round(col("income"), 0).cast("int")
    ).withColumn(
        "is_central_london", col("is_central_london").cast("int")
    )

    print(f"Rows in final cleansed df: {airbnb_final_cleaned_df.count()}")

    # Build + save data dictionary
    data_dict = build_data_dictionary(airbnb_final_cleaned_df)
    print("\n=== DATA DICTIONARY ===")
    print(data_dict.to_string(index=False))
    data_dict.to_csv(os.path.join(output_path, "data_dictionary.csv"), index=False)

    # Save to ADLS
    save_dataframe_csv(airbnb_final_cleaned_df)

    return airbnb_final_cleaned_df


# =========================
# EDA STAGE
# =========================
def run_eda_stage(input_df):
    try:
        print("\n=== RUNNING EDA STAGE ===")
        eda_df = input_df.toPandas() if hasattr(input_df, "toPandas") else input_df
        eda_output_df = perform_eda(eda_df.copy())
        print("\n=== EDA COMPLETED ===")
        return eda_output_df
    except Exception as e:
        print(f"EDA stage failed: {e}")
        raise


# =========================
# RESULTS HELPERS
# =========================
def print_model_results(name, results):
    print(f"\n=== {name} ===")
    print("Best Params:", results["best_params"])
    print("Log Scale  :", results["metrics_log"])
    print("Real Scale :", results["metrics_real"])


def results_to_rows(model_name, results):
    """Flatten a pipeline result dict into one row with both scales."""
    log = results.get("metrics_log", {}) or {}
    real = results.get("metrics_real", {}) or {}
    return {
        "Model":       model_name,
        "RMSE (log)":  log.get("rmse"),
        "MAE (log)":   log.get("mae"),
        "R2 (log)":    log.get("r2"),
        "RMSE (real)": real.get("rmse"),
        "MAE (real)":  real.get("mae"),
        "R2 (real)":   real.get("r2"),
        "Best Params": results.get("best_params"),
    }


# =========================
# RUN FULL PIPELINE
# =========================
def run_full_pipeline():
    final_clean_df = read_raw_data()

    print("EDA Started")
    eda_df = run_eda_stage(final_clean_df)
    print("EDA Ended")

    print("Data Cleansing Started")
    if hasattr(eda_df, "toPandas"):
        eda_df = eda_df.toPandas()
    model_df = get_model_safe_df(eda_df)
    model_df.to_csv(os.path.join(output_path, "airbnb_model_input.csv"), index=False)
    print("Data Cleansing Ended")

    print("Neural Network Models Started")
    nn_ray_results = run_nn_pipeline(df=model_df)
    nn_model_results = run_nn(df=model_df)
    print("Neural Network Models Ended")

    print_model_results("Neural Network (Ray)", nn_ray_results)
    print_model_results("Neural Network (loky)", nn_model_results)

    rows = [
        results_to_rows("Neural Network (Ray)", nn_ray_results),
        results_to_rows("Neural Network (loky)", nn_model_results),
    ]
    results_df = pd.DataFrame(rows)
    results_df.to_csv(
        os.path.join(output_path, "NeuralNetwork_model_performance.csv"),
        index=False
    )
    print("\n=== FINAL RESULTS ===")
    print(results_df)

    return results_df


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_full_pipeline()