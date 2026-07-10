
# AIRBNB DATA ANALYSIS + CLUSTERING PIPELINE


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from logger import get_logger

logger = get_logger()

def get_project_output_path(filename=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        return os.path.join(output_dir, filename)

    return output_dir

def save_dataframe_csv(df, filename="airbnb_output"):
    """
    Saves Spark DataFrame as CSV inside project output folder.
    """
    try:
        output_path = get_project_output_path(filename)

        df.coalesce(1).write \
            .mode("overwrite") \
            .option("header", True) \
            .csv(output_path)

        print(f"Data saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error saving CSV: {e}")


# =========================
# LOAD DATA
# =========================
def load_data(file_path):
    try:
        print("\n=== LOADING DATA ===")

        df = pd.read_csv(
            file_path,
            engine="python",
            quotechar='"',
            escapechar='\\',
            on_bad_lines='skip'
        )

        logger.info("Data loaded successfully")

        print("\n=== BASIC INFO ===")
        print(df.info())

        print("\n=== SHAPE ===")
        print(df.shape)

        print("\n=== NA BEFORE CLEANING ===")
        print(df.isna().sum())

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """
    Pure data cleansing — type conversion, null handling, value normalisation,
    de-duplication. Safe to pass to models without target leakage.
    """
    try:
        print("\n=== CLEANING DATA ===")

        # Description: fill nulls and add a presence flag
        df["description"] = df["description"].fillna("no description provided")
        df["has_description"] = (df["description"] != "no description provided").astype(int)

        # Numeric coercion
        numeric_cols = ["accommodates", "bedrooms", "beds", "bathrooms"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Median fill for capacity columns
        df["bedrooms"]  = df["bedrooms"].fillna(df["bedrooms"].median())
        df["beds"]      = df["beds"].fillna(df["beds"].median())
        df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

        # Review and host fields
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
        df["host_is_superhost"] = df["host_is_superhost"].fillna(
            df["host_is_superhost"].mode()[0]
        )

        # Strip '%' off rate fields and fill with median
        for col in ["host_response_rate", "host_acceptance_rate"]:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        # Boolean encoding
        df["host_is_superhost"] = df["host_is_superhost"].replace({"t": 1, "f": 0})

        # De-duplicate
        df = df.drop_duplicates()

        logger.info("Data cleaned successfully")

        print("\n=== NA AFTER CLEANING ===")
        print(df.isna().sum())

        return df

    except Exception as e:
        logger.error(f"clean_data failed: {e}")
        raise


def feature_engineering(df):
    """
    Derived / engineered features. Kept separate from clean_data so callers
    can choose whether to apply them — useful for ablation studies and to
    avoid passing price-derived features to a price-prediction model
    (data leakage / overfitting risk).

    NOTE: price_per_guest, price_per_bedroom, price_per_bathroom and
    value_score are all derived from `price`. Do NOT include them as
    inputs when predicting price; use them only for EDA / diagnostics.
    """
    try:
        print("\n=== FEATURE ENGINEERING ===")

        # Target transform (safe — the model trains on log_price as the target)
        df["log_price"] = np.log1p(df["price"])

        # Price-derived diagnostics — EDA only, NOT model inputs
        df["price_per_guest"]    = df["price"] / (df.get("accommodates", 1) + 1)
        df["price_per_bedroom"]  = df["price"] / (df.get("bedrooms", 1) + 1)
        df["price_per_bathroom"] = df["price"] / (df.get("bathrooms", 1) + 1)
        df["value_score"]        = df["review_scores_rating"] / (df["price"] + 1)

        # Location composite — safe to use as a model feature
        df["central_score"] = (
            df.get("stations_within_1km", 0) * 0.5 +
            df.get("tourist_spots_within_1km", 0) * 0.5
        )

        logger.info("Feature engineering completed successfully")
        return df

    except Exception as e:
        logger.error(f"feature_engineering failed: {e}")
        raise

# =========================
# NUMERIC DATA
# =========================
def get_numeric_data(df):
    num_df = df.select_dtypes(include=['int64', 'float64'])
    num_df = num_df.drop(columns=["id"], errors="ignore")
    return num_df


# =========================
# CORRELATION
# =========================
def plot_correlation(num_df):
    # Drop constant / near-constant columns
    num_df = num_df.loc[:, num_df.nunique() > 1]

    plt.figure(figsize=(20, 16))

    corr = num_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": 6}
    )

    plt.title("Correlation Heatmap", fontsize=16)

    path = get_project_output_path("correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# =========================
# SCALING
# =========================
def scale_data(num_df):
    scaler = StandardScaler()
    return scaler.fit_transform(num_df)


# =========================
# PCA
# =========================
def perform_pca(scaled_data):
    pca = PCA()
    pca.fit(scaled_data)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\nExplained Variance Ratio:")
    for i, var in enumerate(explained[:5]):
        print(f"  PC{i+1}: {var:.2%}")

    print("\nCumulative Variance:")
    for i, cum in enumerate(cumulative[:5]):
        print(f"  PC1-{i+1}: {cum:.2%}")

    # Scree Plot with labels
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained) + 1), explained, marker='o', linewidth=2, color='steelblue')
    plt.title("Scree Plot: Explained Variance by Principal Component", fontsize=14)
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Explained Variance Ratio", fontsize=12)
    plt.xticks(range(1, len(explained) + 1))
    plt.grid(True, alpha=0.3)
    plt.savefig(get_project_output_path("scree_plot.png"), bbox_inches="tight")
    plt.close()

    return pca, explained



# =========================
# SEGMENT DISTRIBUTION
# =========================
def plot_segments(df):
    plt.figure(figsize=(10, 6))

    ax = sns.countplot(x=df["segment"], order=df["segment"].value_counts().index)

    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.title("Segment Distribution", fontsize=14)
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.xticks(rotation=15)

    path = get_project_output_path("segment_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# =========================
# FIND K
# =========================
def find_optimal_k(scaled_data):
    inertia = []
    silhouette_scores = []
    K_range = range(2, 8)

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)

        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', linewidth=2, color='steelblue')
    plt.title("Elbow Method: Inertia vs Number of Clusters", fontsize=14)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.savefig(get_project_output_path("elbow.png"), bbox_inches="tight")
    plt.close()

    # Silhouette Plot with labels
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', linewidth=2, color='darkorange')
    plt.title("Silhouette Score vs Number of Clusters", fontsize=14)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.savefig(get_project_output_path("silhouette.png"), bbox_inches="tight")
    plt.close()

    best_k = K_range[np.argmax(silhouette_scores)]

    print(f"\nSilhouette suggests K = {best_k}")
    print("Using K = 3 for business insight")

    return 3


# =========================
# KMEANS
# =========================
def run_kmeans(scaled_data, num_df, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled_data)

    num_df["cluster"] = clusters
    return num_df, clusters


# =========================
# SUMMARY
# =========================
def get_cluster_summary(num_df):
    summary = num_df.groupby("cluster").mean(numeric_only=True)

    print("\n=== Cluster Summary ===")
    print(summary)

    return summary


# =========================
# LABEL CLUSTERS
# =========================
def label_clusters(summary, num_df):

    def safe_get(row, col):
        return row[col] if col in row.index else 0

    def safe_median(df, col):
        return df[col].median() if col in df.columns else 0

    def label(row):
        if safe_get(row, "central_score") > safe_median(summary, "central_score"):
            return "Premium Central"
        elif safe_get(row, "value_score") > safe_median(summary, "value_score"):
            return "High Value"
        elif safe_get(row, "accommodates") > safe_median(summary, "accommodates"):
            return "Large Group Premium"
        else:
            return "Mid-Range"

    summary["segment"] = summary.apply(label, axis=1)

    print("\n=== Cluster Labels ===")
    cols = [c for c in ["log_price", "value_score", "accommodates", "central_score"] if c in summary.columns]
    print(summary[cols + ["segment"]])

    num_df["segment"] = num_df["cluster"].map(summary["segment"])

    return num_df


# =========================
# PCA CLUSTER PLOT
# =========================
def plot_pca_clusters(scaled_data, clusters, segment_labels):
    """
    Parameters:
        scaled_data: Standardized feature matrix
        clusters: Array of cluster assignments (0, 1, 2)
        segment_labels: Dict mapping cluster number to segment name
                        e.g., {0: "High Value", 1: "Mid-Range", 2: "Premium Central"}
    """
    pca = PCA(n_components=2)
    data = pca.fit_transform(scaled_data)

    df_plot = pd.DataFrame(data, columns=["PC1", "PC2"])
    df_plot["cluster"] = clusters
    df_plot["segment"] = df_plot["cluster"].map(segment_labels)

    # Define colors for each segment
    colors = {
        "Premium Central": "#2ecc71",
        "High Value": "#3498db",
        "Mid-Range": "#e74c3c",
        "Large Group Premium": "#9b59b6"
    }

    plt.figure(figsize=(10, 7))

    for segment in df_plot["segment"].unique():
        subset = df_plot[df_plot["segment"] == segment]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=segment,
            s=15,
            alpha=0.6,
            color=colors.get(segment, "gray")
        )

    plt.title("Airbnb Listing Segments (PCA Projection)", fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    plt.legend(title="Segment", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)

    path = get_project_output_path("cluster_plot.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

# =========================
# SAVE OUTPUT
# =========================
def save_output_file(df):

    path = get_project_output_path("airbnb_segmented.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved final dataset: {path}")


# =========================
# PIPELINE FUNCTION
# =========================
def perform_eda(df):
    """
    Run EDA + unsupervised analysis (PCA + KMeans) and return a
    cleansing-only DataFrame for downstream modeling.

    The segment / cluster labels are computed for EDA visuals only and
    are deliberately NOT attached to the returned DataFrame, to keep the
    model inputs free of target-leaking derived columns.
    """
    print("\n=== RUNNING EDA ===")

    # 0. Spark -> pandas if needed (don't mutate the caller's frame)
    if hasattr(df, "toPandas"):
        df = df.toPandas()

    # 1. Cleansing only — this is what we return for modeling
    clean_df = clean_data(df.copy())

    # 2. Engineered features on a SEPARATE frame
    eda_df = feature_engineering(clean_df.copy())

    # 3. Numeric matrix for clustering
    num_df = get_numeric_data(eda_df.copy())

    # Fix handle NaNs BEFORE scaling/PCA
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    print("\nColumns used for clustering:")
    print(num_df.columns.tolist())

    # 4. Correlation + PCA + KMeans
    plot_correlation(num_df)
    scaled_data = scale_data(num_df)
    perform_pca(scaled_data)

    k = find_optimal_k(scaled_data)
    num_df, clusters = run_kmeans(scaled_data, num_df, k)

    summary = get_cluster_summary(num_df)
    num_df = label_clusters(summary, num_df)
    # NOTE: label_clusters already turns 'segment' into a readable label
    # string (e.g. "Premium Central"). Do NOT cast to int.

    # Create segment label mapping from summary
    segment_labels = summary["segment"].to_dict()

    # 5. Side frame for plotting / saving — keeps clean_df pure
    eda_with_segment = eda_df.copy()
    eda_with_segment["segment"] = num_df["segment"]   # index-aligned

    # 6. EDA outputs (use the side frame, NOT clean_df)
    plot_pca_clusters(scaled_data, clusters, segment_labels)
    plot_segments(eda_with_segment)
    save_output_file(eda_with_segment)

    print("\n=== EDA COMPLETE ===")

    # 7. Return cleansing-only frame for the modeling pipeline.
    return clean_df