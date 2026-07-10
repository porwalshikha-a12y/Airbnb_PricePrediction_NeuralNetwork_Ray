"""
This is Post Data cleansing and imputation EDA done for
the proper, presentable analysis: target distribution,
correlation heatmap, price-vs-feature plots, geographic maps,
clustering
"""

import warnings
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

TARGET = "price"

# EDIT THESE TWO PATHS TO MATCH YOUR MACHINE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Data", "Cleansed/airbnb_model_input.csv")

OUT_DIR = os.path.join(BASE_DIR, "Output", "NormalEDA")

# Free-text + duplicate columns we never want in plots
DROP_COLS = {
    "id", "name", "description", "amenities",
    "neighbourhood_cleansed",        # duplicate of `area`
    "property_room_combo",           # interaction; redundant with property_type+room_type
    "has_description",               # near-constant flag
}

# Amenity / signal flags (binary) we want to compare against price
FLAG_COLS = [
    "has_luxury_words", "has_balcony", "has_view",
    "has_garden", "has_hot_tub", "host_is_superhost",
    "is_central_london",
]

# Numeric features to correlate against price
NUMERIC_FEATURES = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "review_scores_rating", "number_of_reviews", "reviews_per_month",
    "availability_30", "availability_365",
    "host_response_rate", "host_acceptance_rate",
    "luxury_score", "stations_within_1km", "tourist_spots_within_1km",
    "area_crime_count", "house_price", "income",
    "is_central_london",
]


# Helpers

def get_project_output_path(filename=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output/NormalEDA")
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        return os.path.join(output_dir, filename)

    return output_dir

def safe(s) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def save(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")



# 1. Summary + nulls

def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    lines.append("=== DTYPES ===")
    lines.append(df.dtypes.astype(str).to_string())
    lines.append("\n=== NUMERIC SUMMARY ===")
    lines.append(df.describe(include="number").T.to_string())
    lines.append("\n=== NULL COUNTS ===")
    nulls = df.isna().sum().sort_values(ascending=False)
    null_df = pd.DataFrame({"nulls": nulls, "pct": (nulls / len(df) * 100).round(2)})
    lines.append(null_df.to_string())
    lines.append(f"\nDuplicate rows: {df.duplicated().sum()}")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print("  wrote summary.txt")


def plot_nulls(df: pd.DataFrame, out_dir: Path) -> None:
    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        print("  no nulls — skipping null plots")
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(null_pct))))
    sns.barplot(x=null_pct.values, y=null_pct.index, ax=ax, color="steelblue")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values per column")
    save(fig, out_dir, "01_nulls")



# 2. Price distribution (target)

def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET], bins=60, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Distribution of {TARGET} (raw)")
    axes[0].set_xlabel("Nightly price (£)")

    sns.histplot(np.log1p(df[TARGET]), bins=60, kde=True,
                 ax=axes[1], color="seagreen")
    axes[1].set_title(f"Distribution of log1p({TARGET})")
    axes[1].set_xlabel("log1p(price)")
    save(fig, out_dir, "02_price_distribution")

    # Dedicated boxplot — outliers visible (note showfliers=True is the default)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6),
                             gridspec_kw={"height_ratios": [1, 1]})

    sns.boxplot(x=df[TARGET], ax=axes[0], color="steelblue")
    axes[0].set_title(f"{TARGET} boxplot (outliers shown)")
    axes[0].set_xlabel("Nightly price (£)")

    # Inter-quartile and whisker stats annotated
    q1, med, q3 = df[TARGET].quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    upper_whisker = q3 + 1.5 * iqr
    lower_whisker = max(0, q1 - 1.5 * iqr)
    n_outliers_high = int((df[TARGET] > upper_whisker).sum())
    n_outliers_low  = int((df[TARGET] < lower_whisker).sum())
    stats_text = (
        f"Q1={q1:.0f}   median={med:.0f}   Q3={q3:.0f}   IQR={iqr:.0f}\n"
        f"whiskers: [{lower_whisker:.0f}, {upper_whisker:.0f}]\n"
        f"outliers below: {n_outliers_low:,}    "
        f"outliers above: {n_outliers_high:,}    "
        f"({(n_outliers_low + n_outliers_high) / len(df) * 100:.2f}% of rows)"
    )
    axes[0].text(
        0.99, 0.95, stats_text, transform=axes[0].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgray", alpha=0.85),
    )

    # Violin plot underneath shows the full density shape alongside the box
    sns.violinplot(x=df[TARGET], ax=axes[1], color="lightsteelblue",
                   inner="quartile", cut=0)
    axes[1].set_title(f"{TARGET} violin (density + quartiles)")
    axes[1].set_xlabel("Nightly price (£)")

    save(fig, out_dir, "02b_price_boxplot_outliers")



# 3. Price by categorical features

def plot_price_by_category(
    df: pd.DataFrame, col: str, out_dir: Path,
    top_n: int = 15, order_by: str = "median",
) -> None:
    if col not in df.columns:
        return
    sub = df[[col, TARGET]].dropna()
    sub[col] = sub[col].astype(str)

    # Order categories by chosen statistic of price
    if order_by == "median":
        order = (sub.groupby(col)[TARGET].median()
                    .sort_values(ascending=False).head(top_n).index)
    elif order_by == "count":
        order = sub[col].value_counts().head(top_n).index
    else:
        order = sorted(sub[col].unique())[:top_n]

    sub = sub[sub[col].isin(order)]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(order))))
    sns.boxplot(data=sub, x=TARGET, y=col, order=list(order),
                ax=ax, palette="viridis", showfliers=False)
    ax.set_title(
        f"{TARGET} by {col} "
        f"(top {len(order)} of {df[col].nunique()} by {order_by})"
    )
    ax.set_xlabel("Nightly price (£)")
    save(fig, out_dir, f"03_price_by_{safe(col)}")



# 4. Price vs binary flags — single grouped bar plot

def plot_flag_impact(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for f in FLAG_COLS:
        if f not in df.columns:
            continue
        for v in [0, 1]:
            sub = df[df[f] == v][TARGET]
            if len(sub) > 0:
                rows.append({"flag": f, "value": v, "median_price": sub.median(),
                             "mean_price": sub.mean(), "n": len(sub)})
    if not rows:
        return
    flag_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=flag_df, x="flag", y="median_price",
                hue="value", palette={0: "lightgray", 1: "steelblue"}, ax=ax)
    ax.set_title("Median nightly price by binary feature")
    ax.set_ylabel("Median price (£)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="flag value", loc="upper left")
    save(fig, out_dir, "04_price_by_binary_flags")



# 5. Price vs numeric features — scatter / hexbin grid

def plot_price_vs_numeric(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns and c != TARGET]
    if not cols:
        return

    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for ax, c in zip(axes, cols):
        sub = df[[c, TARGET]].dropna()
        if sub[c].nunique() <= 6:
            # discrete -> boxplot
            sns.boxplot(data=sub, x=c, y=TARGET, ax=ax,
                        palette="crest", showfliers=False)
        else:
            ax.hexbin(sub[c], sub[TARGET], gridsize=30, cmap="Blues",
                      mincnt=1)
        ax.set_title(f"{TARGET} vs {c}")
        ax.set_xlabel(c)
        ax.set_ylabel(TARGET)

    # hide empty axes
    for ax in axes[len(cols):]:
        ax.axis("off")

    save(fig, out_dir, "05_price_vs_numeric")



# 6. Correlation heatmap (numeric only)

def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in [TARGET] + NUMERIC_FEATURES if c in df.columns]
    cols = list(dict.fromkeys(cols))  # de-dupe, preserve order
    if len(cols) < 3:
        return
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(min(13, 1 + len(cols)),
                                    min(11, 1 + len(cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.4, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Pearson correlation (numeric features + price)")
    save(fig, out_dir, "06_correlation_heatmap")

    # Top correlates with price as a separate, easy-to-read bar chart
    if TARGET in corr.columns:
        s = corr[TARGET].drop(TARGET).sort_values(key=lambda x: x.abs(),
                                                  ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(s))))
        sns.barplot(x=s.values, y=s.index, ax=ax,
                    palette=["steelblue" if v >= 0 else "indianred"
                             for v in s.values])
        ax.set_xlabel(f"Pearson r vs {TARGET}")
        ax.set_title(f"Numeric features ranked by |corr| with {TARGET}")
        ax.axvline(0, color="black", linewidth=0.6)
        save(fig, out_dir, "07_top_correlates_with_price")



# 7. Geographic price map

def plot_geographic_map(df: pd.DataFrame, out_dir: Path) -> None:
    if not {"latitude", "longitude", TARGET}.issubset(df.columns):
        return
    sub = df[["latitude", "longitude", TARGET]].dropna()
    if sub.empty:
        return

    # Cap colour scale at the 95th percentile so a few outliers don't wash
    # out the rest of the map.
    vmax = float(sub[TARGET].quantile(0.95))

    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(sub["longitude"], sub["latitude"],
                    c=sub[TARGET].clip(upper=vmax),
                    s=4, alpha=0.5, cmap="viridis")
    plt.colorbar(sc, ax=ax, label=f"{TARGET} (£, capped at p95)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Geographic distribution of London Airbnb {TARGET}")
    ax.set_aspect("equal", adjustable="datalim")
    save(fig, out_dir, "08_geographic_price_map")



# 8. Listings count by area + season

def plot_counts(df: pd.DataFrame, out_dir: Path) -> None:
    if "area" in df.columns:
        counts = df["area"].astype(str).value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(counts))))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="seagreen")
        ax.set_title("Top 20 boroughs by listing count")
        ax.set_xlabel("number of listings")
        save(fig, out_dir, "09_listings_per_area")

    if "dominant_season" in df.columns:
        order = ["winter", "spring", "summer", "autumn"]
        counts = (df["dominant_season"].astype(str).value_counts()
                  .reindex(order).dropna())
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="crest")
        ax.set_title("Listings by dominant booking season")
        ax.set_ylabel("count")
        save(fig, out_dir, "10_listings_per_season")



# 9. Listing-level review-vs-availability scatter

def plot_reviews_vs_availability(df: pd.DataFrame, out_dir: Path) -> None:
    if not {"reviews_per_month", "availability_365",
            TARGET}.issubset(df.columns):
        return
    sub = df[["reviews_per_month", "availability_365", TARGET]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(sub["availability_365"], sub["reviews_per_month"],
                    c=sub[TARGET].clip(upper=sub[TARGET].quantile(0.95)),
                    s=8, alpha=0.6, cmap="viridis")
    plt.colorbar(sc, ax=ax, label=f"{TARGET} (£, capped at p95)")
    ax.set_xlabel("availability over next 365 days")
    ax.set_ylabel("reviews per month")
    ax.set_title("Demand signals: availability vs review velocity, coloured by price")
    save(fig, out_dir, "11_availability_vs_reviews")



# Main

def main() -> int:
    csv_path = Path(CSV_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {csv_path.resolve()} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop free-text and duplicate columns up front
    drop = [c for c in DROP_COLS if c in df.columns]
    if drop:
        print(f"Dropping non-feature columns: {drop}")
        df = df.drop(columns=drop)

    print("\nWriting summary + null analysis ...")
    write_summary(df, out_dir)
    plot_nulls(df, out_dir)

    print("\nPlotting price distribution ...")
    plot_price_distribution(df, out_dir)

    print("\nPlotting price by categorical features ...")
    plot_price_by_category(df, "room_type",        out_dir, top_n=10)
    plot_price_by_category(df, "property_type",    out_dir, top_n=10)
    plot_price_by_category(df, "area",             out_dir, top_n=20)
    plot_price_by_category(df, "dominant_season",  out_dir, top_n=4,
                           order_by="alpha")

    print("\nPlotting binary-flag impact ...")
    plot_flag_impact(df, out_dir)

    print("\nPlotting price vs numeric features ...")
    plot_price_vs_numeric(df, out_dir)

    print("\nPlotting correlation matrix ...")
    plot_correlation(df, out_dir)

    print("\nPlotting geographic price map ...")
    plot_geographic_map(df, out_dir)

    print("\nPlotting listing counts by area + season ...")
    plot_counts(df, out_dir)

    print("\nPlotting reviews vs availability ...")
    plot_reviews_vs_availability(df, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
