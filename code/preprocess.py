# -*- coding: utf-8 -*-
"""
YouTube Trending Data Preprocessing (Enhanced)
- Convert video_duration to seconds
- Add channel authority measures (log(subscriber_count), channel age)
- Safe datetime and numeric processing
- Chunk processing for memory efficiency
- Video-level aggregation + channel-level features
- Save as Parquet for fast subsequent analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------------------
# 1. Paths
# -------------------------------

csv_path = r"D:/homework/youtube/youtube_trending_videos_global.csv"
processed_path = r"D:/homework/youtube/youtube_trending_processed_enhanced.parquet"

# -------------------------------
# 2. Load preprocessed if exists
# -------------------------------

if os.path.exists(processed_path):
    print("Loaded processed data:", processed_path)
    df = pd.read_parquet(processed_path)
    print("Data shape:", df.shape)
else:
    print("Processed file not found. Start processing from CSV...")

    # -------------------------------
    # 3. Columns to use
    # -------------------------------
    cols_to_use = [
        "video_id",
        "channel_id",
        "video_published_at",
        "video_trending__date",
        "video_trending_country",
        "video_category_id",
        "video_view_count",
        "video_like_count",
        "video_comment_count",
        "video_duration",
        "channel_subscriber_count",
        "channel_published_at"
    ]

    rename_dict = {
        "video_published_at": "published_at",
        "video_trending__date": "collection_date",
        "video_trending_country": "region_code",
        "video_category_id": "category_id",
        "video_view_count": "view_count",
        "video_like_count": "like_count",
        "video_comment_count": "comment_count",
        "channel_subscriber_count": "subscriber_count",
        "channel_published_at": "channel_published_at"
    }

    # -------------------------------
    # 4. Chunk reading
    # -------------------------------
    chunksize = 100_000  # approx ~100MB per chunk
    chunks = []

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=cols_to_use, chunksize=chunksize)):
        print(f"Processing chunk {i+1}")

        # Rename columns
        chunk = chunk.rename(columns=rename_dict)

        # -------------------------------
        # 5. Safe datetime processing
        # -------------------------------
        for col in ["published_at", "collection_date", "channel_published_at"]:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=["published_at","collection_date","channel_published_at"])
        chunk["published_at"] = chunk["published_at"].dt.tz_localize(None)
        chunk["collection_date"] = chunk["collection_date"].dt.tz_localize(None)
        chunk["channel_published_at"] = chunk["channel_published_at"].dt.tz_localize(None)

        # -------------------------------
        # 6. Safe numeric conversion
        # -------------------------------
        for col in ["view_count","comment_count","like_count","subscriber_count"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=["view_count","comment_count","like_count","subscriber_count"])

        # -------------------------------
        # 7. Trending lag
        # -------------------------------
        chunk["trending_lag_hours"] = (chunk["collection_date"] - chunk["published_at"]) / np.timedelta64(1,"h")
        chunk = chunk[chunk["trending_lag_hours"] >= 0]

        # -------------------------------
        # 8. Parse ISO 8601 video_duration to seconds
        # -------------------------------
        def iso8601_to_seconds(duration):
            import re
            pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
            match = pattern.fullmatch(str(duration))
            if not match:
                return np.nan
            h, m, s = match.groups()
            h, m, s = int(h or 0), int(m or 0), int(s or 0)
            return h*3600 + m*60 + s
        chunk["video_duration_sec"] = chunk["video_duration"].apply(iso8601_to_seconds)
        chunk = chunk.dropna(subset=["video_duration_sec"])

        # -------------------------------
        # 9. Construct engagement ratios
        # -------------------------------
        chunk["comment_to_view"] = chunk["comment_count"] / (chunk["view_count"] + 1)
        chunk["like_to_view"] = chunk["like_count"] / (chunk["view_count"] + 1)

        # Log transformations
        chunk["log_views"] = np.log1p(chunk["view_count"])
        chunk["log_comments"] = np.log1p(chunk["comment_count"])
        chunk["log_subscriber_count"] = np.log1p(chunk["subscriber_count"])

        # Channel age in years
        chunk["channel_age_years"] = (chunk["collection_date"] - chunk["channel_published_at"]).dt.total_seconds() / (365*24*3600)

        # -------------------------------
        # 10. Video-level aggregation (first trending)
        # -------------------------------
        chunk = chunk.sort_values("collection_date").groupby("video_id").first().reset_index()

        chunks.append(chunk)

    # -------------------------------
    # 11. Merge chunks
    # -------------------------------
    df = pd.concat(chunks, ignore_index=True)
    print("Merged data shape:", df.shape)

    # -------------------------------
    # 12. Channel-level aggregation
    # -------------------------------
    channel_agg = df.groupby("channel_id").agg(
        channel_video_count=("video_id","count"),
        channel_total_views=("view_count","sum"),
        channel_avg_views=("view_count","mean")
    ).reset_index()
    df = df.merge(channel_agg, on="channel_id", how="left")

    # -------------------------------
    # 13. Drop remaining missing values
    # -------------------------------
    df = df.dropna(subset=["view_count","comment_count","trending_lag_hours","video_duration_sec"])

    # -------------------------------
    # 14. Save as Parquet
    # -------------------------------
    df.to_parquet(processed_path, index=False)
    print("Processed data saved as Parquet:", processed_path)

# -------------------------------
# 15. Data check
# -------------------------------
print("Preview of processed data:")
print(df.head())
print("Columns:", df.columns.tolist())
print("Data shape:", df.shape)