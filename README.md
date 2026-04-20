YouTube Trending Visibility Analysis

This project investigates the determinants of trending visibility on algorithmically mediated platforms using large-scale YouTube trending data across multiple countries.

The core objective is to understand what factors influence how quickly a video appears on trending lists, with a focus on engagement, channel authority, content characteristics, and platform dynamics.

--------------------------------------------------

KEY FEATURES

1. Data Processing Pipeline
- Large-scale CSV ingestion with chunk processing for memory efficiency
- Robust handling of:
  - missing values
  - datetime inconsistencies
  - numeric conversion
- Feature engineering:
  - Trending lag (hours from publish → trending)
  - Engagement ratios (like_to_view, comment_to_view)
  - Log transformations (views, comments, subscribers)
  - Video duration parsing (ISO 8601 → seconds)
  - Channel age & authority metrics
- Outputs optimized dataset in Parquet format for fast analysis

File: code/preprocess.py

--------------------------------------------------

2. Statistical & Econometric Analysis
- Correlation analysis across key features
- Multiple linear regression:
  - Controls for country (region_code) and category
- Diagnostic tests:
  - Multicollinearity (VIF)
  - Heteroskedasticity (Breusch–Pagan test)
  - Residual & QQ plots
- Partial regression (marginal effects)

File: code/analysis.py

--------------------------------------------------

3. Cross-Country & Content Analysis
- Trending lag comparison across countries
- Category-level performance differences
- Cross-country engagement intensity
- Category dominance by country

--------------------------------------------------

4. Channel-Level Market Structure Analysis
- Channel size segmentation (Small → Top)
- Concentration analysis:
  - Lorenz-style concentration curves
  - Top 1% / Top 5% dominance
- Market concentration metrics:
  - Gini coefficient
  - Herfindahl–Hirschman Index (HHI)

--------------------------------------------------

5. Interactive Dashboard (Streamlit)
- Fully interactive UI with:
  - country & category filters
  - engagement metric selection
  - channel size filtering
  - numeric sliders (lag, duration, subscribers)
- Visualizations include:
  - distributions
  - boxplots
  - interaction effects
  - partial effects
  - concentration curves
  - Gini/HHI by country

File: code/streamlit_dashboard.py

--------------------------------------------------

KEY INSIGHTS

- How engagement impacts speed of trending visibility
- Whether larger channels dominate trending systems
- Cross-country differences in:
  - algorithm behavior
  - user engagement patterns
- Structural inequality in content exposure (via Gini/HHI)

--------------------------------------------------

TECH STACK

- Python
- pandas / numpy
- matplotlib / seaborn
- statsmodels
- scipy
- Streamlit

--------------------------------------------------

PROJECT STRUCTURE

YoutubeTrendingVisibility/
│
├── code/
│   ├── analysis.py
│   ├── preprocess.py
│   ├── streamlit_dashboard.py
│   └── *.png (visual outputs)
│
├── report.doc
├── .gitignore
└── README

--------------------------------------------------

HOW TO RUN

1. Preprocess data
python code/preprocess.py

2. Run analysis
python code/analysis.py

3. Launch dashboard
streamlit run code/streamlit_dashboard.py

--------------------------------------------------

AUTHOR

Tingyu Zhang