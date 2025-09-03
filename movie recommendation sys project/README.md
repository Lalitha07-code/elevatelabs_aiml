# Movie Recommendation System

This package contains a complete Movie Recommendation System using the MovieLens dataset.
It includes:
- `notebook.ipynb` — A runnable Jupyter notebook with data loading, preprocessing, content-based, collaborative (SVD), hybrid recommendation logic, evaluation (RMSE and Precision@K), and optional sentiment filtering notes.
- `app.py` — Streamlit application to interactively get top-N recommendations. Supports optional TMDb poster fetching if you provide a TMDb API key.
- `requirements.txt` — Python dependencies.
- `data/` — Not included due to size. Download MovieLens `ml-latest-small` and place `movies.csv` and `ratings.csv` into `data/` before running.


## How to run

1. Create a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download MovieLens `ml-latest-small` from https://grouplens.org/datasets/movielens/latest/ and put `movies.csv` and `ratings.csv` inside a `data/` folder in the project root.

4. Run the notebook with Jupyter, or run the Streamlit app:

```bash
streamlit run app.py
```

## TMDb posters (optional)
To show movie posters in the Streamlit UI, obtain a TMDb API key (https://www.themoviedb.org/) and either set it as an environment variable `TMDB_API_KEY` or paste it into the app when prompted.

## Notes
- This package uses a simple TF-IDF on genres for content-based filtering and TruncatedSVD on the user-item matrix for collaborative filtering.
- For production, consider using more features (plot summaries, tags, cast/director), and stronger MF solvers (implicit, alternating least squares), and caching poster images.
