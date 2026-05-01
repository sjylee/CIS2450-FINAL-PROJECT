# The Sonic Paradox Longevity Index
### CIS 2450: Big Data Analytics — Final Project

## Research Question
Does emotional incongruence between music and lyrics (Paradox Score) predict
Spotify popularity after controlling for genre, release year, and audio features?

## Dataset
- **Spotify:** 1,159,764 tracks with audio features and popularity scores
- **Genius:** 5,134,856 song lyrics (English only: 3,374,071 retained)
- **Joined:** 288,819 songs matched on normalized title and artist name

## Engineered Feature
**Paradox Score** = |Spotify valence - VADER lyric sentiment|
- 0 = music and lyrics emotionally aligned
- 1 = maximally incongruent (the "sad-bop" effect)

## Models
| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge Regression | 1.2610 | 0.3629 |
| Random Forest | 1.2179 | 0.4056 |
| Gradient Boosting (best) | 1.2136 | 0.4098 |

## Key Finding
Paradox Score IS a statistically significant positive predictor of popularity
(OLS p=0.0000, Bootstrap 95% CI=[0.0374, 0.0489], excludes zero). However,
the effect is practically negligible, ranking 14th to 17th of 22 features.
Dominant predictors are release year, lyric density, and genre.

## How to Run

**Notebook:**
```bash
jupyter notebook final-notebook.ipynb
```

**Dashboard:**
```bash
pip install dash plotly pandas numpy scikit-learn scipy
python dashboard.py
```
Then open `http://localhost:8050` in your browser.

## Requirements
```bash
pip install polars pandas numpy matplotlib seaborn scikit-learn
pip install vaderSentiment scipy statsmodels dash plotly
```

## File Structure
├── final-notebook.ipynb    # Full analysis pipeline
├── dashboard.py            # Interactive Dash dashboard
├── data/
│   ├── spotify_raw.csv     # Spotify audio features (not tracked in git)
│   └── genius_lyrics_raw.csv  # Genius lyrics (not tracked in git)
└── outputs/
└── model_results.csv   # Model performance summary
Team Members: Stella Lee, Yuetong Zheng
