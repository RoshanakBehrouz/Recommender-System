

###
# 🎬 Movie Recommender System (Weighted Matrix Factorization)

This project implements a movie recommender system using **Weighted Matrix Factorization (WMF)** with the **MovieLens 100k** dataset. The model is trained using the **Weighted Alternating Least Squares (WALS)** optimization algorithm. The application is built with **Streamlit**, providing an interactive interface for both existing and new users to receive movie recommendations.

---

## ✨ Features

- **Weighted Matrix Factorization (WMF):** Learns user and movie embeddings, accounting for both observed (rated) and unobserved (unrated) movies.  
- **Weighted Alternating Least Squares (WALS):** Efficiently optimizes user and item factors.  
- **MovieLens 100k Dataset:** Automatically downloads and uses the MovieLens 100k dataset.  
- **Existing User Recommendations:** Personalized suggestions based on historical data.  
- **New User (Cold Start) Recommendations:** Infers preferences using a few initial ratings.  
- **Interactive Streamlit UI:** User-friendly web interface.  
- **Learning Curve Visualization:** Tracks training and validation RMSE over iterations.


## 🚀 How to Run

### Prerequisites

- Python 3.7+  
- pip

### Installation

```bash
pip install streamlit pandas numpy requests matplotlib seaborn scikit-learn
````

### Run the App

```bash
streamlit run IR-RS.py
```

Then go to [http://localhost:8501](http://localhost:8501)

---

## 📊 Model Details

The recommender system is implemented in the `WeightedMatrixFactorization` class using WMF.

### Objective Function

WMF minimizes the following objective:

<pre> ``` ∑(i,j)∈Obs (rᵢⱼ - uᵢᵗ vⱼ)² + w₀ ∑(i,j)∈Nobs (uᵢᵗ vⱼ)² + λ ( ∑ᵢ ||uᵢ||² + ∑ⱼ ||vⱼ||² ) ``` </pre>



Where:

* $r_{ij}$: rating of user $i$ for movie $j$ ($C_{ij}$- bc of confidence interval I renamed this to r)
* $\mathbf{u}_i$: latent factor for user $i$
* $\mathbf{v}_j$: latent factor for movie $j$
* $\text{Obs}$: observed (rated) pairs
* $\text{Nobs}$: unobserved (unrated) pairs
* $w_0$: weight for unobserved pairs
* $\lambda$: regularization parameter

### Training with WALS

1. **Fix Item Factors $\mathbf{V}$**
   Solve for user vectors $\mathbf{u}_i$

2. **Fix User Factors $\mathbf{U}$**
   Solve for item vectors $\mathbf{v}_j$

Repeat until convergence or max iterations.

---

## 📁 Project Structure

* `IR-RS.py`: Main script including:

  * Streamlit UI
  * Data loading
  * WMF model
  * Recommendation logic




