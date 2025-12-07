# Collaborative Filtering Recommender System

This project implements a movie recommendation system (Movie Recommender System) using the **Collaborative Filtering** technique based on the **Matrix Factorization** method.

The project includes two main approaches to solve the optimization problem:

1. **Handmade implementation (Scratch):** Using NumPy to calculate pure Gradient Descent (for deep mathematical understanding).

2. **Modern implementation (Framework):** Using TensorFlow, GradientTape and Adam optimizer (for high performance and scalability).

# Movie Recommender System - Collaborative Filtering

This project builds a movie recommendation system (Movie Recommender System) using **Collaborative Filtering** technique with **Matrix Factorization** algorithm.

The project is implemented in Jupyter Notebook, comparing two approaches:

1. **Low-level implementation:** Using NumPy to build the Gradient Descent algorithm from scratch.

2. **High-level implementation:** Using TensorFlow (GradientTape, Adam Optimizer) to optimize training and scalability.

## 📂 Structure Project
````
📦 Movie-Recommender-System
┣ 📜 Recommender_System_Collaborative_Filtering.ipynb # Main Source Code
┣ 📜 README.md # Guide
┣ 📂 Dataset (MovieLens Small)
┃ ┣ 📜 movies.csv # List of movies (ID, Title, Genres)
┃ ┣ 📜 ratings.csv # Rating data (User, Movie, Rating)
┃ ┣ 📜 links.csv # Link ID to IMDB/TMDB
┃ ┗ 📜 tags.csv # Keyword tags (Tags)
````
## 🚀 Main features
### 1. Data Preprocessing

- Mapping the original `userId` and `movieId` to the continuous index of the matrix.

- Creating the rating matrix $Y$ (num_movies $\times$ num_users) and the binary matrix $R$ (marking rated movies).

### 2. Mean Normalization

- Performing mean normalization for the rating matrix.

- **Purpose:** Solve the **Cold Start** problem for new movies that have no ratings or new users that have not rated any movies.

### 3. Machine Learning Algorithm

The rating prediction model is based on the linear formula between the Movie Feature Vector ($X$) and the User Parameter Vector ($W$):

$$\text{Prediction} = X \cdot W^T + b$$

#### Cost Function

The objective function includes the mean square error (MSE) and the Regularization component (to avoid Overfitting):

$$
J(X, W, b) = \frac{1}{2} \sum_{(i,j):r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \left( \sum_{j=0}^{n_u-1} \sum_{k=0}^{n-1} (wk^{(j)})^2 + \sum_{i=0}^{nm-1} \sum_{k=0}^{n-1} (x_k^{(i)})^2 \right)
$$

### 4. Optimization Method

- **Method 1: NumPy (Low-level)**
- Calculate partial derivatives $\frac{\partial J}{\partial X}, \frac{\partial J}{\partial W}, \frac{\partial J}{\partial b}$ manually.

- Update weights via Gradient Descent loop.

- **Method 2: TensorFlow (High-level)**
- Use `tf.Variable` to store parameters $X, W, b$.

- Use `tf.GradientTape` for Auto Differentiation.

- Use `keras.optimizers.Adam` for optimizing focusing speed.

## 🛠 Prerequisites

To run this notebook, you need to install the following libraries:

```bash
pip install pandas numpy matplotlib tensorflow
````
