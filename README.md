# Recommender System Project

This project implements and compares two popular approaches for building Recommender Systems: **Content-Based Filtering** using Deep Learning and **Collaborative Filtering** using Matrix Factorization.

The project is written in Python and leverages powerful data analysis and machine learning libraries such as TensorFlow, Pandas, and NumPy.

## 📂 Project Structure

The project consists of two main notebooks corresponding to the two approaches:

### 1. Deep Learning for Content-Based Filtering
**File:** `Deep Learning for Content-Based Filtering.ipynb`

This method suggests movies based on the characteristics (features) of both the user and the items (movies).
* **Technique:** Neural Networks using TensorFlow/Keras.
* **Data Processing:**
    * *Movie Features:* Extracts release year and applies One-hot encoding for movie genres.
    * *User Features:* Computes a user preference vector based on their average ratings across different genres.
* **Model Architecture:**
    * Consists of two parallel sub-networks: **User Network** (generates vector $v_u$) and **Item Network** (generates vector $v_m$).
    * The output is the Dot product of these two vectors to predict the rating.
* **Additional Feature:** Recommends similar movies by calculating the squared distance between movie vectors ($v_m$).

### 2. Collaborative Filtering Recommender Systems
**File:** `Recommender_System_Collaborative_Filtering.ipynb`

This method recommends items based on the similarity in rating behavior among users, without needing detailed knowledge of the movie content.
* **Technique:** Matrix Factorization.
* **Algorithms:**
    * Implementation of **Gradient Descent** from scratch to optimize the Cost Function with Regularization ($\lambda$).
    * Uses **Mean Normalization** to handle the "Cold Start" problem (new users with no ratings).
    * Includes a parallel implementation using **TensorFlow** (with `tf.GradientTape` and `keras.optimizers`) for performance comparison and optimization.
* **Goal:** To learn latent vectors $X$ (for movies) and $W$ (for users) such that their product best predicts the actual ratings.

## 📊 Dataset

The project uses the MovieLens dataset (or a similar structure) containing CSV files:
* `movies.csv`: Contains movie information (`movieId`, `title`, `genres`).
* `ratings.csv`: Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`).

## 🛠 Technologies Used

* **Language:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Machine Learning Tools:** Scikit-learn (StandardScaler, MinMaxScaler, train_test_split)

## 🚀 Installation and Usage

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <your-repo-url>
    cd <your-project-folder>
    ```

2.  **Install necessary libraries:**
    Ensure you have Python installed along with the following libraries:
    ```bash
    pip install numpy pandas matplotlib tensorflow scikit-learn tabulate
    ```

3.  **Prepare Data:**
    Place the `movies.csv` and `ratings.csv` files in the same directory as the notebooks.

4.  **Run the Notebooks:**
    Open Jupyter Notebook or JupyterLab and execute the cells in the `.ipynb` files to observe the training process and recommendation results.

## 📈 Results

* **Content-Based:** The model can predict a user's rating for a specific movie and generate a list of movies with similar content (e.g., genre, style) to a source movie.
* **Collaborative Filtering:** The system predicts missing ratings in the User-Item matrix, providing a personalized Top-N movie list for specific users based on their latent preferences.

---
*This project is part of a study and implementation of Recommender Systems.*
