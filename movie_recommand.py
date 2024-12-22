import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns


class MovieRecommender:
    def __init__(self, ratings_path, movies_path):
        """
        Initialize the MovieRecommender with paths to MovieLens dataset files
        """
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        self.models = {}
        self.metrics = {}

    def preprocess_data(self):
        """
        Perform data preprocessing and feature engineering
        """
        # Merge ratings with movie data
        self.full_data = pd.merge(self.ratings_df, self.movies_df, on='movieId')

        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        # Create genre features
        self.movies_df['genres'] = self.movies_df['genres'].str.split('|')
        self.genre_dummies = self.movies_df['genres'].explode().str.get_dummies().groupby(level=0).sum()

        print("Data preprocessing completed")
        self._plot_rating_distribution()

    def _plot_rating_distribution(self):
        """
        Plot rating distribution
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.ratings_df, x='rating', bins=10)
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

    def train_test_split_data(self, test_size=0.2):
        """
        Split data into train and test sets
        """
        self.train_data, self.test_data = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=42
        )
        return self.train_data, self.test_data

    def build_cosine_similarity_model(self):
        """
        Build and train cosine similarity model
        """
        # Calculate movie-movie similarity matrix
        movie_features = self.genre_dummies.values
        self.movie_similarity = cosine_similarity(movie_features)
        self.models['cosine'] = self.movie_similarity
        print("Cosine similarity model built")

    def build_svd_model(self, n_factors=50):
        """
        Build and train SVD model
        """
        # Perform SVD
        U, sigma, Vt = svds(self.user_movie_matrix.values, k=n_factors)
        sigma = np.diag(sigma)
        self.predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        self.models['svd'] = self.predicted_ratings
        print("SVD model built")

    def build_ncf_model(self, n_users, n_movies, n_factors=50):
        """
        Build and train Neural Collaborative Filtering model
        """
        # User input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(n_users + 1, n_factors, name='user_embedding')(user_input)
        user_vec = Flatten(name='flatten_users')(user_embedding)

        # Movie input
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(n_movies + 1, n_factors, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='flatten_movies')(movie_embedding)

        # Concatenate features
        concat = Concatenate()([user_vec, movie_vec])

        # Dense layers
        fc1 = Dense(128, activation='relu')(concat)
        fc2 = Dense(64, activation='relu')(fc1)
        fc3 = Dense(32, activation='relu')(fc2)
        out = Dense(1)(fc3)

        # Create model
        self.models['ncf'] = Model(inputs=[user_input, movie_input], outputs=out)
        self.models['ncf'].compile(optimizer='adam', loss='mse')
        print("NCF model built")

    def build_rnn_model(self, max_sequence_length, n_movies):
        """
        Build and train RNN model for sequential recommendation
        """
        # Sequential input
        sequence_input = Input(shape=(max_sequence_length,), name='sequence_input')
        embedding = Embedding(n_movies + 1, 50, mask_zero=True)(sequence_input)
        lstm = tf.keras.layers.LSTM(50, return_sequences=True)(embedding)
        lstm = tf.keras.layers.LSTM(50)(lstm)
        dense = Dense(100, activation='relu')(lstm)
        output = Dense(n_movies + 1, activation='softmax')(dense)

        self.models['rnn'] = Model(inputs=sequence_input, outputs=output)
        self.models['rnn'].compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
        print("RNN model built")

    def train_model(self):
        """
        Train all recommendation models
        """
        # Preprocess data first
        self.preprocess_data()
        
        # Get dimensions
        n_users = self.ratings_df['userId'].nunique()
        n_movies = self.ratings_df['movieId'].nunique()
        
        # Train basic models
        self.build_cosine_similarity_model()
        self.build_svd_model()
        
        # Train advanced models
        self.build_ncf_model(n_users, n_movies, n_factors=50)
        self.build_rnn_model(max_sequence_length=10, n_movies=n_movies)
        
        print("All models trained successfully")

    def evaluate_model(self, model_name, y_true, y_pred, threshold=3.5):
        """
        Evaluate model performance using various metrics
        """
        # Convert ratings to binary classes for classification metrics
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        precision = precision_score(y_true_binary, y_pred_binary)
        recall = recall_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        }

        self.metrics[model_name] = metrics
        return metrics

    def get_recommendations(self, user_id, model_name='svd', n_recommendations=5):
        """
        Get movie recommendations for a specific user
        """
        if model_name == 'svd':
            user_pred_ratings = self.predicted_ratings[user_id - 1]
            recommended_movie_ids = np.argsort(user_pred_ratings)[-n_recommendations:]

        elif model_name == 'cosine':
            # Get user's rated movies
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            user_movies = user_ratings['movieId'].values

            # Calculate average similarity scores
            sim_scores = self.movie_similarity[user_movies].mean(axis=0)
            recommended_movie_ids = np.argsort(sim_scores)[-n_recommendations:]

        recommendations = self.movies_df.iloc[recommended_movie_ids][['title', 'genres']]
        return recommendations

    def save_results(self, filename='results.xlsx'):
        """
        Save evaluation metrics to Excel file
        """
        results_df = pd.DataFrame(self.metrics).T
        results_df.to_excel(filename)
        print(f"Results saved to {filename}")


# Example usage
def main():
    # Initialize recommender
    ratings_path = "/Users/raeeskhan/Documents/Movie_Recom/ratings.csv"
    movies_path = "/Users/raeeskhan/Documents/Movie_Recom/movies.csv"
    recommender = MovieRecommender(ratings_path, movies_path)

    # Train all models
    recommender.train_model()
    
    # Define test size for evaluation
    test_size = 0.2
    
    # Save results
    recommender.save_results(f'results_{int((1-test_size)*100)}_{int(test_size*100)}_split.xlsx')


if __name__ == "__main__":
    main()
