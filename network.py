import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

movies_data = pd.read_csv("./data/movies.csv")
ratings_data = pd.read_csv("./data/ratings.csv")

movies_data["genres"] = [eval(s) for s in movies_data["genres"]]
unique_genres = set()
for genres in movies_data["genres"]:
    unique_genres.update(genres)

unique_genres = sorted(list(unique_genres))

# One-hot encode genres for each movie
for unique_value in unique_genres:
    movies_data[unique_value] = movies_data.apply(
        lambda row: 1 if unique_value in movies_data["genres"][row.name] else 0, axis=1
    )


class ImprovedMovieRecommender:
    def __init__(self, movies_df, ratings_df, embedding_dim=100):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.embedding_dim = embedding_dim

        # Initialize encoders and scalers
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.rating_scaler = MinMaxScaler()

        # Define genre columns explicitly
        self.genre_columns = [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "History",
            "Horror",
            "Music",
            "Mystery",
            "Romance",
            "Science Fiction",
            "TV Movie",
            "Thriller",
            "War",
            "Western",
        ]

        # Prepare data
        self.prepare_data()

        # Build model
        self.model = self.build_model()

    def prepare_data(self):
        # Encode users and movies
        self.ratings_df["user_encoded"] = self.user_encoder.fit_transform(
            self.ratings_df["user_id"]
        )
        self.ratings_df["movie_encoded"] = self.movie_encoder.fit_transform(
            self.ratings_df["movie_id"]
        )

        # Scale ratings to [0,1] range
        self.ratings_df["rating_scaled"] = self.rating_scaler.fit_transform(
            self.ratings_df[["rating_val"]]
        )

        # Calculate global statistics
        self.global_mean = self.ratings_df["rating_val"].mean()
        self.user_means = self.ratings_df.groupby("user_encoded")["rating_val"].mean()
        self.movie_means = self.ratings_df.groupby("movie_encoded")["rating_val"].mean()

        # Prepare features
        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)

        # Create genre features
        self.prepare_genre_features()

    def prepare_genre_features(self):
        # Ensure all genre columns exist in the DataFrame
        for genre in self.genre_columns:
            if genre not in self.movies_df.columns:
                self.movies_df[genre] = 0

        # Extract genre features
        self.genre_features = self.movies_df[self.genre_columns].fillna(0).values
        self.n_genres = len(self.genre_columns)

        # RMSE custom metric

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    # R-squared custom metric
    def r_squared(y_true, y_pred):
        ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
        return 1 - (ss_residual / ss_total)

    def build_model(self):
        # Input layers
        user_input = tf.keras.layers.Input(shape=(1,), name="user_input")
        movie_input = tf.keras.layers.Input(shape=(1,), name="movie_input")
        genre_input = tf.keras.layers.Input(shape=(self.n_genres,), name="genre_input")

        # Embedding layers with regularization
        user_embedding = tf.keras.layers.Embedding(
            self.n_users,
            self.embedding_dim,
            embeddings_regularizer=l2(0.001),
            name="user_embedding",
        )(user_input)

        movie_embedding = tf.keras.layers.Embedding(
            self.n_movies,
            self.embedding_dim,
            embeddings_regularizer=l2(0.001),
            name="movie_embedding",
        )(movie_input)

        # Flatten embeddings
        user_vector = tf.keras.layers.Flatten()(user_embedding)
        movie_vector = tf.keras.layers.Flatten()(movie_embedding)

        user_vector = tf.keras.layers.Dense(128, activation="relu")(user_vector)
        movie_vector = tf.keras.layers.Dense(128, activation="relu")(movie_vector)

        # Process genre features
        genre_dense = tf.keras.layers.Dense(32, activation="relu")(genre_input)

        # Concatenate all features
        merged = tf.keras.layers.Concatenate()([user_vector, movie_vector, genre_dense])

        # Dense layers with dropout and batch normalization
        dense1 = tf.keras.layers.Dense(64, activation="relu")(merged)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dense1 = tf.keras.layers.Dropout(0.3)(dense1)

        dense2 = tf.keras.layers.Dense(32, activation="relu")(dense1)
        dense2 = tf.keras.layers.BatchNormalization()(dense2)

        # Output layer
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dense2)

        # Create model
        model = tf.keras.Model(
            inputs=[user_input, movie_input, genre_input], outputs=output
        )

        # Compile model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=["mae", "mse"]
        )

        return model

    def train(self, epochs=100, batch_size=64):
        # Prepare training data
        X_users = self.ratings_df["user_encoded"].values
        X_movies = self.ratings_df["movie_encoded"].values
        y = self.ratings_df["rating_scaled"].values

        # Get genre features for each movie
        movie_indices = self.movie_encoder.transform(self.ratings_df["movie_id"])
        X_genres = self.genre_features[movie_indices]

        # Train-test split
        (
            X_users_train,
            X_users_test,
            X_movies_train,
            X_movies_test,
            X_genres_train,
            X_genres_test,
            y_train,
            y_test,
        ) = train_test_split(
            X_users, X_movies, X_genres, y, test_size=0.2, random_state=42
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=0.00001
            ),
        ]

        # Train model
        history = self.model.fit(
            [X_users_train, X_movies_train, X_genres_train],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_users_test, X_movies_test, X_genres_test], y_test),
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def evaluate(self, X_users_test, X_movies_test, X_genres_test, y_test):
        # Evaluate the model on the test data
        test_metrics = self.model.evaluate(
            [X_users_test, X_movies_test, X_genres_test], y_test, verbose=1
        )

        print("Evaluation Metrics:")
        print(f"MAE: {test_metrics[1]}")
        print(f"MSE: {test_metrics[2]}")
        print(f"RMSE: {test_metrics[3]}")
        print(f"R-squared: {test_metrics[4]}")

    def get_recommendations(self, movie_name, top_k=10):
        try:
            # Get movie index and features
            movie_idx = self.movie_encoder.transform([movie_name])[0]
            movie_genres = self.genre_features[movie_idx]

            # Create synthetic users for diversity
            n_synthetic_users = 100
            user_candidates = np.random.randint(0, self.n_users, size=n_synthetic_users)
            movie_candidates = np.full(n_synthetic_users, movie_idx)
            genre_candidates = np.tile(movie_genres, (n_synthetic_users, 1))
            print(genre_candidates.shape)

            # Get predicted ratings for the input movie
            input_predictions = self.model.predict(
                [user_candidates, movie_candidates, genre_candidates]
            )

            # Find users who would rate this movie highly
            relevant_user_indices = user_candidates[
                np.argsort(input_predictions.flatten())[-10:]
            ]

            # Get recommendations for these users
            all_movies = np.arange(self.n_movies)
            all_predictions = []

            for user_idx in relevant_user_indices:
                user_input = np.full(self.n_movies, user_idx)
                genre_input = self.genre_features

                # Ensure input structure matches model input
                predictions = self.model.predict([user_input, all_movies, genre_input])
                all_predictions.append(predictions.flatten())

            # Aggregate predictions
            mean_predictions = np.mean(all_predictions, axis=0)
            top_movie_indices = np.argsort(mean_predictions)[::-1]

            # Filter out the input movie and get recommendations
            recommended_movies = []
            for idx in top_movie_indices:
                if idx != movie_idx:
                    movie_id = self.movie_encoder.inverse_transform([idx])[0]
                    recommended_movies.append(movie_id)
                    if len(recommended_movies) == top_k:
                        break

            return recommended_movies

        except ValueError:
            print(f"Movie '{movie_name}' not found in the dataset.")
            return []


def create_recommender(movies_df, ratings_df):
    recommender = ImprovedMovieRecommender(movies_df, ratings_df)
    history = recommender.train()
    return recommender, history


recommender, history = create_recommender(movies_data, ratings_data)


def evaluate(self, X_users_test, X_movies_test, X_genres_test, y_test):
    # Evaluate the model on the test data
    test_metrics = self.model.evaluate(
        [X_users_test, X_movies_test, X_genres_test], y_test, verbose=1
    )

    print("Evaluation Metrics:")
    print(f"MAE: {test_metrics[1]}")
    print(f"MSE: {test_metrics[2]}")
    print(f"RMSE: {test_metrics[3]}")
    print(f"R-squared: {test_metrics[4]}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(historyhistory["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historyhistory["mae"], label="Training MAE")
plt.plot(historyhistory["val_mae"], label="Validation MAE")
plt.title("Model MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.show()

# Get movie recommendations
movie_name = "the-dark-knight"  # Replace with an actual movie name from your dataset
recommended_movies = recommender.get_recommendations(movie_name, top_k=10)

print("Top 10 recommended movies for:", movie_name)
print(recommended_movies)
