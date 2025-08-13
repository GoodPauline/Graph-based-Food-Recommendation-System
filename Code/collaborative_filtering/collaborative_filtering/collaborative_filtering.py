"""
Use collaborative filtering to predict unknown scores.
Goal: Predict the unknown scores in the given data.
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


path_user_data = "data/user_data.csv"
path_food_data = "data/food_labels.csv"

ind_start_score_col = 19    # The index where score columns begin in user data file (index starts from 0)


class FoodRecommendor:
    """ Food recommender through collaborative filtering """

    def __init__(self, n_user:int, n_similar_user:int, similarity_metric:str, test_size:float) -> None:
        """
        Initialize the food recommender

        Args:
            n_users (int): The number of users whose information will be used
            n_similar_user (int): The number of similar users to consider
                - It should be no more than `n_users`
            similarity_metric (str): User-user similarity metric
                - Choice: "euclidean", "cosine"
            test_size (float): Proportion of test data
                - A float in (0, 1)
        """
        self.n_user = n_user
        self.n_similar_user = n_similar_user

        self.similarity_metric = similarity_metric
        self.user_similarities = np.zeros(shape=(self.n_user, self.n_user))    # User similarity matrix

        df_score_matrix, test_data = self.split_train_test(test_size=test_size)

        self.user_ids = df_score_matrix.index.to_numpy()   # numpy array of user indices
        self.food_ids = df_score_matrix.columns.to_numpy()  # numpy array of food indices
        self.score_matrix = df_score_matrix.to_numpy()    # numpy matrix form of score matrix

        self.test_data = test_data
        return
    
    def euclidean_distance(self, user_id_1:int, user_id_2:int) -> float:
        """
        Calculate the Euclidean distance between two user vectors

        Args:
            user_id_1 (int): One user's index
            user_id_2 (int): Another user's index
        
        Returns:
            similarity (float): Similarity between the two users
        """
        # Extract the two user vectors
        x1, x2 = self.score_matrix[user_id_1], self.score_matrix[user_id_2]
        # Calculate and return the similarity
        return 1 / (1 + np.linalg.norm(x1 - x2))
    
    def cosine_similarity(self, user_id_1:int, user_id_2:int) -> float:
        """
        Calculate the Euclidean distance between two user vectors

        Args:
            user_id_1 (int): One user's index
            user_id_2 (int): Another user's index
        
        Returns:
            similarity (float): Similarity between the two users
        """
        # Extract the two user vectors
        x1, x2 = self.score_matrix[user_id_1], self.score_matrix[user_id_2]
        # Calculate and return the similarity
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def calculate_similarity(self, user_id_1:int, user_id_2:int) -> float:
        """
        Calculate the similarity between two users

        Args:
            user_id_1 (int): One user's index
            user_id_2 (int): Another user's index
        
        Returns:
            similarity (float): Similarity between the two users
        """
        if self.similarity_metric == 'cosine':
            return self.cosine_similarity(user_id_1=user_id_1, user_id_2=user_id_2)
        elif self.similarity_metric == 'euclidean':
            return self.euclidean_distance(user_id_1=user_id_1, user_id_2=user_id_2)
        else:
            raise ValueError("Invalid similarity metric. Choose 'cosine' or 'euclidean'")
    
    def calculate_similarity_matrix(self) -> None:
        """
        Calculate the similarity matrix
        """
        for i in range(0, self.n_user):
            for j in range(i, self.n_user):   # avoid repeatedly calculate one element in the matrix

                if i == j:
                    self.user_similarities[i][j] = 0  # ignore self-similarity
                else:
                    similarity = self.calculate_similarity(user_id_1=i, user_id_2=j)

                    # Symmetric similarity matrix
                    self.user_similarities[i][j] = similarity
                    self.user_similarities[j][i] = similarity
        return

    def fit(self) -> None:
        """
        Train the recommendation system with the score matrix
        """
        # Compute user-user similarity matrix
        self.calculate_similarity_matrix()
        return
    
    def predict(self, ind_user:int, ind_food:int) -> int:
        """
        Predict rating for a given user and food

        Args:
            ind_user (int): Index of the given user
            ind_food (int): Index of the given food
        
        Returns:
            predicted_score (int) Predicted score, an integer in [1, 5]
        """
        user_scores = self.score_matrix[ind_user]   # The user's scores on each food

        # If the user already scores this food, return that score
        if user_scores[ind_food] > 0:
            return user_scores[ind_food]

        # Sort users based on their similarities to the given user: Most similar -> least similar
        similar_users = np.argsort(self.user_similarities[ind_user])[::-1]
        # Extract the indices of users that score on the given food
        rated_by_similar = np.where(self.score_matrix[:, ind_food] > 0)[0]
        # Extract the indices of similar users' that score on the given food
        similar_users_rated = [u for u in similar_users if u in rated_by_similar]

        if len(similar_users_rated) == 0:
            # No similar users score on the given food: no prediction possible, so return 0
            return 0

        # Determine the number of considered similar users
        max_n_eligible_similar_user = len(similar_users_rated)
        k = max_n_eligible_similar_user if max_n_eligible_similar_user < self.n_similar_user else self.n_similar_user

        # Take top k eligible similar users' indices
        indices_top_k_users = similar_users_rated[:k]
        # Take their similarities to the given user
        similarities_top_k_users = self.user_similarities[ind_user, indices_top_k_users]
        # Take their scores on the given food
        scores_top_k_users = self.score_matrix[indices_top_k_users, ind_food]

        # Weighted average of scores
        predicted_score = np.dot(similarities_top_k_users, scores_top_k_users) / np.sum(similarities_top_k_users)

        # Make sure that the predicted score is an integer in [1, 5]
        predicted_score = np.clip(predicted_score, 1, 5)   # constrained into [1, 5]
        predicted_score = int(np.round(predicted_score))   # constrained to an int

        return predicted_score
    
    def evaluate(self) -> tuple[float, float]:
        """
        Evaluate the model on test data

        Returns:
            eval_metrics (tuple[float, float]): `(RMSE, MAE)`
                - `RMSE` (float): Root Mean Squared Error
                - `MAE` (float): Mean Absolute Error
        """

        predicted_ind_pair = []   # list of 2-dim tuples: (ind_user, ind_food)
        predicted_scores, actual_scores = [], []

        for ind_user, ind_food, actual_score in self.test_data:
            pred_score = self.predict(ind_user, ind_food)

            if pred_score > 0:
                predicted_ind_pair.append((ind_user, ind_food))
                predicted_scores.append(pred_score)
                actual_scores.append(actual_score)
        
        if len(predicted_scores) == 0:   # No predictions are made
            return float('nan'), float('nan')
        
        # RMSE
        mse = mean_squared_error(actual_scores, predicted_scores)
        rmse = np.sqrt(mse)
        # MAE
        mae = np.mean( np.abs(np.array(actual_scores) - np.array(predicted_scores)) )

        return rmse, mae

    def prepare_score_matrix(self, n_user:int) -> pd.DataFrame:
        """
        Prepare the score matrix with users' and foods' indices

        Args:
            n_user (int) The number of users whose information will be used
        
        Returns:
            score_matrix (pd.DataFrame): The score matrix
        """
        global path_user_data, path_food_data
        global ind_start_score_col

        # Load user data
        user_df = pd.read_csv(path_user_data, encoding="utf-8-sig", nrows=n_user)

        # Extract user features (first 19 columns) and food ratings (remaining columns)
        user_features = user_df.iloc[:, :ind_start_score_col]
        food_scores = user_df.iloc[:, ind_start_score_col:]

        # Load food data to get food names
        food_df = pd.read_csv(path_food_data, encoding="utf-8-sig")
        food_names = food_df["FoodName"].values

        # Create a score matrix with users' indices as rows and foods as columns
        df_score_matrix = pd.DataFrame(
            food_scores.values, 
            columns=food_names,
            index=user_df.index
        )

        return df_score_matrix
    
    def split_train_test(self, test_size:float) -> tuple[pd.DataFrame, list]:
        """
        Split the whole score matrix for training and testing data

        Method:
        1. Create test data by sampling known scores
        2. Remove the test data from the original matrix by setting it to 0, in order to get training data

        Args:
            test_size (float) Proportion of test data size, a float in (0, 1)
        """
        # Get the score matrix in DataFrame type
        df_score_matrix = self.prepare_score_matrix(n_user=self.n_user)

        test_data = []   # A list of tuples: (ind_user, ind_food, score)

        # Get the indices of non-zero entries in the score matrix
        non_zero_indices = np.where(df_score_matrix.values > 0)

        # Sample test size of the known ratings
        sample_size = int(len(non_zero_indices[0]) * test_size)
        # Sample user-food pairs' indices
        sampled_indices = np.random.choice(
            len(non_zero_indices[0]), 
            size=sample_size, 
            replace=False
        )

        for ind_pair in sampled_indices:
            # Get the index of user and food respectively
            ind_user = non_zero_indices[0][ind_pair]
            ind_food = non_zero_indices[1][ind_pair]
            # Get the corresponding score
            score = df_score_matrix.iloc[ind_user, ind_food]

            # Append above three types of data into test data
            test_data.append((ind_user, ind_food, score))

            # Set the corresponding entry to 0 in the training matrix to remove it
            df_score_matrix.iloc[ind_user, ind_food] = 0
        
        return df_score_matrix, test_data


def main() -> None:
    similarity_metrics = ["cosine", "euclidean"]
    results = {}

    for similarity_metric in similarity_metrics:
        # Create, train and test a recommendor
        recommendor = FoodRecommendor(
            n_user=100,
            n_similar_user=10,
            similarity_metric=similarity_metric,
            test_size=0.2
        )
        recommendor.fit()
        rmse, mae = recommendor.evaluate()

        # Record results
        results[similarity_metric] = {"RMSE": rmse, "MAE": mae}

    # Print results
    for similarity_metric, eval_metrics in results.items():
        print(f"{similarity_metric.upper()} - RMSE: {eval_metrics['RMSE']:.4f}, MAE: {eval_metrics['MAE']:.4f}")
    
    return


if __name__ == "__main__":
    main()