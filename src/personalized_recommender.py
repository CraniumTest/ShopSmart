import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class PersonalizedRecommender:

    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Creating a user-item interaction matrix
        self.data.fillna(0, inplace=True)
        self.user_item_matrix = self.data.pivot(index='user_id', columns='product_id', values='interaction')
        self.user_item_matrix = self.user_item_matrix.fillna(0)
        
        # Standardize the data
        self.user_item_matrix = self.scaler.fit_transform(self.user_item_matrix)

    def calculate_similarity(self):
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        return self.similarity_matrix

    def recommend_items(self, user_index, num_recommendations=5):
        similarities = self.similarity_matrix[user_index]
        recommended_indices = np.argsort(similarities)[::-1][1:num_recommendations + 1]
        return self.user_item_matrix.columns[recommended_indices]

# To use the recommender
# recommender = PersonalizedRecommender('data/customer_data.csv')
# recommender.preprocess_data()
# recommender.calculate_similarity()
# print(recommender.recommend_items(user_index=0))
