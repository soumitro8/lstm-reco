import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import binarize
import pickle


class LSTM_Movie_Rec:
    def __init__(self, path, sep):
        self.model = None
        self.item_count = None
        self.path = path
        self.sep = sep
        self.movie_id_dict, self.id_movie_dict = self.get_movies()
        self.train_in = None
        self.train_out = None
        self.test_in = None
        self.test_out = None

    def store_model(self):
        self.model.save('./Model/lstm.h5')
        obj = {
            "movie_id_dict": self.movie_id_dict,
            "id_movie_dict": self.id_movie_dict,
            "train_in": self.train_in,
            "train_out": self.train_out,
            "test_in": self.test_in,
            "test_out": self.test_out,
            "item_count": self.item_count
        }
        pickle.dump(obj, open("./Model/params.p", "wb"))

    def load_model(self):
        self.model = tf.keras.models.load_model('./Model/lstm.h5')

        data = pickle.load(open("./Model/params.p", "rb"))
        self.movie_id_dict = data["movie_id_dict"]
        self.id_movie_dict = data["id_movie_dict"]
        self.train_in = data["train_in"]
        self.train_out = data["train_out"]
        self.test_in = data["test_in"]
        self.test_out = data["test_out"]
        self.item_count = data["item_count"]

    def convert_id_to_movie(self, list_of_ids):
        return list(map((lambda x: self.id_movie_dict[x]), list_of_ids))

    def convert_movie_to_id(self, list_of_movies_names):
        return list(map((lambda x: self.movie_id_dict[x]), list_of_movies_names))

    # Reads raw data from file
    # Returns dataframe with columns - user, item, rating, timestamp
    def read_raw_data_from_file(self):
        dataset = []
        with open(self.path, 'r') as f:
            raw_data = f.readlines()
            for line in raw_data:
                line = line.replace(self.sep, " ")
                user, item, rating, timestamp = line.split()
                dataset.append([int(user), int(item), int(rating), int(timestamp)])
        dataset = pd.DataFrame(data=dataset, columns=['user', 'item', 'rating', 'timestamp'])

        # Sort entries based on timestamp
        return dataset.sort_values(by='timestamp')

    # Vectorize input and output sequences to get vectors of fixed length
    # For instance, [2,4] -> [0,0,1,0,1] and [0,1,3] -> [1,1,0,1,0]
    # Returns a sparse numpy array
    def vectorizer(self, sequence, item_count):
        results = np.zeros(shape=item_count, dtype=np.int16)
        for value in sequence:
            # value-1 cuz indexing starts from 0 whereas movie_id starts from 1
            results[value - 1] = 1
        return results

    # Takes in dataframe with user, item, rating, timestamp
    # Returns input and output sequences for testing and training
    # train_test_ratio = (train samples) / (total samples), defaults to 0.8
    # in_out_ratio = (number of items used as input) / (total items in a sequence), defaults to 0.8
    def pre_process_data(self, train_test_ratio=0.8, in_out_ratio=0.8):

        dataset = self.read_raw_data_from_file()

        item_count = dataset['item'].max()
        self.item_count = item_count

        dataset_np_array = dataset.to_numpy(dtype=np.int64)
        # Building a dictionary that stores each movie a given user has watched
        user_item_dict = defaultdict(list)
        for user, item, r, t in dataset_np_array:
            user_item_dict[user].append(item)

        # Generate a list of movie sequences
        movie_sequences = [v for k, v in user_item_dict.items()]

        # Remove sequences with too few samples
        movie_sequences = list(filter((lambda x: len(x) * in_out_ratio > 3), movie_sequences))

        # Perform train-test split based on train_test_ratio
        n_train_sequences = int(len(movie_sequences) * train_test_ratio)
        train_sequences = movie_sequences[:n_train_sequences]
        test_sequences = movie_sequences[n_train_sequences:]

        # Generate input and output sequences based on in_out_ratio
        train_input = list(seq[:int(len(seq) * in_out_ratio)] for seq in train_sequences)
        train_output = list(seq[int(len(seq) * in_out_ratio):] for seq in train_sequences)
        test_input = list(seq[:int(len(seq) * in_out_ratio)] for seq in test_sequences)
        test_output = list(seq[int(len(seq) * in_out_ratio):] for seq in test_sequences)

        # Vectorize input and output sequences to get vectors of fixed length
        # For instance, [2,4] -> [0,0,1,0,1] and [0,1,3] -> [1,1,0,1,0]
        train_input_vectors = np.asarray(list(map((lambda x: self.vectorizer(x, item_count)), train_input)))
        train_output_vectors = np.asarray(list(map((lambda x: self.vectorizer(x, item_count)), train_output)))
        test_input_vectors = np.asarray(list(map((lambda x: self.vectorizer(x, item_count)), test_input)))
        test_output_vectors = np.asarray(list(map((lambda x: self.vectorizer(x, item_count)), test_output)))

        return train_input_vectors, train_output_vectors, test_input_vectors, test_output_vectors

    # Returns an ordered list of all movies from path
    def get_movies(self, path="./data/ml-1m/movies.dat"):
        movies_list = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, movie, _ = line.split("::")
                movies_list.append(movie)

        movie_to_id_dict = dict((movies_list[i], i + 1) for i in range(len(movies_list)))
        id_to_movie_dict = dict(map(reversed, movie_to_id_dict.items()))
        return movie_to_id_dict, id_to_movie_dict

    def create_model(self, shape, output_length, loss, optimizer):
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(1000, input_shape=shape, activation="relu"),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Dense(output_length, activation='sigmoid')
        ])

        lstm_model.compile(optimizer=optimizer, loss=loss)
        return lstm_model

    def train_model(self, loss="binary_crossentropy", optimizer="adam", epochs=10):
        train_in, train_out, self.test_in, self.test_out = self.pre_process_data()

        input_length = output_length = train_in.shape[1]

        # Reshape training data to 3D, each sequences is a timestamp
        train_in = train_in.reshape(-1, 1, input_length)
        self.test_in = self.test_in.reshape(-1, 1, input_length)

        # Define the LSTM model
        lstm_model = self.create_model(shape=train_in.shape[-2:], output_length=output_length, optimizer=optimizer,
                                       loss=loss)

        # Perform training
        lstm_model.fit(x=train_in, y=train_out, epochs=epochs)

        self.model = lstm_model
        self.store_model()
        return lstm_model

    # Takes in list of movie names
    # Returns list of top-k recommended movies
    def top_k_movies(self, user_input, k=10):
        user_input = self.convert_movie_to_id(user_input)
        vectorized_input = self.vectorizer(user_input, self.item_count)
        result = self.model.predict(vectorized_input.reshape(1, 1, -1)).reshape(-1)
        top_k_movie_indexes = np.argsort(result)[-k:]
        return self.convert_id_to_movie(top_k_movie_indexes)

    # Returns hit ratio
    def test_hit_score(self):
        predicted = self.model.predict(self.test_in)
        binarized_prediction = binarize(predicted, np.mean(predicted))
        n_total_samples = predicted.shape[0]
        hits = 0
        for i in range(n_total_samples):
            hits += np.dot(self.test_out[i], binarized_prediction[i]) / np.sum(self.test_out[i])

        return hits / n_total_samples


ML_1M_PATH = "./data/ml-1m/ratings.dat"
Rec = LSTM_Movie_Rec(path=ML_1M_PATH, sep='::')

# Rec.train_model(epochs=5)

Rec.load_model()

print("Hit score", Rec.test_hit_score())
watched = ["Star Wars: Episode I - The Phantom Menace (1999)",
           "Star Wars: Episode IV - A New Hope (1977)",
           "Star Wars: Episode V - The Empire Strikes Back (1980)"]
print("You should watch", Rec.top_k_movies(watched))
