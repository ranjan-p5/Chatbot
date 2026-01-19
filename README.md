# Chatbot-

This project is a simple, lightweight chatbot built using Python, Natural Language Processing (NLP), and Deep Learning. It uses a Deep Neural Network to classify user inputs into predefined "intents" and generates appropriate responses.


🌟 Features


Natural Language Understanding: Uses NLTK for tokenization and lemmatization.


Deep Learning Model: A Feed-Forward Neural Network built with TensorFlow/Keras.


Dynamic Training: Automatically creates training data and builds a model (.h5) if one doesn't exist.


Easy Customization: Simply edit the intents_data dictionary to add new conversation topics.


🛠️ Technology Stack


Language: Python 3.x


NLP Library: NLTK


Deep Learning: TensorFlow / Keras


Numerical Processing: NumPy


⚙️ How it WorksData 

Ingestion: The bot reads the intents.json file containing patterns and responses.

Preprocessing:

Tokenization: Breaking sentences into individual words.

Lemmatization: Converting words to their root form (e.g., "running" $\rightarrow$ "run").

Neural Network:

Input Layer: Size of the total vocabulary (Bag of Words).

Hidden Layers: Two dense layers (128 and 64 neurons) with ReLU activation and Dropout to prevent overfitting.

Output Layer: Softmax activation to predict the probability of each intent class.

Inference: When you type a message, the bot converts it to a "Bag of Words" vector and predicts the most likely intent.
