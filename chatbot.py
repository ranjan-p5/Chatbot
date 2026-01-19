import nltk
import json
import pickle
import numpy as np
import random
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

 
lemmatizer = WordNetLemmatizer()

 
intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
            "responses": ["Hello! How can I help you?", "Good to see you again!", "Hi there!"],
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "I am leaving"],
            "responses": ["See you later!", "Have a nice day!", "Bye! Come back soon."]
        },
        {
            "tag": "help",
            "patterns": ["How can you help?", "What do you do?", "Help me"],
            "responses": ["I can answer basic questions and chat with you!"]
        },
        {
            "tag": "weather",
            "patterns": ["What is the weather?", "Is it raining?", "How is the temperature?"],
            "responses": ["I'm not connected to a satellite, but it looks sunny from here!"]
        }
    ]
}

with open('intents.json', 'w') as f:
    json.dump(intents_data, f)

 
nltk.download('punkt')
nltk.download('wordnet')

 
def train_bot():
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    intents = json.loads(open('intents.json').read())

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

     
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    print("Training started...")
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)
    model.save('chatbot_model.h5')
    print("Model created and saved!")

 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I don't understand that."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

 
if __name__ == "__main__":
    # Train the model if it doesn't exist
    if not os.path.exists('chatbot_model.h5'):
        train_bot()
    
    
    model = load_model('chatbot_model.h5')
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    intents = json.loads(open('intents.json').read())

    print("\nBot is running! (Type 'quit' to stop)")

    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        
        ints = predict_class(message, model, words, classes)
        res = get_response(ints, intents)
        print("Bot:", res)