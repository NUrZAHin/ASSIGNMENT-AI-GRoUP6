# Import necessary libraries
import os # Import the os module
import tensorflow as tf # Import the TensorFlow module
from tensorflow.keras.preprocessing.text import Tokenizer # Import the tokenizer class
from tensorflow.keras.preprocessing.sequence import pad_sequences # Import the pad_sequences function
from tensorflow.keras.layers import Dense, LSTM # Import the necessary layers
from tensorflow.keras.layers import Bidirectional# Import the necessary layers
from bs4 import BeautifulSoup # Import the BeautifulSoup class
import pandas as pd # Import the pandas module
from tqdm import tqdm # Import the tqdm module
import re # Import the re module
import matplotlib.pyplot as plt # Import the matplotlib module

fig ,ax = plt.subplots() # Create a figure and axes for plotting later

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix # Import the necessary metrics
from sklearn.model_selection import train_test_split # Import the train_test_split function

import nltk # Import the NLTK module
nltk.download('stopwords') # Download the NLTK stopwords list
nltk.download('wordnet') # Download the WordNet lemmatizer
nltk.download('omw-1.4') # Download the Open Multilingual Wordnet
from nltk.corpus import stopwords # Import the stopwords list
from nltk.stem.wordnet import WordNetLemmatizer # Import the WordNet lemmatizer

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin") # Add the CUDA bin directory to the PATH
tf.config.list_physical_devices('GPU') #Check if GPU is available

# Check if TensorFlow is using a GPU
if tf.test.is_gpu_available():  # If a GPU is available
    print("TensorFlow is using a GPU") # Print a message
else: # Otherwise
    print("TensorFlow is not using a GPU") # Print a message

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use GPU 0
print ("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # Print the number of available GPUs
print ("Show the Gpus: ", tf.config.experimental.list_physical_devices('GPU')) # Print the available GPUs

# List the available devices
devices = tf.config.experimental.list_physical_devices('GPU') # List the available devices

# Print the name and type of each device
for device in devices: # For each device
    print(f'{device.name} ({device.device_type})') # Print the name and type of the device

# Load the dataset
# dataset = pd.read_csv('Reviews.csv') # Load the reviews dataset from a CSV file
dataset = pd.read_csv('Reviews.csv') # Load the reviews dataset from a CSV file
dataset = dataset.sample(50000) # Sample 50,000 reviews from the dataset

print ("Finish loading the dataset")

def scorePartition(x): # Create a function to partition the scores
    if x < 3:  # If the score is less than 3
        return 0  # Return 0
    return 1  # Otherwise, return 1

# Dropping the dups in dataset
dataset = dataset.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False) # Remove duplicates


print ("Finish filtering the duplicates and the columns")

# Generalizing the score
actualScore = dataset['Score'] # Get the scores
positiveNegative = actualScore.map(scorePartition) # Map the scores to 0 or 1
dataset['Score'] = positiveNegative # Replace the original scores with the mapped scores

print ("Finish generalizing the score")

def removeHTMLTags(review): # Create a function to remove HTML tags
    soup = BeautifulSoup(review, 'lxml') # Parse the HTML
    return soup.get_text()  # Extract the text from the 

def removeApostrophe(review): # Create a function to remove apostrophes
    phrase = re.sub(r"won't", "will not", review)  # Replace "won't" with "will not"
    phrase = re.sub(r"can\'t", "can not", review)  # Replace "can't" with "can not"
    phrase = re.sub(r"n\'t", " not", review)  # Replace "n't" with " not"
    phrase = re.sub(r"\'re", " are", review)  # Replace "'re" with " are"
    phrase = re.sub(r"\'s", " is", review)  # Replace "'s" with " is"
    phrase = re.sub(r"\'d", " would", review)  # Replace "'d" with " would"
    phrase = re.sub(r"\'ll", " will", review)  # Replace "'ll" with " will"
    phrase = re.sub(r"\'t", " not", review)  # Replace "'t" with " not"
    phrase = re.sub(r"\'ve", " have", review)  # Replace "'ve" with " have"
    phrase = re.sub(r"\'m", " am", review)  # Replace "'m" with " am"
    return phrase  # Return the modified review


def removeAlphaNumericWords(review): # Create a function to remove words containing letters and numbers
     return re.sub("\S*\d\S*", "", review).strip()  # Remove words that contain letters and numbers
 
def removeSpecialChars(review): # Create a function to remove special characters
     return re.sub('[^a-zA-Z]', ' ', review)  # Remove special characters


# Create a function to preprocess the text data
def preprocess_text(text):
    # Remove HTML tags
    text = removeHTMLTags(text)
    
    # Replace certain phrases with expanded forms
    text = removeApostrophe(text)

    # Remove words containing letters and numbers
    text = removeAlphaNumericWords(text)

    # Remove special characters
    text = removeSpecialChars(text)

    # Lowercase all the words
    text = text.lower()

    # Tokenize the text
    text = text.split()

    # Remove stopwords and lemmatize remaining words
    lmtzr = WordNetLemmatizer() # Create a WordNet lemmatizer
    text = [lmtzr.lemmatize(word, 'v') for word in text if not word in set(stopwords.words('english'))] # Lemmatize the words
    text = " ".join(text) # Join the words
    

    return text # Return the preprocessed text

# Preprocess the text data and create a list of cleaned reviews
corpus = [] # Create an empty list

for index, row in tqdm(dataset.iterrows()): # Iterate over the dataset
    review = preprocess_text(row['Text']) # Preprocess the review text
    corpus.append(review) # Append the preprocessed text to the list

data = pd.DataFrame({'Text': corpus, 'Score': dataset['Score']}) # Create a DataFrame with the preprocessed text and the scores
data.to_csv('preprocessed_data.csv', index=False) # Save the DataFrame as a CSV file
print ("Finish preprocessing the text data")

# Use the Tokenizer class to vectorize the text data
max_features = 4000 # Set the maximum number of words to use
tokenizer = Tokenizer(num_words=max_features, lower=True, split=' ') # Create a Tokenizer object
tokenizer.fit_on_texts(corpus) # Fit the Tokenizer on the corpus
X = tokenizer.texts_to_sequences(corpus) # Convert the text to sequences

print ("Finish vectorizing the text data")

# Pad the sequences to the same length
maxlen = 100 # Set the maximum length of the sequences
X = pad_sequences(X, maxlen=maxlen) # Pad the sequences
print ("Finish padding the sequences")


print ("Start putting the data into a DataFrame word index")
data = pd.DataFrame(X) # Create a DataFrame with the padded sequences
data.to_csv("tokenized_data.csv", index=False) # Save the DataFrame as a CSV file

token_index = tokenizer.word_index # Get the word index
index = pd.RangeIndex(start=1, stop=len(token_index)+1, step=1) # Create a range index
data = pd.DataFrame(token_index,index=index) # Create a DataFrame with the word index
data.to_csv("tokenized_index.csv", index=False) # Save the DataFrame as a CSV file

# Split the data into training and test sets
y = dataset.iloc[:,6].values # Get the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # Split the data into training and test sets

print ("Finish splitting the data into training and test sets")

vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the zero-padding
max_length = max([len(s) for s in X]) # Get the maximum length of the sequences


    
print ("Building model...")

with tf.device('/device:GPU:0'): # Use the GPU
    model = tf.keras.Sequential() # Create a sequential model
    model.add(tf.keras.layers.Embedding(vocab_size,64,input_length=max_length)) # Add an embedding layer
    model.add(Bidirectional(LSTM(128, return_sequences=True))) # Add a bidirectional LSTM layer
    model.add(tf.keras.layers.GlobalMaxPool1D()) # Add a pooling layer
    model.add(Dense(32, activation="relu")) # Add a dense layer
    model.add(tf.keras.layers.Dropout(0.2)) # Add a dropout layer
    model.add(Dense(32)) # Add a dense layer
    model.add(tf.keras.layers.Dropout(0.2)) # Add a dropout layer
    model.add(Dense(32, activation="relu")) # Add a dense layer
    model.add(Dense(1, activation="sigmoid")) # Add a dense layer with a sigmoid activation function
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.00001) # Create an Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy']) # Compile the model


    
print ("Model built!")  

print (model.summary()) # Print the model summary

print("Training model...")

history = model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.05) # Train the model

print ("Model trained!")

# Plot the training and validation loss and accuracy
acc = history.history['accuracy'] # Get the training accuracy
val_acc = history.history['val_accuracy'] # Get the validation accuracy
loss = history.history['loss'] # Get the training loss
val_loss = history.history['val_loss'] # Get the validation loss


ax.plot(acc, label='acc' , color='blue', linestyle='solid') # Plot the training accuracy
ax.plot(val_acc, label='val_acc' , color='green' , linestyle='dashed') # Plot the validation accuracy
ax.plot(loss, label='loss', color = 'red' , linestyle='solid') # Plot the training loss
ax.plot(val_loss, label='val_loss', color = 'purple' , linestyle='dashed') # Plot the validation loss

plt.xlabel('Number of Epochs') # Set the x-axis label
plt.ylabel('Accuracy and Loss') # Set the y-axis label
plt.legend( loc='center right') # Set the legend

plt.show() # Display the plot

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test) # Evaluate the model on the test data
print('Test loss:', loss) # Print the loss
print('Test accuracy:', accuracy) # Print the accuracy

# Make predictions on the test set
y_pred = model.predict(X_test) # Make predictions on the test set

# Convert the predictions to a binary class (0 or 1)
y_pred = (y_pred > 0.5).astype(int) # Convert the predictions to a binary class (0 or 1)

# Compute the precision, recall, and F1 score
precision = precision_score(y_test, y_pred) # Compute the precision
recall = recall_score(y_test, y_pred) # Compute the recall
f1 = f1_score(y_test, y_pred) # Compute the F1 score

# Print the results
print("Precision:", precision) # Print the precision
print("Recall:", recall) # Print the recall
print("F1 score:", f1) # Print the F1 score

# Compute the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred) # Compute the confusion matrix
print("Confusion matrix:", confusion_matrix) # Print the confusion matrix

model.save('REALtest_1_2126_3jan2022.h5') # Save the model