import numpy as np

np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import random
import aidrtokenize as aidrtokenize
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

random.seed(1337)


def file_exist(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False


def read_stop_words(file_name):
    if (not file_exist(file_name)):
        print("Please check the file for stop words, it is not in provided location " + file_name)
        sys.exit(0)
    stop_words = []
    with open(file_name, 'rU') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            stop_words.append(line)
    return stop_words;


stop_words_file = "stop_words/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)


def read_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """
    data = []
    labels = []
    with open(dataFile, 'rb') as f:
        next(f)
        for line in f:
            line = line.decode(encoding='utf-8', errors='strict')
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            txt = row[3].strip().lower()
            txt = aidrtokenize.tokenize(txt)
            label = row[6]
            if (len(txt) < 1):
                print (txt)
                continue
            data.append(txt)
            labels.append(label)

    data_shuf = []
    lab_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        lab_shuf.append(labels[i])

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab_shuf)
    labels = list(le.classes_)

    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data_shuf)
    sequences = tokenizer.texts_to_sequences(data_shuf)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data tensor:', data.shape)
    return data, y, le, labels, word_index, tokenizer

def generate_data_file(data_file,label,delim,labels):                                                                
                                               
    image_len = len(data_file)                                                  
    all_images = np.empty([image_len, 224, 224, 3],dtype="int8")                
    #list_of_pics = list()                                                      
    all_labels = []                                                             
                                                                                
    for i in range(image_len):                                                  
           img = image.load_img(data_file[i], target_size=(224, 224))
           #print(img)                                                          
           img = image.img_to_array(img)                                        
           img = np.expand_dims(img, axis=0)                                    
           img = preprocess_input(img)                                          
           lab = label[i]                                                  
           #list_of_pics.append(np.asarray(img))                                
           all_images[i, :, :, :] = img                                         
           all_labels.append (lab)                                              
    num_class=len(set(labels))  

    #print(all_labels.shape)                                                    
    #all_images = all_images.astype('int8')                                     
    #all_images = np.array(list_of_pics)                                        
    #print(all_images.shape)                                                    
    print("num classes: "+str(num_class))                                       
    le = preprocessing.LabelEncoder()                                           
    le.fit(labels)
    y=le.transform(all_labels)                                              
    y=np.asarray(y)                                                             
    all_labels = to_categorical(y, num_classes=num_class)                       
    print(all_labels.shape)                                                     
    return all_images,all_labels,le,num_class

def read_dev_data(text_file,label,tokenizer, MAX_SEQUENCE_LENGTH,delim,labels):
    """
    Prepare the data
    """
    id_list=[]
    data = []
    lab = []
    for i in range(len(text_file)):
        txt = aidrtokenize.tokenize(text_file[i])
        text = " ".join(txt)
        if (len(txt) < 1):
            print ("TEXT SIZE:" + txt)
            continue
        data.append(text)
        lab.append(label[i])

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    yL=le.transform(lab)
    labels = list(le.classes_)

    label = yL.tolist()
    yC = len(set(labels))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data, y, le, labels


def load_embedding(fileName):
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index;


def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            if(word in model):
                embedding_vector = model[word][0:EMBEDDING_DIM]  # embeddings_index.get(word)
                embedding_matrix[i] = np.asarray(embedding_vector, dtype=np.float32)
            else:
                rng = np.random.RandomState()
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                #embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
        except KeyError:
            try:
                print(word.encode('utf-8') +" not found... assigning random")
                rng = np.random.RandomState()
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                #embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
            except KeyError:
                continue
    return embedding_matrix;


def str_to_indexes(s):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    input_size = 1014
    length = input_size
    alphabet_size = len(alphabet)
    char_dict = {}  # Maps each character to an integer
    # self.no_of_classes = num_of_classes
    for idx, char in enumerate(alphabet):
        char_dict[char] = idx + 1
    length = input_size

    """
    Convert a string to character indexes based on character dictionary.
    Args:
        s (str): String to be converted to indexes
    Returns:
        str2idx (np.ndarray): Indexes of characters in s
    """
    s = s.lower()
    max_length = min(len(s), length)
    str2idx = np.zeros(length, dtype='int64')
    for i in range(1, max_length + 1):
        c = s[-i]
        if c in char_dict:
            str2idx[i - 1] = char_dict[c]
    return str2idx