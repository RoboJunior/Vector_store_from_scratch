from vector_store import Vector_store
import numpy as np


vector_store = Vector_store()
sentences = [
    "The sun sets behind the mountains.",
    "Cats enjoy lounging in the warm sunlight.",
    "Learning a new language opens doors to new opportunities.",
    "Music has the power to evoke strong emotions.",
    "The smell of freshly baked bread fills the room.",
    "Running in the morning energizes me for the day ahead.",
    "Love is the most powerful force in the universe.",
    "Trees sway gently in the breeze.",
    "Laughter is contagious and brings people together.",
    "The stars twinkle in the night sky.",
    "Raindrops patter softly against the windowpane.",
    "Reading transports you to different worlds.",
    "Kindness costs nothing but means everything.",
    "Time heals all wounds.",
    "Dreams are the seeds of reality.",
    "The ocean waves crash against the shore.",
    "Birds chirp merrily in the early morning.",
    "Happiness comes from within.",
    "Challenges make us stronger and more resilient.",
    "Forgiveness liberates the soul."
]


#Tokenization and vocabulary Building
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)


#Assing unqiue indices for words in the vocab
word_to_index = {word:i for i,word in enumerate(vocabulary)}

#Vectorization 
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]]+=1
    sentence_vectors[sentence] = vector


#Add vector to vectorstore
for sentence,vector in sentence_vectors.items():
    vector_store.add_vector(sentence,vector)


# Searching for the similarity
query_sentence = "I love coding"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]]+=1

similar_sentence = vector_store.similar_vector(query_vector,num_results=2)

print("Query Sentenece :",query_sentence)

print("Similar Sentences :")
for sentence , similarity in similar_sentence:
    print(f"Sentence : {sentence} , {similarity:.4f}")