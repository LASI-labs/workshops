from scipy import spatial
import numpy as np


def return_cosine_sorted(target_word_embedding,
                         all_words,
                         all_word_embeddings):
    words = []
    cosines = []
    for i in range(len(all_word_embeddings)):
        cosines.append(1 - spatial.distance.cosine(target_word_embedding, all_word_embeddings[i]))
        
    sorted_indexes = np.argsort(cosines)[::-1]
    
    return np.vstack((np.array(all_words)[sorted_indexes], np.array(cosines)[sorted_indexes])).T

def return_similar_words(word, all_words, all_word_embeddings, top_n=5):
    return return_cosine_sorted(return_embedding(word, all_words, all_word_embeddings),
                                all_words, all_word_embeddings)[1:top_n+1]


def return_embedding(word, all_words, all_word_embeddings):
    if(word in all_words):
        target_embedding_index = [i for i, s in enumerate(all_words) if word in s][0]
        return all_word_embeddings[target_embedding_index]
    else:
        return None
    
def return_analogy(source_word_1, source_word_2, target_word_1,
                   all_words, all_word_embeddings, top_n=5):
    
    em_sw_1 = return_embedding(source_word_1, all_words, all_word_embeddings)
    em_sw_2 = return_embedding(source_word_2, all_words, all_word_embeddings)
    em_tw_1 = return_embedding(target_word_1, all_words, all_word_embeddings)
    
    if((em_sw_1 is None) | (em_sw_2 is None) | (em_tw_1 is None)):
        return 0
    
    target_embedding = em_tw_1 + (em_sw_2 - em_sw_1)
    return return_cosine_sorted(target_embedding, all_words,
                                all_word_embeddings)[1:top_n+1]