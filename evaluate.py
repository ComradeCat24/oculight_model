import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu


def word_for_id(word_id: int, tokenizer: Tokenizer, default='Unknown'):
    if word_id < 1 or word_id > len(tokenizer.word_index):
        raise ValueError(
            f"word_id should be between 1 and {len(tokenizer.word_index)}")
    return tokenizer.index_word.get(word_id, default)


def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # initialize empty list to keep track of predicted words
    predicted_words = []
    i = 0
    # use while loop
    while i < max_length:
        try:
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            yhat = model.predict([photo, sequence], verbose=0)
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = tokenizer.index_word[yhat]
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # keep track of predicted words
            predicted_words.append(word)
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
            i += 1
        except Exception as e:
            print(str(e))
            return 'Error Generating Description'
    # remove 'startseq' from the predicted words
    predicted_words = predicted_words[1:]
    # join the predicted words to form a string
    final_text = " ".join(predicted_words)
    # remove 'endseq' from the final string
    final_text = final_text.replace('endseq', '')
    return final_text, predicted_words


def calculate_bleu_scores(model, descriptions, photos, tokenizer, max_length):
    """
    Calculates BLEU scores for the model's generated descriptions.
    """
    bleu_weights = [(1.0, 0, 0, 0), (0.5, 0.5, 0, 0),
                    (0.3, 0.3, 0.3, 0), (0.25, 0.25, 0.25, 0.25)]
    actual, predicted = [], []
    for i, (key, desc_list) in enumerate(tqdm(descriptions.items(), "Evaluating")):
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        actual.append([d.split() for d in desc_list])
        predicted.append(yhat.split())
    bleu_scores = {f"BLEU-{i+1}": corpus_bleu(actual, predicted, weights=w)
                   for i, w in enumerate(bleu_weights)}
    return bleu_scores
