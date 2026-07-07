import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Models
encoder_model = load_model("encoder_model.keras")
decoder_model = load_model("decoder_model.keras")

# Load Tokenizers
with open("eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer, max_eng = pickle.load(f)

with open("tel_tokenizer.pkl", "rb") as f:
    tel_tokenizer, max_tel = pickle.load(f)

reverse_word_index = {v: k for k, v in tel_tokenizer.word_index.items()}

start_token = tel_tokenizer.word_index["startseq"]
end_token = tel_tokenizer.word_index["endseq"]


def translate(sentence):

    sequence = eng_tokenizer.texts_to_sequences([sentence.lower()])

    sequence = pad_sequences(
        sequence,
        maxlen=max_eng,
        padding="post"
    )

    states = encoder_model.predict(sequence, verbose=0)

    target_seq = np.array([[start_token]])

    translated_sentence = []

    while True:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states,
            verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_word = reverse_word_index.get(sampled_token_index, "")

        if sampled_word == "endseq":
            break

        if sampled_word != "startseq":
            translated_sentence.append(sampled_word)

        target_seq = np.array([[sampled_token_index]])

        states = [h, c]

        if len(translated_sentence) > max_tel:
            break

    return " ".join(translated_sentence)


if __name__ == "__main__":

    text = input("Enter English Sentence: ")

    print(translate(text))