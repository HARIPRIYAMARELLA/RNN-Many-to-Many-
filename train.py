import pandas as pd
import numpy as np
import pickle
import re

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense

# -----------------------------
# Read Dataset
# -----------------------------

df = pd.read_csv(
    "tel.txt",
    sep="\t",
    header=None,
    usecols=[0,1],
    names=["english","telugu"]
)

# -----------------------------
# Clean Text
# -----------------------------

def clean(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z0-9\u0C00-\u0C7F\s]", "", text)

    return text.strip()

df["english"] = df["english"].apply(clean)
df["telugu"] = df["telugu"].apply(clean)

# -----------------------------
# Add Start and End Tokens
# -----------------------------

df["telugu"] = df["telugu"].apply(
    lambda x: "startseq " + x + " endseq"
)

# -----------------------------
# Tokenizers
# -----------------------------

eng_tokenizer = Tokenizer()

eng_tokenizer.fit_on_texts(df["english"])

tel_tokenizer = Tokenizer(filters="")

tel_tokenizer.fit_on_texts(df["telugu"])

# -----------------------------
# Convert To Sequence
# -----------------------------

encoder_input = eng_tokenizer.texts_to_sequences(
    df["english"]
)

decoder_input = tel_tokenizer.texts_to_sequences(
    df["telugu"]
)

max_eng = max(len(i) for i in encoder_input)

max_tel = max(len(i) for i in decoder_input)

encoder_input = pad_sequences(
    encoder_input,
    maxlen=max_eng,
    padding="post"
)

decoder_input = pad_sequences(
    decoder_input,
    maxlen=max_tel,
    padding="post"
)

eng_vocab = len(eng_tokenizer.word_index)+1

tel_vocab = len(tel_tokenizer.word_index)+1

print("English Vocabulary :", eng_vocab)
print("Telugu Vocabulary :", tel_vocab)

print("English Shape :", encoder_input.shape)
print("Telugu Shape :", decoder_input.shape)

# Save Tokenizers

pickle.dump(
    (eng_tokenizer,max_eng),
    open("eng_tokenizer.pkl","wb")
)

pickle.dump(
    (tel_tokenizer,max_tel),
    open("tel_tokenizer.pkl","wb")
)
# -----------------------------
# Prepare Decoder Output
# -----------------------------

decoder_output = np.zeros(
    (
        len(decoder_input),
        max_tel,
        tel_vocab
    ),
    dtype="float32"
)

for i in range(len(decoder_input)):

    for t in range(1, max_tel):

        word = decoder_input[i][t]

        if word != 0:

            decoder_output[i, t-1, word] = 1

# -----------------------------
# Encoder
# -----------------------------

encoder_inputs = Input(shape=(max_eng,))

encoder_embedding = Embedding(
    eng_vocab,
    128
)(encoder_inputs)

encoder = LSTM(
    256,
    return_state=True
)

encoder_outputs, state_h, state_c = encoder(
    encoder_embedding
)

encoder_states = [state_h, state_c]

# -----------------------------
# Decoder
# -----------------------------

decoder_inputs = Input(shape=(max_tel,))

decoder_embedding = Embedding(
    tel_vocab,
    128
)(decoder_inputs)

decoder_lstm = LSTM(
    256,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_dense = Dense(
    tel_vocab,
    activation="softmax"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

# -----------------------------
# Model
# -----------------------------

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------

model.fit(
    [encoder_input, decoder_input],
    decoder_output,
    batch_size=16,
    epochs=200,
    validation_split=0.2
)

# -----------------------------
# Save Model
# -----------------------------

model.save("model.keras")

print("\nTraining Completed Successfully")
# -----------------------------
# Save Encoder Inference Model
# -----------------------------

encoder_model = Model(
    inputs=encoder_inputs,
    outputs=encoder_states
)

encoder_model.save("encoder_model.keras")
# -----------------------------
# Decoder Inference Model
# -----------------------------

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))

decoder_embedding2 = Embedding(
    input_dim=tel_vocab,
    output_dim=128
)(decoder_inputs_single)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_embedding2,
    initial_state=decoder_states_inputs
)

decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

decoder_model.save("decoder_model.keras")
decoder_model.save("decoder_model.keras")

print("Decoder Model Saved Successfully")