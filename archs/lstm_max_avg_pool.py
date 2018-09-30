from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Dense,
    Input,
    Conv1D,
    LSTM,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    MaxPooling1D,
    Add,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    concatenate,
    SpatialDropout1D,
)


def arch(max_len, max_features, embed_size, embedding_matrix):
    sequence_input = Input(shape=(max_len,))

    x = Embedding(
        max_features, embed_size, weights=[embedding_matrix], trainable=False
    )(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)

    conv1 = Conv1D(
        192, kernel_size=1, padding="valid", kernel_initializer="glorot_uniform"
    )(x)
    conv1_avg_pool = GlobalAveragePooling1D()(conv1)
    conv1_max_pool = GlobalMaxPooling1D()(conv1)

    x = concatenate([conv1_avg_pool, conv1_max_pool])
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    preds = Dense(6, activation="sigmoid")(x)

    model = Model(sequence_input, preds)
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=["accuracy"]
    )

    return model
