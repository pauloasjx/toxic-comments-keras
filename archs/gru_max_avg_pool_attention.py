from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Dense,
    Input,
    Conv1D,
    GRU,
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
    RepeatVector,
    Permute,
    Activation,
)


def arch(max_len, max_features, embed_size, embedding_matrix):
    sequence_input = Input(shape=(max_len,))

    x = Embedding(
        max_features, embed_size, weights=[embedding_matrix], trainable=False
    )(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)

    conv1 = Conv1D(
        192, kernel_size=1, padding="valid", kernel_initializer="glorot_uniform"
    )(x)
    conv1_avg_pool = GlobalAveragePooling1D()(conv1)
    conv1_max_pool = GlobalMaxPooling1D()(conv1)

    x = concatenate([conv1_avg_pool, conv1_max_pool])

    attention = Dense(1, activation="tanh")(x)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(256)(attention)
    attention = Permute([2, 1])(attention)

    rep = merge([activations, attention], mode="mul")
    rep = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(rep)

    preds = Dense(6, activation="sigmoid")(rep)

    model = Model(sequence_input, preds)
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=["accuracy"]
    )

    return model
