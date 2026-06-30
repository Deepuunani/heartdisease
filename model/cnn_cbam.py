import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Multiply,
    Reshape,
    Add
)
from tensorflow.keras.models import Model


# -----------------------------
# Channel Attention
# -----------------------------
def channel_attention(inputs, ratio=8):

    channels = inputs.shape[-1]

    avg_pool = GlobalAveragePooling1D()(inputs)

    dense1 = Dense(channels // ratio,
                   activation='relu')(avg_pool)

    dense2 = Dense(channels,
                   activation='sigmoid')(dense1)

    scale = Reshape((1, channels))(dense2)

    return Multiply()([inputs, scale])


# -----------------------------
# Residual Block
# -----------------------------
def residual_block(x, filters):

    shortcut = x

    x = Conv1D(filters,
               kernel_size=3,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters,
               kernel_size=3,
               padding="same")(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters,
                          kernel_size=1,
                          padding="same")(shortcut)

    x = Add()([x, shortcut])

    x = Activation("relu")(x)

    return x


# -----------------------------
# CBAM Block
# -----------------------------
def cbam_block(x):

    x = channel_attention(x)

    return x


# -----------------------------
# Build Model
# -----------------------------
def build_model(input_shape=(360, 1),
                num_classes=7):

    inputs = Input(shape=input_shape)

    x = Conv1D(
        32,
        kernel_size=7,
        padding="same",
        activation="relu")(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    # Residual Block 1
    x = residual_block(x, 32)

    x = cbam_block(x)

    # Residual Block 2
    x = residual_block(x, 64)

    x = MaxPooling1D(pool_size=2)(x)

    x = cbam_block(x)

    # Residual Block 3
    x = residual_block(x, 128)

    x = MaxPooling1D(pool_size=2)(x)

    x = cbam_block(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(256,
              activation="relu")(x)

    x = Dropout(0.5)(x)

    x = Dense(128,
              activation="relu")(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        num_classes,
        activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    model = build_model()

    model.summary()