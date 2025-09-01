import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"x train shape: {x_train.shape}")
print(f"y train shape: {y_train.shape}")



# preprocessing 
# (ResNetは基本RGB img (3channel) -> ResNetのfirst Convolutional Layerを1channel対応に変える)

# Normalization & Reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# model construction (簡易2層)

"""

基礎を学習する必要あり

"""
from tensorflow.keras import layers, models

def resnet_model():
    inputs = layers.Input(shape=(28,28,1))

    # 1st Conv
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # 残差ブロックを作る関数
    def res_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x
    
    # 2つの残差ブロック
    x = res_block(x, 32)
    x = res_block(x, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

model = resnet_model()
model.summary()


# compile & training
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, 
          validation_data=(x_test, y_test))

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("test accuracy: ", test_acc)

model.save("resnet_mnist.keras")
