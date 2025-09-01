from keras import models

model = models.load_model('resnet_mnist.h5')
model.save('resnet_mnist.keras')