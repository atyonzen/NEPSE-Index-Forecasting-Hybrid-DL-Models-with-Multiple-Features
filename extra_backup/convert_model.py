import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
# import tensorflowjs as tfjs

# Load the model
model = keras.models.load_model('best_model.keras')

# Print the model summary
print(model.summary())

# Convert the model into TF.js Format model
# tfjs.converters.save_keras_model(model, '/')