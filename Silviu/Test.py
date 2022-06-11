import tensorflow as tf
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()
