import tensorflow as tf

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super(SaveCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)