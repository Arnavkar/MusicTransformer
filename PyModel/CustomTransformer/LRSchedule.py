# Implementing a learning rate scheduler
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import math
import tensorflow as tf

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter - taken directly from Vaswani et al.
        arg1 = tf.cast(step_num,tf.float32) ** tf.cast(-0.5, tf.float32)
        arg2 = tf.cast(step_num, tf.float32) * tf.cast((self.warmup_steps ** -1.5), tf.float32)
 
        return tf.cast((self.d_model ** -0.5) * math.minimum(arg1, arg2), tf.float32)