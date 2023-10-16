# Implementing a learning rate scheduler
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import math, cast, float32
import tensorflow as tf

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
 
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps
 
    def __call__(self, step_num):
        # print(step_num)
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
 
        return tf.cast((self.d_model ** -0.5) * math.minimum(arg1, arg2), tf.float32)