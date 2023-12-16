import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy

def check_shape(name,tensor,expectedshape):
        assert tensor.shape == expectedshape, f" {name} expected shape {expectedshape}, shape: {tensor.shape}"


def padding_mask(input):
        mask = tf.math.equal(input,0)
        mask = tf.cast(mask,tf.float32)
        #NOT SURE
        return mask[:,tf.newaxis,tf.newaxis,:]


def lookahead_mask(shape):
        # Mask out future entries by marking them with a 1.0
        return 1 - tf.linalg.band_part(tf.ones((shape,shape)), -1, 0)

#define the custom loss function
#@tf.function
def custom_loss(y_true,y_pred):
        # Create mask so that the zero padding values are not included in the computation of loss
        padding_mask = tf.math.logical_not(tf.equal(y_true, 0))
        padding_mask = tf.cast(padding_mask, tf.float32)

        # Compute a sparse categorical cross-entropy loss on the unmasked values - logits = True because we do not have the softmax in our model
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False) * padding_mask

        # Compute the mean loss over the unmasked values
        return tf.reduce_sum(loss) / tf.reduce_sum(padding_mask)


#Defining the accuracy function
#@tf.function
def custom_accuracy(y_true,y_pred):
        # Create mask so that the zero padding values are not included in the computation of accuracy
        padding_mask = tf.math.logical_not(tf.equal(y_true, 0))

        # Find equal prediction and target values, and apply the padding mask
        accuracy = tf.equal(tf.cast(y_true,tf.int64), tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(padding_mask, accuracy)

        # Cast the True/False values to 32-bit-precision floating-point numbers
        padding_mask = tf.cast(padding_mask, tf.float32)
        accuracy = tf.cast(accuracy, tf.float32)

        # Compute the mean accuracy over the unmasked values
        return tf.reduce_sum(accuracy) / tf.reduce_sum(padding_mask)

if __name__ == "__main__":
        # padding_test = tf.constant([[1,2,3,4,0,0,0]])
        # assert tf.math.reduce_all(tf.equal(padding_mask(padding_test), tf.constant([0,0,0,0,1,1,1],dtype=tf.float32)))

        # lookahead_mask_test  = lookahead_mask(5)
        # assert tf.math.reduce_all(tf.equal(
        #         lookahead_mask_test,
        #         tf.constant([[0,1,1,1,1],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]],dtype=tf.float32)))
        

        test_y_true = [1,276,43,57,2,0,0,0]
        test_y_pred = [1,276,43,2,0,0,0,0]

        loss = custom_loss(test_y_true,test_y_pred)
        print(loss)

        accuracy = custom_accuracy(test_y_true,test_y_pred)
        print(accuracy)


