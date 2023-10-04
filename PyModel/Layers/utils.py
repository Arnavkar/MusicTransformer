import tensorflow as tf

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

if __name__ == "__main__":
        padding_test = tf.constant([1,2,3,4,0,0,0])
        assert tf.math.reduce_all(tf.equal(padding_mask(padding_test), tf.constant([0,0,0,0,1,1,1],dtype=tf.float32)))

        lookahead_mask_test  = lookahead_mask(5)
        assert tf.math.reduce_all(tf.equal(
                lookahead_mask_test,
                tf.constant([[0,1,1,1,1],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]],dtype=tf.float32)))

