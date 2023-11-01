#Why this embedding size??
# Look at https://saturncloud.io/blog/what-are-embeddings-in-pytorch-and-how-to-use-them/
# As a rule of thumb, the embedding size should be between the square root and the cube root 
# of the number of categories. For example, if we have a variable with 100 categories, the embedding size should be between 4 and 5.

import midi_neural_preprocessor.processor as sequence

#event_dim equal to 388
event_dim = sequence.RANGE_NOTE_ON + sequence.RANGE_NOTE_OFF + sequence.RANGE_TIME_SHIFT + sequence.RANGE_VEL

#test params from tutorial - for translation task
baseline_test_params= {
    "num_heads": 8,  # Number of self-attention heads
    "key_dim": 64,  # Dimensionality of linearly projected queries and keys
    "value_dim": 64,  # Dimensionality of linearly projected values
    "model_dim":512,  # Dimensionality of the model final output
    "batch_size" :64,  # Batch size from the training process
    "feed_forward_dim" : 2048,
    "dropout_rate" : 0.2,
    "encoder_vocab_size" : 20,
    "num_encoder_layers" : 6,
    "decoder_vocab_size" : 20,
    "num_decoder_layers" : 6,
    "epochs":20,
    "beta_1":0.9,
    "beta_2":0.98,
    "epsilon":1e-9,
    "encoder_seq_len": 5,
    "decoder_seq_len": 5,
    "debug":True,
    "l_r":0.001,
}


#test params from tutorial - for translation task
midi_test_params_v1 = {
    "num_heads": 8,  # Number of self-attention heads
    "key_dim": 64,  # Dimensionality of linearly projected queries and keys
    "value_dim": 64,  # Dimensionality of linearly projected values
    "model_dim":512,  # Dimensionality of the model final output
    "batch_size" :30,  # Batch size from the training process
    "l_r":0.001,
    "feed_forward_dim" : 2048,
    "dropout_rate" : 0.1,
    "encoder_vocab_size" : event_dim,
    "num_encoder_layers" : 6,
    "decoder_vocab_size" : event_dim,
    "num_decoder_layers" : 6,
    "epochs":50,
    "beta_1":0.9,
    "beta_2":0.98,
    "epsilon":1e-9,
    "encoder_seq_len": 1042,
    "decoder_seq_len": 1042,
    "pad_token" : 0,
    "token_sos" : 1,
    "token_eos" : 2,
    "debug":True,
    "record_data_stats":False,

}


class Params:
    def __init__(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)

    def print_params(self):
        all_attrs = vars(self)
        for key, value in all_attrs.items():
            print(key, ":", value)

    def get_params(self):
        return vars(self)
    
    def __repr__(self):
        return '<class Params has variables: {}>'.format(vars(self))

if __name__ == "__main__":
    params = Params(midi_test_params_v1)
    params.print_params()


        
