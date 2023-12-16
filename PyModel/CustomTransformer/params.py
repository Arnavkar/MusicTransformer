
event_dim = 388 

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

#test params for model that uses model.fit
midi_test_params_v2 = {
    "num_heads": 4,  # Number of self-attention heads
    "key_dim": 128,  # Dimensionality of linearly projected queries and keys
    "value_dim": 128,  # Dimensionality of linearly projected values
    "model_dim":512,  # Dimensionality of the model final output
    "batch_size" :64,  # Batch size from the training process
    "l_r":0.001, #Learning rate when not useing a learning rate scheduler
    "feed_forward_dim" : 2048, #Dimensionality of the hidden layer in the feed forward network
    "dropout_rate" : 0.1, #Dropout rate for the dropout layers 
    "encoder_vocab_size" : event_dim, #Vocab size of the input tokens - same in encoder and decoder
    "num_encoder_layers" : 2, #Number of encoder stacks
    "decoder_vocab_size" : event_dim,
    "num_decoder_layers" : 2,
    "epochs":200, #Number of epochs to train for
    "beta_1":0.9, #Adam optimizer beta_1
    "beta_2":0.98, #Adam optimizer beta_2
    "epsilon":1e-8, #Adam optimizer epsilon
    "encoder_seq_len": 1042, #Encoder input sequence length -  context sequence
    "decoder_seq_len": 1044, #Decoder sequence length - larger by two because we add sos and eos tokens
    "max_seq_len":1044, #Maximum sequence length - used for positional encoding
    "pad_token" : 0, #Padding token
    "token_sos" : 1, #Start of sequence token
    "token_eos" : 2, #End of sequence token 
    "debug":True,
    "steps_per_epoch": 1000, #Number of steps per epoch - useful for data generators
    "save_freq": 10, #Save frequency
    "seed":236, #Random seed for reproduceable results
}

class Params:
    def __init__(self, param_dict):
        #Given a dictionary of parameters, set the parameters as attributes
        for key, value in param_dict.items():
            setattr(self, key, value)

    def print_params(self):
        #Print all the parameters
        all_attrs = vars(self)
        for key, value in all_attrs.items():
            print(key, ":", value)

    def get_params(self):
        #Return all the parameters
        return vars(self)
    
    def __repr__(self):
        return '<class Params has variables: {}>'.format(vars(self))

if __name__ == "__main__":
    params = Params(midi_test_params_v2)
    params.print_params()


        
