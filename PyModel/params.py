import midi_processor.processor as sequence

#Why this embedding size??
# Look at https://saturncloud.io/blog/what-are-embeddings-in-pytorch-and-how-to-use-them/
# As a rule of thumb, the embedding size should be between the square root and the cube root 
# of the number of categories. For example, if we have a variable with 100 categories, the embedding size should be between 4 and 5.
loss_type = 'categorical_crossentropy'

#event_dim equal to 388
event_dim = sequence.RANGE_NOTE_ON + sequence.RANGE_NOTE_OFF + sequence.RANGE_TIME_SHIFT + sequence.RANGE_VEL

pad_token = event_dim
token_eos = event_dim + 1 
token_sos = event_dim + 2

#test params from tutorial - for translation task
baseline_test_params= {
    "num_heads": 8,  # Number of self-attention heads
    "embedding_dim":64,  # Dimensionality of the embedded queries and keys
    "model_dim":512,  # Dimensionality of the model final output
    "batch_size" :64,  # Batch size from the training process
    "seq_len" : 5, #sequence length
    "feed_forward_dim" : 2048,
    "batch_size" :64, #number of sequences in a batch
    "dropout_rate" : 0.1,
    "encoder_vocab_size" : 20,
    "num_encoder_layers" : 6,
    "decoder_vocab_size" : 20,
    "num_decoder_layers" : 6
}

class Params:
    def __init__(self, param_dict):
        self.num_heads = param_dict["num_heads"]
        self.embedding_dim = param_dict["embedding_dim"]
        self.model_dim = param_dict["model_dim"]
        self.batch_size = param_dict["batch_size"]
        self.seq_len = param_dict["seq_len"]
        self.feed_forward_dim = param_dict["feed_forward_dim"]
        self.dropout_rate = param_dict["dropout_rate"]
        self.encoder_vocab_size = param_dict["encoder_vocab_size"]
        self.num_encoder_layers = param_dict["num_encoder_layers"]
        self.decoder_vocab_size = param_dict["decoder_vocab_size"]
        self.num_decoder_layers = param_dict["num_decoder_layers"]
        
