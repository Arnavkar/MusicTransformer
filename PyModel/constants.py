import midi_processor.processor as sequence

max_seq=2048
learn_rate = 0.001

#Why this embedding size??
# Look at https://saturncloud.io/blog/what-are-embeddings-in-pytorch-and-how-to-use-them/
# As a rule of thumb, the embedding size should be between the square root and the cube root 
# of the number of categories. For example, if we have a variable with 100 categories, the embedding size should be between 4 and 5.

embedding_dim = 256
num_attention_layer = 6
batch_size = 10
loss_type = 'categorical_crossentropy'
#event_dim equal to 388
event_dim = sequence.RANGE_NOTE_ON + sequence.RANGE_NOTE_OFF + sequence.RANGE_TIME_SHIFT + sequence.RANGE_VEL
pad_token = event_dim

token_ss = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3