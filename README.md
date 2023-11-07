# MusicTransformer
A music transformer built with Tensorflow based on Huang et. al 2018 and interfaced with Tensorflow.js and Max/MSP for real time performance - In Progress! 

Work so far:
- Built a custom dataset class to serve up batches of encoded sequences to the model
- Built a baseline transformer model by subclassing Keras.Model that can be trained on encoded midi sequences with a fully custom training loop
- Altered model to override keras.Model methods and use built-in model.fit(), model.evaluate()
- Created V2 of dataset class to ensure all data is covered over each epoch, previous batches taken via random sampling
- Currently working on building a custom generator for the data to reduce local memory requirements + distributing training across multiple GPUs
