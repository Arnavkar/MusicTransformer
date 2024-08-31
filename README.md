# MusicTransformer
A music transformer built with Tensorflow based on Huang et. al 2018 and interfaced with Tensorflow.js and Max/MSP for real time performance - Project is currently on hold!

Work so far:
- Built different variants of custom dataset classes to serve up batches of encoded sequences to the model
- Built a baseline transformer model by subclassing Keras.Model that can be trained on encoded midi sequences with a fully custom training loop. The Transformer model I coded from scratch was unsuccessful (to revisit)
- Altered model to override keras.Model methods and use built-in model.fit(), model.evaluate()
- Created V2 of dataset classes to ensure all data is covered over each epoch, previous batches taken via random sampling
- Built Custom data generators to reduce local memory requirements (only working with 16GB Ram) + distributing training across multiple GPUs with CUDA

#General Thoughts
- The MAX/MSP interface has potential, especially since it is familiar to many musicians. However, it is likely unfeasible to host the model within the NodeJS runtime, especially a model of the size I attempted to build. AWS Sagemaker and API calls could be an alternative, but would greatly increase latency and reduce real-time viability
- The model overall was unsuccessful largely because of lack of resources for model training (Training over all the data would take 300 hours for one epoch). I'd like to experiment with much smaller models and alternative datasets, maybe even a genetic algorithm rather than a full model. However, the analysis yields that this method could have some success as the model does learn certain traits. To see more, check out the paper and read the results section (https://digitalcommons.bard.edu/senproj_f2023/54/)
