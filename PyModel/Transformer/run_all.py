import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

subprocess.run(['bash', 'Transformer/run_all.sh'])