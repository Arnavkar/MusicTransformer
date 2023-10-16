import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

subprocess.run(['bash', 'neural_translator/run_test_train.sh'])