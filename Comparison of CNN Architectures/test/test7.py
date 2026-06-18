import pyhelayers
import utils
import h5py
import os
import numpy as np
##### For reproducibility
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)

# import activations
import sys
sys.path.append(os.path.join('.', 'data_gen'))
from activations import SquareActivation

PATH = os.path.join('data', 'net_mnist')
if not os.path.exists(PATH):
    os.makedirs(PATH)



utils.verify_memory()

print('Misc. initializations')

batch_size=500
he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
he_run_req.optimize_for_batch_size(16)

nn = pyhelayers.NeuralNet()
nn.encode_encrypt([os.path.join(PATH, "model6.json"), os.path.join(PATH, "model6.h5")], he_run_req)

context = nn.get_created_he_context()

with h5py.File(os.path.join(PATH, "x_test2.h5")) as f:
    x_test = np.array(f["x_test2"])
with h5py.File(os.path.join(PATH, "y_test2.h5")) as f:
    y_test = np.array(f["y_test2"])

# plain_samples, labels = utils.extract_batch(x_test, y_test, batch_size, 0)
batch_size=500
plain_samples, labels = utils.extract_batch(x_test, y_test, 500, 0)
print('Batch of size', batch_size, 'loaded')

model_io_encoder = pyhelayers.ModelIoEncoder(nn)
samples = pyhelayers.EncryptedData(context)
model_io_encoder.encode_encrypt(samples, [plain_samples])
print('Test data encrypted')

utils.start_timer()

predictions = pyhelayers.EncryptedData(context)
nn.predict(predictions, samples)

duration=utils.end_timer('predict')
utils.report_duration('predict per sample',duration/batch_size)

plain_predictions = model_io_encoder.decrypt_decode_output(predictions)
print('predictions',plain_predictions)

utils.assess_results(labels, plain_predictions)

