
import os
import numpy as np
import requests
import operator
import json
from PIL import Image


env_data_dir = "DATA_DIR"

def main(data_dir):
	print("data dir:",data_dir)
	files = os.listdir(data_dir)
	print("get files:",files)
	first = files[0]
	image = Image.open(os.path.join(data_dir, first))
	img = image.resize((224,224))
	raw_data = np.array(img)/255.0

	# Normalization
	data = raw_data - np.array([0.485,0.456,0.406])
	data = np.divide(data, np.array([0.229,0.224,0.225]))
	data = np.moveaxis(data, 2, 0)
	data = np.expand_dims(data, 0)

	# send request
	r = requests.post(
		'http://127.0.0.1:7788/predict',
		data = json.dumps({"instances":[{"gpu_0/data_0": data.astype(np.float32).tolist()}]}))

	# Return result
	if r.status_code != 200:
		print("error")
		exit()

	result_json = json.loads(r.text)
	result = result_json['predictions'] if 'predictions' in result_json else result_json['outputs']
	idx, _ = max(enumerate(result[0][0]), key = operator.itemgetter(1))
	print("idx:",idx)


if __name__ == "__main__":
	data_dir = os.getenv(env_data_dir)
	main(data_dir)