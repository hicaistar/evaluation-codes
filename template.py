import os
import json
import argparse
import requests

def metrics(response):
    return {'accuracy':'0.9'}

def load_data(data_dir):
    files = os.listdir(data_dir)
    return files[0]

def execute_evaluation(evaluation):
    # add user code here:
    # 1. data = load_data(evaluation.data)
    # 2. payload = {"instances":[{"input": data.astype(np.float32).tolist()}]}
    # 3. response = evaluation.post_request(payload)
    # 4. result = metrics(response)
    # 5. evaluation.write_output(result)

class Evaluation:
    def __init__(self, function, data, server, output):
        self.name = function
        self.server = server
        self.data = data
        self.output = output

    def post_request(self,payload):
        response = requests.post(self.server, payload)
        if response.status_code == 200:
            result_json = json.loads(response.text)
            result = result_json['predictions'] if 'predictions' in result_json else result_json['outputs']
            return result
        else:
            print("serving error:",response.text)
            return ''

    def write_output(self,result):
        res = json.dumps(result)
        out = {
            "name": self.name,
            "result":res,
        }
        filename = ("%s.json" % self.name)
        file = os.path.join(self.output,filename)
        with open(file, 'w') as f:
            json.dump(out, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process args for evaluation')
    parser.add_argument("--output", "-o", help="set output directory")
    parser.add_argument("--data", "-d", help="set dataset directory")
    parser.add_argument("--name","-n", help="set function name")
    parser.add_argument("--server","-s", help="set serving server address")
    args = parser.parse_args()

    server = "http://{s}/predict".format(s=args.server)

    evaluation = Evaluation(args.name, args.data, server, args.output)

    execute_evaluation(evaluation)
