import json
import os
import numpy as np

WORKSPACE_PATH = "/Users/bprovan/University/dissertation/mpd-master/SAPIEN"
MODEL_ID_JSON = "SAPIEN_ALL_CHILDREN.json"
models_json = "SAPIEN_model_ids.json"
split_json_name = "UNSEEN_MODEL_SPLIT.json"

validation_split = 0.1
test_split = 0.2
seed = 44

json_file = os.path.join(WORKSPACE_PATH, models_json)
split_json = {}


with open(json_file, "r") as f:
    data: dict = json.load(f)
    categories = list(data.keys())

    for key, model_ids in data.items():
        split_dict = {}

        rng = np.random.default_rng(seed)
        dataset_size = len(model_ids)
        indices = list(range(dataset_size))
        val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
        test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

        print(val_split_index)
        print(test_split_index)

        train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

        print("lengths")
        print(len(train_indices), len(val_indices), len(test_indices))

        train_model_ids = [model_ids[i] for i in train_indices]
        val_model_ids = [model_ids[i] for i in val_indices]
        test_model_ids = [model_ids[i] for i in test_indices]

        split_dict['train'] = train_model_ids
        split_dict['val'] = val_model_ids
        split_dict['test'] = test_model_ids

        split_json[key] = split_dict


split_json_file_path = os.path.join(WORKSPACE_PATH, split_json_name)
with open(split_json_file_path, "w") as write_json_file:
    json.dump(split_json, write_json_file)


