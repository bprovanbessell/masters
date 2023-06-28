import json
import os

WORKSPACE_PATH = "/Users/bprovan/University/dissertation/mpd-master/SAPIEN"
MODEL_ID_JSON = "SAPIEN_ALL_CHILDREN.json"


json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
with open(json_file, "r") as f:
    data: dict = json.load(f)
    categories = list(data.keys())


print(categories)