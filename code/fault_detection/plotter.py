"""
Plot train vs validation loss and accuracies over epochs

Anything else that needs to be done
"""

import json

class Plotter():
    def __init__(self) -> None:
        pass

    def plot_train_val_acc():
        pass



def plot_latex_table_rows(res_json):
    with open(res_json, "r") as f:
        data: dict = json.load(f)
        categories = list(data.keys())

        for category in categories:
            cat_res = data[category]
            print("{} & {:.4f}\%  & {:.4f} \\".format(category, cat_res['accuracy'] * 100, cat_res['precision']))


if __name__ == "__main__":
    res_json = "/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/baseline_binary.json"
    plot_latex_table_rows(res_json)