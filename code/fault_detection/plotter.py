"""
Plot train vs validation loss and accuracies over epochs

Anything else that needs to be done
"""
import numpy as np
import json

class Plotter():
    def __init__(self) -> None:
        pass

    def plot_train_val_acc():
        pass



def plot_latex_table_rows(res_json, brev=False):
    accuracies = []
    precisions = []
    with open(res_json, "r") as f:
        data: dict = json.load(f)
        categories = list(data.keys())

        if brev:
            # hand chosen good and bad categories
            categories = ['KitchenPot', 'USB', 'Eyeglasses', 'Laptop', 'Globe', 
                          'Sitting Furniture', 'Table', 'Kettle', 'Storage Furniture', 'WashingMachine']

        for category in categories:
            cat_res = data[category]
            print("{} & {:.4f}\%  & {:.4f} \\".format(category, cat_res['accuracy'] * 100, cat_res['precision']))
            accuracies.append(cat_res['accuracy'] * 100)
            precisions.append(cat_res['precision'])
        print("{} & {:.4f}\%  & {:.4f} \\".format("Mean", np.mean(accuracies), np.mean(precisions)))


if __name__ == "__main__":
    # res_json = "/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/baseline_binary_occ.json"
    res_json = "/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/results/siamese_res_occ.json"
    # res_json = "/Users/bprovan/University/dissertation/masters/code/fault_detection/logs/results/multiview_standard_occ_finetune.json"

    plot_latex_table_rows(res_json, brev=False)