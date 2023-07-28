"""
Should import the image, can just do colour avaeraging. Basically if it is not totally white, then the anomalous part is visible
so we can use that image. Otherwise it is not visible so we should not use it.

So just generate mask lists for each category... and then move them to make a new datastet
"""

from PIL import Image
import numpy as np
import os
import glob
import json
import shutil
""

# for im in im_list:

im = "/Users/bprovan/Desktop/test6.png"

imarr=np.array(Image.open(im),dtype=np.float64)

im_mean = np.mean(imarr)

if im_mean < 255.0:
    print("part detected!")


def make_occ_json():
    occlusion_dict = {}

    base_folder = "/Users/bprovan/University/dissertation/datasets/images_mask"
    json_file = "/Users/bprovan/University/dissertation/masters/code/occlusion.json"

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                    'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                    'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                    'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                    'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']

    # categories = ["WashingMachine"]

    for category in categories:
        instance_dir_list = glob.glob(os.path.join(base_folder, category, "*"))
        occlusion_dict[category] = {}

        for instance_dir in instance_dir_list:
            instance_id = instance_dir.split('/')[-1]
            im_list = glob.glob(os.path.join(instance_dir, "*.png"))
            occlusion_dict[category][instance_id] = {}

            for im_file in im_list:
                im_id = im_file.split('/')[-1]
                if "orig" not in im_file:

                    imarr=np.array(Image.open(im_file),dtype=np.float64)

                    im_mean = np.mean(imarr)

                    if im_mean < 255.0:
                        # print("part detected!")
                        label = 1
                    else:
                        label = 0

                    occlusion_dict[category][instance_id][im_id] = label

        with open(json_file, 'w') as fp:
            json.dump(occlusion_dict, fp)

def make_occ_json2():
    occlusion_dict = {}

    base_folder = "/Users/bprovan/University/dissertation/datasets/images_mask"
    json_file = "/Users/bprovan/University/dissertation/masters/code/occlusion2.json"

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                    'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                    'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                    'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                    'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']

    # categories = ["WashingMachine"]

    for category in categories:
        instance_dir_list = glob.glob(os.path.join(base_folder, category, "*"))
        occlusion_dict[category] = {}

        for instance_dir in instance_dir_list:
            instance_id = instance_dir.split('/')[-1]
            im_list = glob.glob(os.path.join(instance_dir, "*.png"))
            occlusion_dict[category][instance_id] = {}

            for im_file in im_list:
                im_id = im_file.split('/')[-1]
                view_num = im_id.split('_')[-1][0:-4]

                # 0 -> 0
                # 2 -> 1
                # 4 -> 2
                if "orig" not in im_file and int(view_num) % 2 == 0:

                    view_num_2 = int(int(view_num) / 2)
                    view_num_2_label = str(view_num_2) + ".png"
                    im_id_2 = "_".join(im_id.split('_')[0:-1]) + "_" + view_num_2_label

                    imarr=np.array(Image.open(im_file),dtype=np.float64)

                    im_mean = np.mean(imarr)

                    if im_mean < 255.0:
                        # print("part detected!")
                        label = 1
                    else:
                        label = 0

                    occlusion_dict[category][instance_id][im_id_2] = label

        with open(json_file, 'w') as fp:
            json.dump(occlusion_dict, fp)


def move_occluded_images():

    images_folder = "/Users/bprovan/University/dissertation/datasets/images_ds_v0"
    json_file = "/Users/bprovan/University/dissertation/masters/code/occlusion2.json"
    occluded_folder = "/Users/bprovan/University/dissertation/datasets/images_ds_v0_occluded"


    with open(json_file, "r") as f:
        data: dict = json.load(f)

        for category in data.keys():
            for instance in data[category].keys():

                # copy the original files
                new_dir_path = os.path.join(occluded_folder, category, instance)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)

                dir_path = os.path.join(images_folder, category, instance)
                orig_ims = glob.glob(os.path.join(dir_path, "orig*"))

                for im_path in orig_ims:
                    im_id = im_path.split('/')[-1]
                    new_im_path = os.path.join(new_dir_path, im_id)

                    shutil.move(im_path, new_im_path)

                for im_id, label in data[category][instance].items():

                    if label == 1:
                        im_path = os.path.join(dir_path, im_id)
                        new_im_path = os.path.join(new_dir_path, im_id)
                        shutil.move(im_path, new_im_path)


if __name__ == "__main__":
    # make_occ_json2()
    move_occluded_images()