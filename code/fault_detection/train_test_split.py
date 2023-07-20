# Generate pairs, then split the query images into the 

# Have directory for anchor images, directory for query images
# Then split query images into train, test, validation

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from custom_dataset import ViewCombDataset, ViewCombDifferenceDataset
from unseen_model_anomoly_dataset import ViewCombUnseenModelDataset
from trainer import train_multiview_epoch
from eval import evaluate_multiview, ModelSaver
from logger import MetricLogger
import os
import glob
import shutil
import json


def gen_original_pairs(category):
    batch_size = 8
    validation_split = 0.1
    test_split = 0.2
    shuffle_dataset = True
    epochs = 10
    seed = 44

    # weights = ResNet50_Weights.IMAGENET1K_V2
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"

    print(device)
    missing_parts_base_dir = '/Users/bprovan/University/dissertation/datasets/images_ds_v0'
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v2 = '/Users/bprovan/University/dissertation/datasets/images_ds_v2'

    ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                            n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)
    test_ds = ViewCombDataset(img_dir=missing_parts_base_dir_v1, category=category, 
                            n_views=12, n_samples=12, transforms=preprocess, train=False, seed=seed)

    # Creating data indices for training and validation splits:
    rng = np.random.default_rng(seed)
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    val_split_index = int(np.floor(dataset_size * (1-(validation_split + test_split))))
    test_split_index = int(np.floor(dataset_size * (1 - (test_split))))

    print(val_split_index)
    print(test_split_index)
    if shuffle_dataset:
        rng.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:val_split_index], indices[val_split_index:test_split_index], indices[test_split_index:]

    print("lengths")
    print(len(train_indices), len(val_indices), len(test_indices))

    pairs = ds.pairs

    val_pairs = [pairs[x] for x in val_indices]
    test_pairs = [pairs[x] for x in test_indices]


    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Should work for the basic train test split
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
                                    sampler=train_sampler, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        sampler=test_sampler)


    # just generate list of test and val query images

    val_query_images = [val_pairs[x][1] for x in range(len(val_pairs))]
    test_query_images = [test_pairs[x][1] for x in range(len(test_pairs))]
    res_json = {category: {"test_query_images": test_query_images,
                "val_query_iamges": val_query_images }}
    
    # print(res_json)
    return res_json





# so the anchor views, and the query views are distinct
def split_by_30_degrees():
    missing_parts_base_dir_v1 = '/Users/bprovan/University/dissertation/datasets/images_ds_v1'
    missing_parts_base_dir_v2 = '/Users/bprovan/University/dissertation/datasets/images_ds_v2'
    all_imgs = glob.glob(os.path.join(missing_parts_base_dir_v1, "*/*/*.png"))

    for img_path in all_imgs:
        l = img_path.split('/')

        view_num = int(img_path.split('/')[-1].split('_')[-1][0:-4])


        if view_num % 2 == 0:
            # anchor view
            dir_path = os.path.join(missing_parts_base_dir_v2, "anchor_images", l[-3], l[-2])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            img_v2 = os.path.join(missing_parts_base_dir_v2, "anchor_images", l[-3], l[-2], l[-1])
            shutil.copy(img_path, img_v2)
        else:
            dir_path = os.path.join(missing_parts_base_dir_v2, "query_images", l[-3], l[-2])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            img_v2 = os.path.join(missing_parts_base_dir_v2, "query_images", l[-3], l[-2], l[-1])
            shutil.copy(img_path, img_v2)


# def move_query_imgs(res_json):
#     with open(res_json, "r") as f:
#         data: dict = json.load(f)

#         for key, item in data.items():
#             for split, imgs in item.items():



# ------------ 
# for each instance, choose 1 query image for validation and 2 for test (random) for each class

def make_img_split(category, seed):
    missing_parts_base_dir_v2 = '/Users/bprovan/University/dissertation/datasets/images_ds_v2/query_images/'


    rng = np.random.default_rng(seed)
    img_dir = missing_parts_base_dir_v2
        
    base_dir_instance = os.path.join(img_dir, category, "*")
    instance_dirs = glob.glob(base_dir_instance)

    print(base_dir_instance)
    for index in range(len(instance_dirs)):

        instance_dir = instance_dirs[index]
        # first seperate the images into their respective classes

        class_mapper = {"orig": 0}
        class_counter = 1

        classes_dict = {"orig": []}

        for img_path in glob.glob(os.path.join(instance_dir, "*.png")):

            label_str_base = img_path.split('/')[-1].split('_')[0]
            label_str_part_num = img_path.split('/')[-1].split('_')[1]

            if label_str_base == "orig":
                classes_dict[label_str_base].append(img_path)

            else:
                base_obj_str = label_str_base + label_str_part_num

                if base_obj_str not in class_mapper:
                    class_mapper[base_obj_str] = class_counter
                    class_counter += 1
                    classes_dict[base_obj_str] = []

                classes_dict[base_obj_str].append(img_path)
            
            print(classes_dict)

            # for each instance, for each class, choose 1 img for validation, and 1 img for test (query)

        for class_name, item in classes_dict.items():
                # randomly choose 1 for val, 2 for test

            test_sample = rng.choice(len(item), size=3, replace=False)

            val_img = item[test_sample[0]]
            test_img_1, test_img_2 = item[test_sample[1]], item[test_sample[2]]

            l = val_img.split('/')

            dir_path = os.path.join(missing_parts_base_dir_v2, "validation", l[-3], l[-2])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_v2 = os.path.join(missing_parts_base_dir_v2, "validation", l[-3], l[-2], l[-1])
            shutil.move(val_img, img_v2)

            l = test_img_1.split('/')
            dir_path = os.path.join(missing_parts_base_dir_v2, "test", l[-3], l[-2])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_v2 = os.path.join(missing_parts_base_dir_v2, "test", l[-3], l[-2], l[-1])
            shutil.move(test_img_1, img_v2)

            l = test_img_2.split('/')
            dir_path = os.path.join(missing_parts_base_dir_v2, "test", l[-3], l[-2])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            img_v2 = os.path.join(missing_parts_base_dir_v2, "test", l[-3], l[-2], l[-1])
            shutil.move(test_img_2, img_v2)


            



if __name__ == "__main__":
    # split_by_30_degrees()
    test_query_imgs_fp = "view_comb_query_imgs.json"

    categories = [
        'KitchenPot', 
                  'USB', 'Cart', 'Box',
                   'Pliers', 'WashingMachine', 
                'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                'Luggage', 'Window', 'Faucet', 'Eyeglasses', 'Kettle', 'Toilet', 
                'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']
    
    all_res_dict = {}

    for category in categories:
    #     print(category)
        
    #     res_dict = gen_original_pairs(category)
    #     all_res_dict.update(res_dict)

    #     with open(test_query_imgs_fp, 'w') as fp:
    #         json.dump(all_res_dict, fp)
        make_img_split(category, 44)


            
