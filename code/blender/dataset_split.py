import os
import random
import shutil
import time
from collections import defaultdict
import glob


# TODO: split unseen abnormally to val and test datasets
def split_dataset(
    image_dir,
    # output_dir,
    category,
    train_ratio=0.8,
    val_ratio=0.1,
    intra_object_val_test_ratio=0.1,
):
    start_time = time.time()

    # Create output directories
    # train_dir = os.path.join(output_dir, "train")
    # val_dir = os.path.join(output_dir, "val")
    # test_dir = os.path.join(output_dir, "test")
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)

    # Get a list of unique object IDs
    print("Getting object IDs...")
    # image_files = os.listdir(image_dir)
    # image_files = glob.glob()

    # object_ids = list(set([img_file.split("_")[0] for img_file in image_files]))

    src_path = os.path.join('/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded', category)
    dst_path = '/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded_unseen_anomaly2'
    print("HMMMM")
    print("PATH", src_path)
    object_ids = os.listdir(src_path)
    img_dir = os.path.join(src_path, '*/' '*.png')
    image_files = glob.glob(img_dir)
    random.shuffle(object_ids)

    # Split the object IDs into train, val, test
    print("Splitting object IDs...")
    num_train = int(len(object_ids) * train_ratio)
    num_val = int(len(object_ids) * val_ratio)
    train_ids = object_ids[:num_train]
    train_ids = object_ids
    # val_ids = object_ids[num_train : num_train + num_val]
    # test_ids = object_ids[num_train + num_val :]

    object_parts = defaultdict(lambda: defaultdict(list))
    for img_file in image_files:
        sole_img_file = img_file.split('/')[-1]
        # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
        # label_str_part_num = img_path.split('/')[-1].split('_')[1]
        object_id = img_file.split('/')[-2]
        part_name, part_id = sole_img_file.split("_")[0:2]
        if (
            object_id in train_ids
            and part_name != "orig"
            and part_id not in object_parts[object_id][part_name]
        ):
            object_parts[object_id][part_name].append(part_id)
            random.shuffle(object_parts[object_id][part_name])

    # Split the image files based on the object IDs
    print("Splitting image files...")
    for img_file in image_files:
        # object_id, part_name, part_id = img_file.split("_")[0:3]
        sole_img_file = img_file.split('/')[-1]
        # later this will be important for multi-class class splitting, the id of the removed part. E.g. leg 0, leg 1 ...
        # label_str_part_num = img_path.split('/')[-1].split('_')[1]
        object_id = img_file.split('/')[-2]
        part_name, part_id = sole_img_file.split("_")[0:2]
        view_num = sole_img_file.split("_")[-1][0:-4]
        # src_file = os.path.join(image_dir, img_file)
        src_file = img_file

        if int(view_num) %2 == 1:
            # Move a portion of intra-object images to validation and test sets

            # eg so if there is a second possible anomaly, eg other arm of chair, other leg of glasses, then this goes to val and test
            if part_name != "orig":
            
                if object_parts[object_id][part_name].index(part_id) % 2 == 1:
                    if random.random() < 0.5:
                        if not os.path.exists(os.path.join(dst_path, "validation", category, object_id)):
                            os.makedirs(os.path.join(dst_path, "validation", category, object_id))
                        dst_dir = os.path.join(dst_path, "validation", category, object_id, sole_img_file)
                    else:
                        if not os.path.exists(os.path.join(dst_path, "test", category, object_id)):
                            os.makedirs(os.path.join(dst_path, "test", category, object_id))
                        dst_dir = os.path.join(dst_path, "test", category, object_id, sole_img_file)
                # put it in training
                else:

                    if not os.path.exists(os.path.join(dst_path, "train", category, object_id)):
                        os.makedirs(os.path.join(dst_path,  "train", category, object_id))
                    dst_dir = os.path.join(dst_path,  "train", category, object_id, sole_img_file)

        # elif object_id in val_ids:
        #     dst_dir = val_dir
        # else:
        #     dst_dir = test_dir

                shutil.copy(src_file, dst_dir)

    end_time = time.time()
    print(f"Done, took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # categories = [
    #     "Box",
    #     "Bucket",
    #     "Cart",
    #     "Dish Washer",
    #     "Globe",
    #     "Kettle",
    #     "Lamp",
    #     "Laptop",
    #     "Lighter",
    #     "Luggage",
    #     "Oven",
    #     "Phone",
    #     "Pliers",
    #     "Pot",
    #     "Scissors",
    #     "Sitting Furniture",
    #     "Stapler",
    #     "Storage Furniture",
    #     "Table",
    #     "Toilet",
    #     "Trash Can",
    #     "Trashcan",
    #     "USB",
    #     "WashingMachine",
    # ]

    categories = ['KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'WashingMachine', 
                'Lighter', 'Switch', 'Laptop', 'Bucket', 'Globe', 'Trashcan', 
                'Luggage', 'Window', 'Faucet', 'Kettle', 'Toilet',
                'Oven', 'Stapler', 'Phone', 'Trash Can', 'Scissors', 'Dish Washer', 
                'Lamp', 'Sitting Furniture', 'Table', 'Storage Furniture', 'Pot']
    categories = ['Eyeglasses']
    for category in categories:
        # src_path = rf"E:\UOE\Dissertation\dataset\MySAPIEN_woOccl\{category}\images"
        # dst_path = rf"E:\UOE\Dissertation\dataset\MySAPIEN\{category}\images"

        src_path = os.path.join('/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded', category)
        dst_path = os.path.join('/Users/bprovan/University/dissertation/datasets/images_ds_v1_occluded_unseen_anomaly2')
        split_dataset(src_path, category)
