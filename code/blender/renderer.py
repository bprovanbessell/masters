import glob
import json
import os
import time
import traceback
from math import *

import bpy
import numpy as np
from mathutils import *

"""
TODO:

1. How to decide the image resolution
2. What number of images per state of one model should we render
3. Whether to render the entire panorama around the model or just the upper part of the object
4. How to handle images rendered from parts that are removed but not visible in the camera itself,
   and whether these images should be marked as positive or negative
"""



DATASET_PATH = "/Users/bprovan/University/dissertation/datasets/partnet-mobility"
 # path to save pics
SAVE_PATH = "/Users/bprovan/University/dissertation/datasets/images_mask"
# path to save this scripts and used json files

WORKSPACE_PATH = "/Users/bprovan/University/dissertation/mpd-master/SAPIEN"

ERROR_FILE_PATH = "Users/bprovan/University/dissertation/mpd-master/SAPIEN"

MODEL_CHILDREN_JSON = "SAPIEN_ALL_CHILDREN.json"
MODEL_ID_JSON = "SAPIEN_model_ids.json"

json_file = os.path.join(WORKSPACE_PATH, MODEL_CHILDREN_JSON)
with open(json_file, "r") as f:
    data: dict = json.load(f)
    CATEGORIES = list(data.keys())


# parse the removable parts stored in SAPIEN_ALL_CHILDREN. The json file now
# holds the names of all possible parts that appear in all levels of children.
# The parts of the list that cannot be deleted need to be removed as needed
# during the rendering process
REMOVEABLES: dict = {}
REMOVEABLE_JSON = "SAPIEN_ALL_CHILDREN.json"
json_file = os.path.join(WORKSPACE_PATH, REMOVEABLE_JSON)
with open(json_file, "r") as f:
    REMOVEABLES: dict = json.load(f)


def set_render_resolution(res_x, res_y):
    r = bpy.context.scene.render
    r.resolution_x = res_x
    r.resolution_y = res_y


VIEW_RADIUS = 5.0  # view radius of the camera


def load_model(path_to_model: str):
    """Load model in blender

    Args:
        path_to_model (str): model folder abspath, which is also the model id
    """
    path_to_objs = os.path.join(path_to_model, "textured_objs", "*.obj")
    obj_files = glob.glob(path_to_objs)

    all_object_ids = []
    for obj_file in obj_files:
        bpy.ops.import_scene.obj(filepath=obj_file)
        obj_name = os.path.split(os.path.splitext(obj_file)[0])[-1]
        bpy.context.selected_objects[0].name = obj_name
        all_object_ids.append(obj_name)

    return all_object_ids


def clear_scene() -> None:
    """Clear blender scene"""
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
        else:
            obj.select_set(False)

    bpy.ops.object.delete()

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def look_at(obj, target):
    """Adjust the angle of the obj so that the obj faces the target

    Args:
        obj (_type_): The object to be adjusted, in this script mainly the camera
        target (_type_): The target object, in this script mainly the model
    """
    direction = target - obj.location
    # Compute the rotation matrix that aligns Z axis to this direction
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def generate_camera_points(num_points: int, angle_deg: float = 30):
    """Generate points for camera, distributed evenly over a 360 degree circle
       at a specific angle from horizontal plane.

    Args:
        num_points (int): number of generated points
        angle_deg (float): angle from horizontal plane in degrees. Defaults to 30.

    Returns:
        2D vector: points
    """
    # Convert angle to radians
    angle_rad = radians(angle_deg)

    # Generate evenly distributed points over the 360 degree range
    theta = np.linspace(0, 2 * np.pi, num_points)

    x = np.cos(theta)
    y = np.sin(theta)
    z = np.tan(angle_rad) * np.sqrt(x * x + y * y)
    return np.column_stack((x, y, z))


def take_panorama(
    model_id: str, category: str, prefix: str, num_images: int = 24, angle_deg: float = 30
) -> None:
    """Take panorama photos of target obj. These photographs will be taken evenly
       from a circular path at a specific angle from the horizontal plane of the object.

    Args:
        model_id (str): model_id, i.e. the folder name
        prefix (str): prefix to save the pics
        num_images (int, optional): number of images to be taken. Defaults to 12.
        angle_deg (float): angle from horizontal plane in degrees. Defaults to 30.
    """
    cam = bpy.data.objects["Camera"]
    target = Vector((0.0, 0.0, 0.0))  # The target point you want the camera to look at

    points = generate_camera_points(num_images, angle_deg)
    for idx, point in enumerate(points):
        cam.location.x = VIEW_RADIUS * point[0]
        cam.location.y = VIEW_RADIUS * point[1]
        cam.location.z = VIEW_RADIUS * point[2]

        look_at(cam, target)  # Adjust the rotation of the camera

        dir_path = os.path.join(SAVE_PATH, category, model_id)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        save_file = os.path.join(dir_path, f"{prefix}_{idx}")

        bpy.context.scene.render.filepath = save_file
        bpy.ops.render.render(write_still=True)


def recur_remove_render(model_id: str, category: str, part_data: dict, removables: list,  occlusion_mask: bool = False) -> None:
    """recursively remove parts of the target model

    Args:
        model_id (str): target model id
        part_data (dict): target model structure data, parsing from result.json
        removables (list): removable parts list
    """

    def collect_part_ids(data: dict) -> list:
        """Recursively collect all part_ids from children structures

        Args:
            data (dict): target model structure data

        Returns:
            list: list of part_ids
        """
        part_ids = []
        if "objs" in data:
            part_ids.extend(data["objs"])
        if "children" in data:
            for child in data["children"]:
                part_ids.extend(collect_part_ids(child))
        return part_ids

    if part_data.setdefault("text", "") in removables:
        part_ids = collect_part_ids(part_data)

        # Hide all parts for rendering
        for part_id in part_ids:
            if part_id in bpy.data.objects:
                if occlusion_mask:
                    # set anomalous to 1, all the rest to 0
                    
                    bpy.data.objects[part_id].pass_index = 1
                else:
                    # otherwise hide the part. Will this fix it??
                    bpy.data.objects[part_id].hide_render = True

        # Take panorama
        take_panorama(model_id, category, f"{part_data['text']}_{part_data['id']}")

        # Unhide all parts after rendering
        for part_id in part_ids:
            if part_id in bpy.data.objects:
                bpy.data.objects[part_id].hide_render = False

                if occlusion_mask:
                    # set anomalous to 1, all the rest to 0
                    bpy.data.objects[part_id].pass_index = 0

    # Continue recursion for children
    if "children" in part_data:
        for child in part_data["children"]:
            recur_remove_render(model_id, category, child, removables)


def render(model_id: str, category: str, occlusion_mask: bool = False) -> None:
    """render one model. Take photos of complete model and model with missing parts

    Args:
        model_id (str): target model id
    """
    model_path = os.path.join(DATASET_PATH, model_id)
    all_obj_ids = load_model(model_path)
    # Get panorama of original obj
    # set all part object ids to 0
    for obj_id in all_obj_ids:
        bpy.data.objects[obj_id].pass_index = 0

    take_panorama(model_id, category, "orig")
    part_data: dict = {}
    json_path = os.path.join(model_path, "result.json")
    with open(json_path, "r") as fin:
        # result.json is a list, with the first ele storeing data
        part_data = json.load(fin)[0]
    category = part_data.setdefault("text", "default")
    removables = REMOVEABLES.setdefault(category, [])
    if len(removables) == 0:
        return
    recur_remove_render(model_id, category, part_data, removables,  occlusion_mask)


def set_render_background():
    bpy.context.space_data.shader_type = 'WORLD'
    bpy.data.worlds["World"].node_tree.nodes["Background"].use_custom_color = True
    bpy.data.worlds["World"].node_tree.nodes["Background"].color = (1, 1, 1)


def main():
    set_render_resolution(128, 128)

    start_time = time.time()

    # CATEGORIES = ['KitchenPot', 'Switch', "Eyeglasses"]
    to_remove = ['Remote', 'Clock', 'CoffeeMachine', 'Suitcase', 'Printer', 'Refrigerator', 
                 'FoldingChair', 'Knife', 'Fan', 'Dispenser', 'Camera', 'Safe', 'Mouse',
                 'Pen', 'Toaster', 'Keyboard', 'Bottle', 'Display', 'Microwave', 'Mug', 'Door Set']
    # questionable - toilet, phone, Dishwasher, Lamp, Sitting Furniture (should remove both leg and wheels), Storage Furniture,
    CATEGORIES = ['Remote', 'KitchenPot', 'USB', 'Cart', 'Box', 'Pliers', 'Suitcase', 
     'Printer', 'WashingMachine', 'Lighter', 'Refrigerator', 'Switch', 
     'Laptop', 'Bucket', 'FoldingChair', 'Globe', 'Trashcan', 'Luggage', 
     'Window', 'Knife', 'Faucet', 'Fan', 'Eyeglasses', 'Kettle', 'Toilet', 
     'Dispenser', 'Camera', 'Safe', 'Mouse', 'Pen', 'Oven', 'CoffeeMachine', 
     'Stapler', 'Phone', 'Toaster', 'Trash Can', 'Scissors', 'Dish Washer', 
     'Keyboard', 'Lamp', 'Sitting Furniture', 'Table', 'Bottle', 'Display', 
     'Storage Furniture', 'Pot', 'Clock', 'Microwave', 'Mug', 'Door Set']
    
    CATEGORIES = [x for x in CATEGORIES if x not in to_remove]

    occlusion_mask = False

    for category in CATEGORIES:
        model_ids = []
        json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
        with open(json_file, "r") as f:
            data: dict = json.load(f)
            for key in data.setdefault(category, []):
                model_ids.append(key)
        
        category_dir = os.path.join(SAVE_PATH, category)
        os.mkdir(category_dir)

        for id in model_ids:
            try:
                clear_scene()
                render(id, category, occlusion_mask)
            except Exception as e:
                error_message = f"An error occurred with model id {id}: {str(e)}"
                print(error_message)  # output to console
                error_file = os.path.join(ERROR_FILE_PATH, "error.log")
                with open(error_file, "a") as f:  # write to file
                    f.write(error_message + "\n")
                    traceback.print_exc(file=f)  # write traceback info to file
        total_time = time.time() - start_time
        print(
            f"Total time: {total_time:.2f}s, "
            f"average time per model: {total_time / len(model_ids):2f}s"
        )


if __name__ == "__main__":
    main()
