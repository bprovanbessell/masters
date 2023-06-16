import glob
import json
import os
from math import *

import bpy
import numpy as np
from mathutils import *

# DATASET_PATH = r"E:\UOE\Dissertation\dataset\SAPIEN"
DATASET_PATH = "/Users/bprovan/Downloads/dataset"
# SAVE_PATH = r"E:\UOE\Dissertation\dataset\MyDataset"  # path to save pics
SAVE_PATH = "/Users/bprovan/Desktop/gen_images_640"
# path to save this scripts and used json files
# WORKSPACE_PATH = r"E:\UOE\Dissertation\Workspace\SAPIEN"
WORKSPACE_PATH = "/Users/bprovan/University/dissertation/masters/code/blender"


# parse model ids of one category. Change CATEGORY to render the pics of all
# the objs of the category
MODEL_IDS = []
CURR_CATEGORY = "KitchenPot"
MODEL_ID_JSON = "SAPIEN_model_ids.json"
json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
with open(json_file, "r") as f:
    data: dict = json.load(f)
    for key in data.setdefault(CURR_CATEGORY, []):
        MODEL_IDS.append(key)


# parse the removable parts stored in SAPIEN_ALL_CHILDREN. The json file now
# holds the names of all possible parts that appear in all levels of children.
# The parts of the list that cannot be deleted need to be removed as needed
# during the rendering process
REMOVEABLES: dict = {}
REMOVEABLE_JSON = "SAPIEN_ALL_CHILDREN.json"
json_file = os.path.join(WORKSPACE_PATH, REMOVEABLE_JSON)
with open(json_file, "r") as f:
    REMOVEABLES: dict = json.load(f)


VIEW_RADIUS = 5.0  # view radius of the camera

def set_render_resolution(res_x, res_y):
    r = bpy.context.scene.render
    r.resolution_x = res_x
    r.resolution_y = res_y


def load_model(path_to_model: str) -> None:
    """Load model in blender

    Args:
        path_to_model (str): model folder abspath, which is also the model id
    """
    path_to_objs = os.path.join(path_to_model, "textured_objs", "*.obj")
    obj_files = glob.glob(path_to_objs)
    for obj_file in obj_files:
        bpy.ops.import_scene.obj(filepath=obj_file)
        obj_name = os.path.split(os.path.splitext(obj_file)[0])[-1]
        bpy.context.selected_objects[0].name = obj_name


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


def golden_spiral_points(num_points: int):
    """Generate golden spiral points, These points will be evenly distributed
       over the hemispherical range on the ground of the target object (z > 0)

    Args:
        num_points (int): number of generated points

    Returns:
        2D vector: points
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio
    z = 1 - (indices / num_points)  # z goes from 1 to 0
    radius = np.sqrt(1 - z * z)  # radius at z

    theta = 2 * np.pi * phi * indices

    x, y = radius * np.cos(theta), radius * np.sin(theta)
    return np.column_stack((x, y, z))


def take_panorama(model_id: str, prefix: str, num_images: int = 20) -> None:
    """Take panorama photos of target obj. These photographs will be taken evenly
       from the upper hemispherical range of the object.

    Args:
        model_id (str): model_id, i.e. the folder name
        prefix (str): prefix to save the pics
        num_images (int, optional): number of iamges to be taken. Defaults to 20.
    """
    cam = bpy.data.objects["Camera"]
    target = Vector((0.0, 0.0, 0.0))  # The target point you want the camera to look at

    points = golden_spiral_points(num_images)
    for idx, point in enumerate(points):
        cam.location.x = VIEW_RADIUS * point[0]
        cam.location.y = VIEW_RADIUS * point[1]
        cam.location.z = VIEW_RADIUS * point[2]

        look_at(cam, target)  # Adjust the rotation of the camera

        dir_path = os.path.join(SAVE_PATH, model_id)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        save_file = os.path.join(dir_path, f"{prefix}_{idx}")

        bpy.context.scene.render.filepath = save_file
        bpy.ops.render.render(write_still=True)


def recur_remove_render(model_id: str, part_data: dict, removables: list) -> None:
    """recursively remove parts of the target model

    Args:
        model_id (str): target model id
        part_data (dict): target model structure data, parsing from result.json
        removables (list): removable parts list
    """
    if "children" in part_data:
        for child in part_data["children"]:
            recur_remove_render(model_id, child, removables)
    if part_data.setdefault("text", "") in removables and "objs" in part_data:
        for part_id in part_data["objs"]:
            bpy.data.objects[part_id].hide_render = True
        take_panorama(model_id, f"{part_data['text']}_{part_data['id']}")
        for part_id in part_data["objs"]:
            bpy.data.objects[part_id].hide_render = False


def render(model_id: str) -> None:
    """render one model. Take photos of complete model and model with missing parts

    Args:
        model_id (str): target model id
    """
    model_path = os.path.join(DATASET_PATH, model_id)
    load_model(model_path)
    # Get panorama of original obj
    take_panorama(model_id, "orig")
    part_data: dict = {}
    json_path = os.path.join(model_path, "result.json")
    with open(json_path, "r") as fin:
        # result.json is a list, with the first ele storeing data
        part_data = json.load(fin)[0]
    category = part_data.setdefault("text", "default")
    removables = REMOVEABLES.setdefault(category, [])
    if len(removables) == 0:
        return
    recur_remove_render(model_id, part_data, removables)


def main():

    set_render_resolution(640, 640)

    for id in MODEL_IDS:
        clear_scene()
        render(id)


if __name__ == "__main__":
    main()
