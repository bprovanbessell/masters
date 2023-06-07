import bpy
import glob
import os
from math import *
from mathutils import *


paths = ['/Users/bprovan/Downloads/dataset/148/textured_objs/*.obj']
save_path = '/Users/bprovan/Desktop/blender_pics/'

# testing purposes
glasses = {}

# all parts prepended with 'original-'
# glasses[101284] = [2,3,4,5]
# glasses[101285] = [1,3,4,5]

# try wuth the most simple example first
glasses[101287] = [2,3]


def load_objs(path_to_folder):

    fn_list = glob.glob(path_to_folder)
    for fn in fn_list:
        
        bpy.ops.import_scene.obj(filepath=fn, filter_glob='*.obj;*.mtl')

def delete_objects():
    # select all the objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
        else:
            obj.select_set(False)
            
    bpy.ops.object.delete()

        #remove it, this way you dont need to reopen/close file...
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


def take_panorama(instance_id, save_folder):

    #set your own target here
    # target = bpy.data.objects['new-0']
    # there should be something loaded, if this throws an error, then we haven't loaded properly
    mesh1 = bpy.data.meshes[0].name
    target = bpy.data.objects[mesh1]
    cam = bpy.data.objects['Camera']
    t_loc_x = target.location.x
    t_loc_y = target.location.y
    cam_loc_x = cam.location.x
    cam_loc_y = cam.location.y
    # The different radii range
    #radius_range = range(3,9)
    radius_range = [9]

    R = (target.location.xy-cam.location.xy).length # Radius
    
    target_angle = pi*2
    #pi/2 is 90 degrees, 2pi is 360
    num_steps = 20 #how many rotation steps

    for r in radius_range:
        for x in range(num_steps):
            alpha = (x)*target_angle/num_steps
            cam.rotation_euler[2] = pi/2 + alpha #
            cam.location.x = t_loc_x+cos(alpha)*r
            cam.location.y = t_loc_y+sin(alpha)*r

            # Define SAVEPATH and output filename

            dir_path = os.path.join(save_folder, str(instance_id))
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            save_file = os.path.join(dir_path, str(x)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))

            # Render
            bpy.context.scene.render.filepath = save_file
            bpy.ops.render.render(write_still=True)


# This should hide the specific part of the instance
def part_hide_render(part_id: str):

    # use the hide render method
    bpy.data.objects[part_id].hide_render = True

# Reset hide render back to default (false)
def part_unhide_render(part_id: str):

    # use the hide render method
    bpy.data.objects[part_id].hide_render = False

def hide_parts_list(parts_to_hide):
    for part_id in parts_to_hide:
        part_hide_render(part_id)

    # then take the photos
    for part_id in parts_to_hide:
        part_unhide_render(part_id)

def make_panorama_dataset(instances_dict: dict):

    for instance_key, instance_parts in instances_dict.items():
        paths = ['/Users/bprovan/Downloads/dataset/148/textured_objs/*.obj']

        path = os.path.join('/Users/bprovan/Downloads/dataset', str(instance_key), 'textured_objs/*.obj')

        # main part, load, take pictures, remove

        load_objs(path)

        save_path = '/Users/bprovan/Desktop/blender_pics/'

        # so, we take pictures of the instance with no parts removed

        pics_folder = str(instance_key) + '_true'
        take_panorama(pics_folder, save_path)
        
        # instance parts represents the parts that can be removed
        for part_id in instance_parts:

            str_part_id = str(part_id)
            part_name = 'mesh{}/mesh{}-geometry#mesh{}-geometry'.format(str_part_id, str_part_id, str_part_id)

            # hide the part
            part_hide_render(part_id=part_name)

            pics_folder = str(instance_key) + '_' + str_part_id
            take_panorama(pics_folder, save_path)

            part_unhide_render(part_id=part_name)

        
        # delete the objects
        delete_objects()


            

# 'mesh1/mesh1-geometry#mesh1-geometry'
#  where the number is substituted? I'm not sure exactly how this importing works, 
# I just hope it is consistent with how the folder is laod out (should be!)
 

# load_objs(paths[0])

# part_to_hide = 'Group1/mesh1/mesh1-geometry#mesh1-geometry'

# part_hide_render(part_id=part_to_hide)

# take_panorama(148, save_path)

#delete_objects()

make_panorama_dataset(glasses)