import bpy
import glob
import os
from math import *
from mathutils import *


paths = ['/Users/bprovan/Downloads/dataset/148/textured_objs/*.obj']
save_path = '/Users/bprovan/Desktop/blender_pics/'
view_angles = [0, pi/8, pi/4, pi*3/8, pi/2]

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
        id = "".join(list(filter(str.isdigit, fn[-10:-1])))
        bpy.context.selected_objects[0].name = 'original-' + id


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


def take_panorama(instance_id, save_folder, num_steps=20):

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
    radius_range = [5]

    R = (target.location.xy-cam.location.xy).length # Radius
    
    target_angle = pi*2
    #pi/2 is 90 degrees, 2pi is 360
    # num_steps = 20 #how many rotation steps

    for r in radius_range:
        for x in range(num_steps):
            alpha = (x)*target_angle/num_steps
            cam.rotation_euler[2] = pi/2 + alpha #
            cam.location.x = t_loc_x+cos(alpha)*R
            cam.location.y = t_loc_y+sin(alpha)*R

            # Define SAVEPATH and output filename

            dir_path = os.path.join(save_folder, str(instance_id))
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            # save_file = os.path.join(dir_path, str(x)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))
            save_file = os.path.join(dir_path, str(x)+'_' + str(round(alpha,2)))

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


def set_camera_view_angle(view_angle: float, radius):
    cam = bpy.data.objects['Camera']

    # set it to the origin
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler[0] = view_angle
    cam.rotation_euler[2] = pi/2

    # now we set the angle and position
    cam.location.x = radius*sin(view_angle)
    cam.location.z = radius*cos(view_angle)


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


def make_view_panorama_dataset(instances_dict: dict, 
                               view_angles=[0, pi/8, pi/4, pi*3/8, pi/2], 
                               shots_per_pan=20,
                               radius=5):
    
    for instance_key, instance_parts in instances_dict.items():

        path = os.path.join('/Users/bprovan/Downloads/dataset', str(instance_key), 'textured_objs/*.obj')

        # main part, load, take pictures, remove

        load_objs(path)

        save_path = '/Users/bprovan/Desktop/blender_pics/'

        # so, we take pictures of the instance with no parts removed

        for view_angle in view_angles:
            # set the camera view
            set_camera_view_angle(view_angle, radius)

            pics_folder = str(instance_key) + '_view' +str(round(view_angle, 2)) + '_true'
            take_panorama(pics_folder, save_path)
            
            # instance parts represents the parts that can be removed
            for part_id in instance_parts:

                str_part_id = str(part_id)
                part_name = 'mesh{}/mesh{}-geometry#mesh{}-geometry'.format(str_part_id, str_part_id, str_part_id)

                # hide the part
                part_hide_render(part_id=part_name)

                pics_folder = str(instance_key) + '_view' +str(round(view_angle,2)) + '_part' + str_part_id
                take_panorama(pics_folder, save_path)

                part_unhide_render(part_id=part_name)

        
        # delete the objects
        delete_objects()
    
#load_objs(paths[0])

# part_to_hide = 'Group1/mesh1/mesh1-geometry#mesh1-geometry'

# part_hide_render(part_id=part_to_hide)
#set_camera_view_angle(pi/4, radius=5)

# take_panorama(148, save_path)

#delete_objects()

#make_panorama_dataset(glasses)

make_view_panorama_dataset(glasses, 
                               view_angles=[0, pi/8, pi/4, pi*3/8, pi/2], 
                               shots_per_pan=20,
                               radius=5)