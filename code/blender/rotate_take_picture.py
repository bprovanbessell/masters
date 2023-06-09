import os
import bpy
from math import radians

# This rotates the object, not the camera
# we need to rotate the camera

def rotate_and_render(output_dir, output_file_pattern_string = 'render%d.jpg', rotation_steps = 32, rotation_angle = 360.0, subject = bpy.context.object):
  
  original_rotation = subject.rotation_euler
  for step in range(0, rotation_steps):
    subject.rotation_euler[2] = radians(step * (rotation_angle / rotation_steps))
    bpy.context.scene.render.filepath = os.path.join(output_dir, (output_file_pattern_string % step))
    bpy.ops.render.render(write_still = True)
  subject.rotation_euler = original_rotation

rotate_and_render('/Users/bprovan/Desktop/blender_pics', 'render%d.jpg')


# does rotate the camera around, However, we need to rotate the camera around the object, as opposed to just rotating the camera

import os
import bpy
from math import radians

def rotate_and_render(output_dir, output_file_pattern_string = 'render%d.jpg', rotation_steps = 32, rotation_angle = 360.0, subject = bpy.context.object):
  
  camera = bpy.data.objects["Camera"]
  original_rotation = camera.rotation_euler
  for step in range(0, rotation_steps):
    camera.rotation_euler[2] = radians(step * (rotation_angle / rotation_steps))
    bpy.context.scene.render.filepath = os.path.join(output_dir, (output_file_pattern_string % step))
    bpy.ops.render.render(write_still = True)
  camera.rotation_euler = original_rotation

#rotate_and_render('/Users/bprovan/Desktop/blender_pics', 'render%d.jpg')

rotate_and_render('/Users/bprovan/Desktop/blender_pics', 'render3.%d.jpg', subject = bpy.data.objects["new-0"])


# Rotate at a range of radii, can change this of course, good for data augmentations
# A modified version of this will work

# We also want something that works 

import bpy
import os
from math import *
from mathutils import *

#set your own target here
target = bpy.data.objects['new-0']
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
cam_loc_x = cam.location.x
cam_loc_y = cam.location.y
# The different radii range
#radius_range = range(3,9)
radius_range = [9]

R = (target.location.xy-cam.location.xy).length # Radius

init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/R)-2*pi*bool((cam_loc_y-t_loc_y)<0) # 8.13 degrees

#can we get rid of the init angle stuff??`
target_angle = (2*pi - init_angle) # Go 90-8 deg more
#pi/2 is 90 degrees, 2pi is 360
num_steps = 20 #how many rotation steps

for r in radius_range:
    for x in range(num_steps):
        alpha = init_angle + (x)*target_angle/num_steps
        cam.rotation_euler[2] = pi/2 + alpha #
        cam.location.x = t_loc_x+cos(alpha)*r
        cam.location.y = t_loc_y+sin(alpha)*r

        # Define SAVEPATH and output filename
        file = os.path.join('/Users/bprovan/Desktop/blender_pics/', str(x)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))

        # Render
        bpy.context.scene.render.filepath = file
        bpy.ops.render.render(write_still=True)



import bpy
import os
from math import *
from mathutils import *

#set your own target here
target = bpy.data.objects['new-0']
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
cam_loc_x = cam.location.x
cam_loc_y = cam.location.y
# The different radii range
#radius_range = range(3,9)
radius_range = [9]

R = (target.location.xy-cam.location.xy).length # Radius

init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/R)-2*pi*bool((cam_loc_y-t_loc_y)<0) # 8.13 degrees

#can we get rid of the init angle stuff??`
target_angle = (2*pi - init_angle) # Go 90-8 deg more
#pi/2 is 90 degrees, 2pi is 360
num_steps = 20 #how many rotation steps

for r in radius_range:
    for x in range(num_steps):
        alpha = init_angle + (x)*target_angle/num_steps
        cam.rotation_euler[2] = pi/2 + alpha #
        cam.location.x = t_loc_x+cos(alpha)*r
        cam.location.y = t_loc_y+sin(alpha)*r

        # Define SAVEPATH and output filename
        file = os.path.join('/Users/bprovan/Desktop/blender_pics/', str(x)+'_'+ str(r)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))

        # Render
        bpy.context.scene.render.filepath = file
        bpy.ops.render.render(write_still=True)




def change_camera_z_view_angle():
    mesh1 = bpy.data.meshes[0].name
    target = bpy.data.objects[mesh1]
    cam = bpy.data.objects['Camera']
    t_loc_x = target.location.x
    t_loc_y = target.location.y
    cam_loc_x = cam.location.x
    cam_loc_y = cam.location.y

    # cos along the x, sin along the y (z)

    # we still have the same target. We need to find out what axis the camera is initially oriented at?, I don't want to think 
    #  a 3d coordingate orientation
    # which camera rotation euler has to be changed? previously they changed the [2], the z (makes sense)
    # so change the x one??  I think so at least

    # cam.rotation_euler[0] = 0 # pointing straight down
    # cam.rotation_euler[0] = pi/2 # pointing horizontally = 90 degrees

    # so probably the best thing is to initially translate (move) the camera to it with angle rotation of 0, (z)
    # and then change the location...

    # Could do this based on fixed radius
    # so if its at 45 degrees initially, then cos(45) along x and sin(45) along the z, nah i think we need to invert it
    # eg when its pointing horizontally, we need the x to be 1, and z to be 0
    # so we can invert and do sin along the x, and cos along the y

    # so, shift the camera to (0,0,0)

    # camera rotation euler[2] should be pi/2 (to point in origin on z plane), where x is positive
    # setting initial camera view (horizontal)

    # rot angle is camera rotation euler x

    cam.rotation_euler[0] = rot_angle

    r = 8 # radiu
    

    cam.location.x = 5*sin(rot_angle)
    cam.location.z = 5*cos(rot_angle)

    cam.rotation_euler[2] = pi/2
