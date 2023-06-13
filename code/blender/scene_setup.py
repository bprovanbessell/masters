
# adding a plane (floor) Worry about this later, it can be quite easily added
# In fact there are minumum and maximum bounding boxes in the bounding box.json
# Thus could be useful for object scaling, and also for putting in the floor(we just set the z 
# location to the z minimum bounding box, and make it have a large size)
import bpy
from math import *

bpy.ops.mesh.primitive_plane_add(size=2.0, align='WORLD', location=(0, 0, 0))

# much cleaner
def new_plane(mylocation, mysize, myname):
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=mylocation,
        rotation=(0, 0, 0),
        scale=(0, 0, 0))
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    return

# for adding a material
# MAT_NAME = "TackyGold"
# bpy.data.materials.new(MAT_NAME)
# material = bpy.data.materials[MAT_NAME]
# material.use_nodes = Truematerial.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.1
# material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.75,0.5,0.05,1)
# material.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.9if len(plane.data.materials.items()) != 0:
#     plane.data.materials.clear()
# else:
#     plane.data.materials.append(material)

def set_camera_view_angle(view_angle: float, radius):
    cam = bpy.data.objects['Camera']

    # set it to the origin
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler[0] = view_angle

    # now we set the angle and position
    cam.location.x = radius*sin(view_angle)
    cam.location.z = radius*cos(view_angle)

    cam.rotation_euler[2] = view_angle

def set_render_resolution(res_x, res_y):
    r = bpy.context.scene.render
    r.resolution_x = res_x
    r.resolution_y = res_y