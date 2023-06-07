

import bpy
import glob

# but what about multiple files?
# seems to work with the material too

fn_list = glob.glob('/Users/bprovan/Downloads/dataset/148/textured_objs/*.obj')
for fn in fn_list:
    
    bpy.ops.import_scene.obj(filepath=fn, filter_glob='*.obj;*.mtl')
#obj_object = bpy.context.selected_objects[0] ####<--Fix
#print('Imported name: ', obj_object.name)


# So you can iterate through the objects in bpy.data.objects
# and then remove said objects,
# However this also includes the camera and the lights? so we should avoid moving those?

# object.hide_render()

# Or just create a new scene entirely..., not sure how this would work.
# Also removes all of the camera and light stuff, not ideal...
"""
bpy.ops.wm.read_factory_settings(use_empty=True)
"""

# can call this after every instance is imported
import bpy
# remove it?
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
