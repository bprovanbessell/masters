"""
You have to pass the obect index mask to the composite (that is what actually saves the image?)

No idea how you would do this with python, re creating it could be really shite

What did i do

set to render to rgba

make the composit mask -> and somehjow this gets rendered only? I changed it in the render screen -> top right went to objec thang...

render then also the alpha channel


My student's answer: We can render a mask of the concerned part.
 In Blender, one can assign object index values to each part-level object by setting pass_index to specific values ie. 1 or 0. 
 In our case, we have to set 1 for the anomalized part and 0 for the rest of the parts. 
 After rendering, you'll find the mask image, which will be black everywhere except where the part object is located. 
 If it is fully occluded by other parts, then it will be black everywhere.

But then even once I have these masks, I need to create a script that loads the image, if the colour average is anything but white
Then the anomaly is occluded, so you don't use it.

"""

import bpy

bpy.context.space_data.context = 'VIEW_LAYER'
bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

bpy.context.scene.use_nodes = True


D.objects[1].pass_index = 1

# for white background bpy.context.scene.render.film_transparent = True
