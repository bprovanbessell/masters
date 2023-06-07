import bpy


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

