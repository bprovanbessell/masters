import json
import os

WORKSPACE_PATH = "/Users/bprovan/University/dissertation/mpd-master/SAPIEN"
MODEL_ID_JSON = "SAPIEN_ALL_CHILDREN.json"


json_file = os.path.join(WORKSPACE_PATH, MODEL_ID_JSON)
with open(json_file, "r") as f:
    data: dict = json.load(f)
    categories = list(data.keys())


print(categories)

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

print(CATEGORIES)