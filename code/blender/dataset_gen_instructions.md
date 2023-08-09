# Dataset Generation Instructions

## 3D model downloading
The 3D models used for the dataset generation can be downloaded from https://sapien.ucsd.edu/downloads. 
Please follow the instructions on that webpage to download the PartNet-mobility dataset.

## Object rendering
To render the images of the 3D models, use the ```renderer.py``` python file. Please set the respective paths to point to the generated dataset location, as well as the partnet mobility dataset location.

## Object Occlusion
First you must set up Blender as follows:
1. Change render engine to cycles
2. Activate object level pass
3. In View layer properties - in passes - data - select the checkbox for object indexes
4. Select use node in compositor tab - Create new pipeline that uses the object index, and that is passed into the alpha for the composite to generate the image
5. set render output to RGBA, to ensure the alpha channel is rendered.

After this, use the renderer.py script with ```occlusion_mask = True``` 

To remove the occluded files from the dataset, use the ```remove_occluded.py``` file.
