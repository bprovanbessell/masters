"""

This will form the baseline binary fault classification.
2 main models
- Most basic one which takes in one image as imput, and trys to detect whether it has a part missing or not (fault)
- base it off https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

- feature vector stacking
One that takes in multiple images, a feature vector is taken from each of them, and they are stacked together, 
and then a classifier is trained on them.

"""

import torch
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     x = torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")

device = "mps" if torch.backends.mps.is_available() \
    else "gpu" if torch.cuda.is_available() else "cpu"