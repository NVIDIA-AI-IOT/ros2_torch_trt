import torch
from torchvision import models
from torch2trt import torch2trt
from torch2trt import TRTModule
from timeit import default_timer as timer
import os

model_path = os.getenv("HOME") + '/ros2_models/ros2_classification.pth' 

print("Importing model......")

# Use the SqueezeNet model for classification
squeezenet = models.squeezenet1_0(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

print("Create tensorRT version.....")
# convert to TensorRT feeding sample data as input
squeezenet_trt = torch2trt(squeezenet, [x])

start = timer()
y = squeezenet(x)
end = timer()
print("Squeezenet time: ", (end-start))

start = timer()
y_trt = squeezenet_trt(x)
end = timer()
print("Suqeezenet TRT time: ", (end - start))


# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))

print("Saving TRT version....")
torch.save(squeezenet_trt.state_dict(), model_path)
