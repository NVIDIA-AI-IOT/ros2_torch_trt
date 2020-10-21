'''Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''


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
