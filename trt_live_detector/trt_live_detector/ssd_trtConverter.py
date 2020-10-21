from mobilenetv1_ssd import create_mobilenetv1_ssd

from torch2trt import torch2trt
import torch
import os

net_type = 'mb1-ssd'
model_path = os.getenv("HOME")+ '/ros2_models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = os.getenv("HOME") + '/ros2_models/voc-model-labels.txt'


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

print("Loading the model.....")
model = create_mobilenetv1_ssd(len(class_names), is_test=True)
model.load_state_dict(torch.load(model_path))
model.eval().cuda()

x = torch.ones((1,3,300,300)).cuda()

print("Creating TRT version...........")
model_trt = torch2trt(model, [x])
print("Created TRT version.......")

save_location = os.getenv("HOME") + '/ros2_models/mb1SSD_trt.pth'

print("Saving TRT model......")
torch.save(model_trt.state_dict(), save_location)
