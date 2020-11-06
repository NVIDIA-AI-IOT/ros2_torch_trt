sudo xhost +si:localuser:root
sudo docker run -it --rm --runtime nvidia --device="/dev/video0:/dev/video0" --network host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v ${pwd}:/workdir ros2_torch2trt_base:jp44
