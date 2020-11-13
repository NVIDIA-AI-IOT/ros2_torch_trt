# Run ROS2-torch2trt in Docker

For more Jetson dockers, please have a look at [jetson-containers](https://github.com/dusty-nv/jetson-containers) github repository.

## Docker Default Runtime

To enable access to the CUDA compiler (nvcc) during `docker build` operations, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding.

## Building the Containers

``` sh docker_build.sh ```

If you face an error at ```COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc```
Copy ``` jetson-ota-public.asc ``` from ``` /etc/apt/trusted.gpg.d ``` to your current directory with ```docker_build```

## Run Container

``` sh docker_run.sh ```


``` cv_bridge ``` will be needed for the packages to run. Add it to your current ROS2 workspace and build it:  https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge

Now, all the dependecies are installed. The packages from this repository can now be built after cloning it into your workspace. 

Follow the instructions in the main ReadMe for using the packages. 



