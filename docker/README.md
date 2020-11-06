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

## Run Container

``` sh docker_run.sh ```



