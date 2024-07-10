docker run --gpus all -iPt \
    --volume="$PWD:/neugraspnet" \
    --net=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTHORITY:$XAUTHORITY" \
    --env="XAUTHORITY=$XAUTHORITY" \
    --name="neugrasp_container" \
    --rm \
    neugrasp \
    bash
