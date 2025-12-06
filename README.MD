ros2 launch mlp_controller mlp_controller.launch.py \
    njoints:=3 \
    model_path:=/path/to/mlp_model_best3.pth \
    scaler_path:=/path/to/scalers_3joints.pkl
