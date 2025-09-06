accelerate launch --config_file configs/accelerate_config.yaml aws_train_gliclass_audio.py

# accelerate launch \
#     --mixed_precision bf16 \
#     --num_processes 2 \
#     --num_machines 1 \
#     aws_train_gliclass_audio.py \
#     --bf16 True \
#     --max_length 2048 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --encoder_lr 1e-5 \
#     --audio_lr 1e-5 \
#     --others_lr 1e-5 \
#     --save_steps 1000