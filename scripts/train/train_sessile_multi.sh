time=$(date "+%Y%m%d_%H%M%S")
# The current settings are for 8 GPUs. If you have less GPUs, please change CUDA_VISIBLE_DEVICES and nproc_per_node.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=21482 --nproc_per_node=8 main.py \
--method imedsam \
--model_type "vit_b" \
--checkpoint "./sam_ckp/sam_vit_b_01ec64.pth" \
--data_path "./dataset/sessile-Kvasir" \
--work_dir "./work_dir/sessile-Kvasir/train_debug_${time}" \
--image_size 384 \
--label_size 384 \
--num_epochs 1000 \
--use_amp \
--dist-url env://localhost:12345 \
--distributed \