time=$(date "+%Y%m%d_%H%M%S")
python main.py \
--method imedsam \
--model_type "vit_b" \
--checkpoint "./sam_ckp/sam_vit_b_01ec64.pth" \
--data_path "./dataset/sessile-Kvasir" \
--work_dir "./work_dir/sessile-Kvasir/train_debug_${time}" \
--image_size 384 \
--label_size 384 \
--num_epochs 1000 \
--use_amp \