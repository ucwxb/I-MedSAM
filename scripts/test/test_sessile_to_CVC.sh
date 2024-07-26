# To run the code on multi-gpu, please use the following command:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=21482 --nproc_per_node=8 main.py \
python main.py \
--method imedsam \
--model_type "vit_b" \
--checkpoint "./sam_ckp/sam_vit_b_01ec64.pth" \
--data_path "./dataset/CVC" \
--label_size 384 \
--use_amp \
--dist-url env://localhost:12345 \
--distributed \
--test_only \
--resume "./work_dir/model_best.pth" \
# --save_pic \ # for visulization