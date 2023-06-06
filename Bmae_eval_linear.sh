# make sure you have the checkpoint
#   run the training script first or download from the cloud
python cifar10_linprobe.py --finetune=output_dir/checkpoint-bmae_pretrain.pth --batch_size=2048 --blr=0.1 --gpu_id=5