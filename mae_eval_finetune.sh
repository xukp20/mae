# make sure you have the checkpoint
#   run the training script first or download from the cloud
python cifar10_finetune.py --finetune=output_dir/checkpoint-mae_pretrain.pth --batch_size=2048 --gpu_id=0