# to train the unequally distributed BMAE model
python cifar10_bootstrap_pretrain_extra.py --batch_size=1024 --gpu_id=0 --norm_pix_loss --epochs=200
# eval
python cifar10_finetune.py --finetune=output_dir/checkpoint-bmae_pretrain_unequal.pth --batch_size=2048 --gpu_id=0
python cifar10_linprobe.py --finetune=output_dir/checkpoint-bmae_pretrain_unequal.pth --batch_size=2048 --blr=0.1 --gpu_id=0