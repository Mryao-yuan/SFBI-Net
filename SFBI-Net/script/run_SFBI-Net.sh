#!/usr/bin/env bash

gpus=5

checkpoint_root=output/checkpoints
vis_root=output/vis

data_name=LEVIR	#LEVIR, EGY_BCD CLCD-256

img_size=256		
batch_size=4
lr=0.01
max_epochs=200
net_G=SFBIN

lr_policy=linear
optimizer=sgd		                    
loss=ce


sfd_loss_weight=0.01
multi_scale_train=False
multi_scale_infer=False
shuffle_AB=False
add_loss=True
split=train
split_val=test
project_name=${net_G}_${data_name}_b${batch_size}_optimizer_${optimizer}_loss${loss}_lr${lr}_LW${sfd_loss_weight}

python train.py --img_size ${img_size} --loss ${loss} --optimizer ${optimizer} --sfd_loss_weight ${sfd_loss_weight} --add_loss ${add_loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
