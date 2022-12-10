# CKPTNAME=/home/yhxu/qhzhang/didrive/demo/cilrs/noise2.pth.tar
# CKPTNAME=/home/yhxu/qhzhang/workspace/taco_exp,1_cont004_nosampler_100.pth.tar
# CKPTNAME=/home/yhxu/qhzhang/workspace/cilrs_no_color_sampler.pth.tar
CKPTNAME=/home/yhxu/qhzhang/workspace/aco.ckpt
# python cilrs_train_bn.py --shrink 20 --lr 0.0005022>22_bn 2>&1 &

CUDA_VISIBLE_DEVICES="0,1" python cilrs_train_bn.py --ckpt $CKPTNAME --shrink 20 --lr 0.00005022>22_bn 2>&1 &
CUDA_VISIBLE_DEVICES="2,3" python cilrs_train_bn.py --ckpt $CKPTNAME --shrink 10 --lr 0.00005011>11_bn 2>&1 &
CUDA_VISIBLE_DEVICES="4,5" python cilrs_train_bn.py --ckpt $CKPTNAME --shrink 5 --lr 0.0000505>5_bn 2>&1 &
CUDA_VISIBLE_DEVICES="6,7" python cilrs_train_bn.py --ckpt $CKPTNAME --shrink 2 --lr 0.0000502>2_bn 2>&1 &

# CUDA_VISIBLE_DEVICES="0,1" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 20 --lr 0.0005>0.0005 2>&1 &
# CUDA_VISIBLE_DEVICES="2,3" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 20 --lr 0.0001>0.0001 2>&1 &
# CUDA_VISIBLE_DEVICES="4,5" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 20 --lr 0.00001>0.00001 2>&1 &
# CUDA_VISIBLE_DEVICES="6,7" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 20 --lr 0.00005>0.00005 2>&1 &

# CUDA_VISIBLE_DEVICES="0,1" python cilrs_train.py --fix --shrink 20 --lr 0.0005022>0.0005_20 2>&1 &
# CUDA_VISIBLE_DEVICES="0,1" python cilrs_train.py  --shrink 20 --lr 0.0005022>0.0005_20 2>&1 &
# CUDA_VISIBLE_DEVICES="2,3" python cilrs_train.py  --shrink 10 --lr 0.0005011>0.0005_10 2>&1 &
# CUDA_VISIBLE_DEVICES="4,5" python cilrs_train.py  --shrink 5 --lr 0.000505>0.0005_5 2>&1 &
# CUDA_VISIBLE_DEVICES="6,7" python cilrs_train.py  --shrink 2 --lr 0.000502>0.0005_2 2>&1 &

# CUDA_VISIBLE_DEVICES="0,1" python cilrs_train_vae.py  --shrink 20 --lr 0.0005022>0.0005_20 2>&1 &
# CUDA_VISIBLE_DEVICES="2,3" python cilrs_train_vae.py  --shrink 10 --lr 0.0005011>0.0005_10 2>&1 &
# CUDA_VISIBLE_DEVICES="4,5" python cilrs_train_vae.py  --shrink 5 --lr 0.000505>0.0005_5 2>&1 &
# CUDA_VISIBLE_DEVICES="6,7" python cilrs_train_vae.py  --shrink 2 --lr 0.000502>0.0005_2 2>&1 &
