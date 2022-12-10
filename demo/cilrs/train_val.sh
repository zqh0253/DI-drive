CKPTNAME=taco_exp,1_cont004_nosampler.pth.tar
CUDA_VISIBLE_DEVICES="0,1" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 20 --lr 0.0001022>0.0001_22_bn 2>&1
CUDA_VISIBLE_DEVICES="2,3" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 10 --lr 0.0001011>0.0001_11_bn 2>&1
CUDA_VISIBLE_DEVICES="4,5" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 5 --lr 0.000105>0.0001_5_bn 2>&1 
CUDA_VISIBLE_DEVICES="6,7" python cilrs_train_bn.py --ckpt /home/yhxu/qhzhang/workspace/$CKPTNAME --shrink 2 --lr 0.000102>0.0001_2_bn 2>&1 

python cilrs_eval.py
