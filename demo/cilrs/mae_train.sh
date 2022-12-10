for i in {1..3}
do
    python cilrs_train_mae.py --ckpt /home/yhxu/qhzhang/mae/output_dir/checkpoint-799.pth --lr 0.00005 --shrink 10
    python cilrs_mae_eval.py  --lr 5e-05
    python cilrs_train_mae.py --ckpt /home/yhxu/qhzhang/mae/output_dir/checkpoint-799.pth --lr 0.00005 --shrink 5
    python cilrs_mae_eval.py  --lr 5e-05
    python cilrs_train_mae.py --ckpt /home/yhxu/qhzhang/mae/output_dir/checkpoint-799.pth --lr 0.00005 --shrink 2
    python cilrs_mae_eval.py  --lr 5e-05
done
