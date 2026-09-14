export CUDA_VISIBLE_DEVICES=0,1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# NCCL_DEBUG=INFO

export EXP_NAME="DTW-regnetv2-VisDrone"

torchrun --master_port=7789 --nproc_per_node=2 train.py \
     -c src/zoo/DTW_Det/configs/dtwdet/dtwdet_regnetv2_visdrone.yml --seed=0  2>&1 | tee "logs/${EXP_NAME}-$(date +%Y%m%d_%H%M%S).log"