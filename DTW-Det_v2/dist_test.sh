export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_SOCKET_IFNAME=lo
# export SAVE_TEST_VISUALIZE_RESULT=False
# export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
torchrun --master_port=7778 --nproc_per_node=2 train.py -c src/zoo/DTW_Det/configs/dtwdet/dtwdet_regnetv2_visdrone.yml --test-only -r output/dtwdet_regnetv2_visdrone_DTW_Det/checkpoint0107.pth