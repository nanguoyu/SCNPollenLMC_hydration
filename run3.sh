#!/bin/bash



OPT="--batchsize=64 --epochs=200 --learning_rate=0.0001 --output=output --nlayers=2 --width=32 --transform=brightness"
CUDA_VISIBLE_DEVICES=1 python3 imgtrans_scn_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness  --arch=scn_lenet --dimensions=1
CUDA_VISIBLE_DEVICES=2 python3 imgtrans_scn_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness  --arch=scn_lenet --dimensions=2
CUDA_VISIBLE_DEVICES=3 python3 imgtrans_scn_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness  --arch=scn_lenet --dimensions=3
CUDA_VISIBLE_DEVICES=4 python3 imgtrans_scn_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness  --arch=scn_lenet --dimensions=5
CUDA_VISIBLE_DEVICES=5 python3 imgtrans_scn_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness  --arch=scn_lenet --dimensions=8

CUDA_VISIBLE_DEVICES=3 python3 imgtrans_one4all_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet
CUDA_VISIBLE_DEVICES=7 python3 imgtrans_inverse_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet
CUDA_VISIBLE_DEVICES=2 python3 imgtrans_one4one_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet
CUDA_VISIBLE_DEVICES=3 python3 linearconnect.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet


# 2xwider one4all
CUDA_VISIBLE_DEVICES=4 python3 imgtrans_one4all_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet2xwider
# 4xwider one4all
CUDA_VISIBLE_DEVICES=5 python3 imgtrans_one4all_pollen.py --batchsize=64 --epochs=100 --learning_rate=0.001 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet4xwider

python3 plot.py
