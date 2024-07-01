

# IoT evaluation experiments


## Install FFCV to speed up dataloading

1. Install FFCV
Note, FFCV currently works with Python<=3.11, see https://github.com/libffcv/ffcv
Bascially, you need to install the following packages:

```Shell
conda install cupy pkg-config libjpeg-turbo opencv numba
```

2.Then you install FFCV

```Shell
pip install ffcv
```

3. Convert your Pollen dataset into FFCV format

```Shell
python ffcv_prepare.py
```

## Train one4all, one4one, and SCN models

You can go to [run3.sh](run3.sh) and run training scripts to train one4all, one4one, and SCN models.

## Change data transformation or image size

You can go to [ffcv_dataset.py](ffcv_dataset.py) and modify the  `load_ffcv_data` function.

## Install packages for model conversion from Pytorch to TF Lite


```Shell
wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
&& tar -zxvf flatc.tar.gz \
&& chmod +x flatc \
&& mv flatc ~/bin
```

```Shell
pip install onnx-simplifier
```

```Shell
pip install -U onnx==1.15.0 \
&& pip install -U nvidia-pyindex \
&& pip install -U onnx-graphsurgeon \
&& pip install -U onnxruntime==1.17.1 \
&& pip install -U onnxsim==0.4.33 \
&& pip install -U simple_onnx_processing_tools \
&& pip install -U sne4onnx \
&& pip install -U sng4onnx \
&& pip install -U tensorflow==2.16.1 \
&& pip install -U protobuf==3.20.3 \
&& pip install -U onnx2tf \
&& pip install -U h5py==3.11.0 \
&& pip install -U psutil==5.9.5 \
&& pip install -U ml_dtypes==0.3.2 \
&& pip install -U tf-keras~=2.16 \
&& pip install flatbuffers==23.5.26
```

data shape: (batch_size, 3, 96, 96)

## Important note

If you changed the img size of data, please also adjut the   `dummy_input` in `imgtrans_scn_pollen_modelconversion.py` and 

## Conversion for LeNet models

### Convert One4All model and its variants Pytoch-->ONNX
```Shell
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_one4all_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_one4all_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet2xwider
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_one4all_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet4xwider
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_one4all_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --arch=lenet8xwider
```

### ONNX-->TFLite-->TFLite Micro
```Shell
onnx2tf -i output/brightness/One4All/lenet_2_32/model.onnx  -v info -o output/brightness/One4All/lenet_2_32/
xxd -i output/brightness/One4All/lenet_2_32/model_float32.tflite > output/brightness/One4All/lenet_2_32/model_data.cc
tail -n1 output/brightness/One4All/lenet_2_32/model_data.cc

onnx2tf -i output/brightness/One4All/lenet2xwider_2_32/model.onnx  -v info -o output/brightness/One4All/lenet2xwider_2_32/
xxd -i output/brightness/One4All/lenet2xwider_2_32/model_float32.tflite > output/brightness/One4All/lenet2xwider_2_32/model_data.cc
tail -n1 output/brightness/One4All/lenet2xwider_2_32/model_data.cc

onnx2tf -i output/brightness/One4All/lenet4xwider_2_32/model.onnx  -v info -o output/brightness/One4All/lenet4xwider_2_32/
xxd -i output/brightness/One4All/lenet4xwider_2_32/model_float32.tflite > output/brightness/One4All/lenet4xwider_2_32/model_data.cc
tail -n1 output/brightness/One4All/lenet4xwider_2_32/model_data.cc

onnx2tf -i output/brightness/One4All/lenet8xwider_2_32/model.onnx  -v info -o output/brightness/One4All/lenet8xwider_2_32/
xxd -i output/brightness/One4All/lenet8xwider_2_32/model_float32.tflite > output/brightness/One4All/lenet8xwider_2_32/model_data.cc
tail -n1 output/brightness/One4All/lenet8xwider_2_32/model_data.cc
```


## Conversion for SCN-LeNet models

### Convert hypernet and base models of  SCN(D=3 and D=5) Pytoch-->ONNX

```Shell
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_scn_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --dimension=3 --arch=scn_lenet
CUDA_VISIBLE_DEVICES=6 python3 imgtrans_scn_pollen_modelconversion.py --batchsize=32 --output=output --nlayers=2 --width=32 --transform=brightness --dimension=5 --arch=scn_lenet
```

### Conversion for hypernet of SCN(D=3 and D=5): ONNX-->TFLite

```Shell
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_3/model_hypernet.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_3/hypernet
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/model_hypernet.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/hypernet
```

### Conversion for basemodels: ONNX-->TFLite

#### Basemodels of SCN(D=3)
```Shell
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel0.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_3/bsmodel0
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel1.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_3/bsmodel1
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel2.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_3/bsmodel2
```

#### Basemodels of SCN(D=5) 

```Shell
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel0.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/bsmodel0
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel1.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/bsmodel1
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel2.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/bsmodel2
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel3.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/bsmodel3
onnx2tf -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel4.onnx -v info -o output/brightness/SCN/scn_lenet_2_32_5/bsmodel4
```

### Conversion for SCNs: TFLite-->TFLite Micro

#### SCN(D=3)
```Shell
xxd -i output/brightness/SCN/scn_lenet_2_32_3/hypernet/model_hypernet_float32.tflite > output/brightness/SCN/scn_lenet_2_32_3/hypernet/hypernet_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel0/bsmodel0_float32.tflite > output/brightness/SCN/scn_lenet_2_32_3/bsmodel0/bsmodel0_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel1/bsmodel1_float32.tflite > output/brightness/SCN/scn_lenet_2_32_3/bsmodel1/bsmodel1_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_3/bsmodel2/bsmodel2_float32.tflite > output/brightness/SCN/scn_lenet_2_32_3/bsmodel2/bsmodel2_data.cc
```

#### SCN(D=5)
```Shell
xxd -i output/brightness/SCN/scn_lenet_2_32_5/hypernet/model_hypernet_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/hypernet/hypernet_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel0/bsmodel0_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/bsmodel0/bsmodel0_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel1/bsmodel1_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/bsmodel1/bsmodel1_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel2/bsmodel2_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/bsmodel2/bsmodel2_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel3/bsmodel3_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/bsmodel3/bsmodel3_data.cc
xxd -i output/brightness/SCN/scn_lenet_2_32_5/bsmodel4/bsmodel4_float32.tflite > output/brightness/SCN/scn_lenet_2_32_5/bsmodel4/bsmodel4_data.cc
```
