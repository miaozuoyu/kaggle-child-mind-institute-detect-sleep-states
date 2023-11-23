# Child Mind Institute - Detect Sleep States

This repository is for [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## Build Environment
### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

### Set path
Rewrite run/conf/dir/local.yaml to match your environment

```yaml
data_dir: 
processed_dir: 
output_dir: 
model_dir: 
sub_dir: ./
```

## Prepare Data

### 1. Download data

```bash
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
```

### 2. Preprocess data

```bash
python run/prepare_data.py -m phase=train,test
```

## Train Model
The following commands are for training the model of LB0.714
```bash
python run/train.py downsample_rate=2 duration=5760 exp_name=exp001 dataset.batch_size=32
python run/train.py downsample_rate=2 duration=5760 exp_name=exp_backbone dataset.batch_size=32 model.params.encoder_name=efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7
python run/train.py exp_name=exp_b2 model.params.encoder_name=efficientnet-b2 optimizer.lr=0.0001 trainer.epochs=100

python run/train.py exp_name=exp002 model.params.encoder_name=efficientnet-b2 optimizer.lr=0.0003 trainer.epochs=100
python run/train.py exp_name=exp003 optimizer.lr=0.0003 trainer.epochs=100
python run/train.py exp_name=exp004 dataset.batch_size=8 optimizer.lr=0.0005 trainer.epochs=100
python run/train.py exp_name=exp005 dataset.batch_size=8 optimizer.lr=0.0005 trainer.epochs=100 feature_extractor.params.base_filters=128
python run/train.py exp_name=exp006 feature_extractor.params.base_filters=128
python run/train.py exp_name=exp007 model.params.encoder_name=efficientnet-b2 feature_extractor.params.base_filters=128
python run/train.py exp_name=exp008 feature_extractor.params.base_filters=128 feature_extractor.params.kernel_sizes=[64,32,16,2]
python run/train.py exp_name=exp009 feature_extractor.params.base_filters=128 feature_extractor.params.kernel_sizes=[128,64,32,2]
python run/train.py exp_name=exp010 feature_extractor.params.base_filters=128 feature_extractor.params.kernel_sizes=[128,64,2]
```

You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with downsample_rate of 2, 4, 6, and 8.

```bash
python run/train.py -m downsample_rate=2,4,6,8
python run/train.py -m downsample_rate=2,3,4 exp_name=dummy2
python run/train.py -m downsample_rate=2 exp_name=cnn_feature
python run/train.py -m exp_name=all_feature dataset.batch_size=4,8,16,32 optimizer.lr=0.0001,0.001,0.0005,0.00005 
python run/train.py -m exp_name=all_feature2 dataset.batch_size=4,8,16,32 optimizer.lr=0.0001,0.001,0.0005,0.00005 scheduler.num_warmup_steps=0.1 
python run/train.py -m exp_name=exp_backbone model.params.encoder_name=efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7 trainer.epochs=100
python run/train.py -m exp_name=exp_backbone_long model.params.encoder_name=resnet34,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7 trainer.epochs=100

```

## Upload Model
```bash
python tools/upload_dataset.py
```

## Inference
The following commands are for inference of LB0.714 
```bash
rye run python run/inference.py dir=kaggle exp_name=exp001 weight.run_name=single downsample_rate=2 duration=5760 model.params.encoder_weights=null pp.score_th=0.005 pp.distance=40 phase=test
```

## Implemented models

The model is built with two components: feature_extractor and decoder.

The feature_extractor and decoder that can be specified are as follows

### Feature Extractor

- [CNNSpectrogram](https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch)
- LSTMFeatureExtractor
- [PANNsFeatureExtractor](https://arxiv.org/abs/1912.10211)
- SpecFeatureExtractor

### Decoder

- MLPDecoder
- LSTMDecoder
- TransformerDecoder
- TransformerCNNDecoder
- [UNet1DDecoder](https://github.com/bamps53/kaggle-dfl-3rd-place-solution/blob/master/models/cnn_3d.py)
- MLPDecoder

### Model

- Spec2DCNN: Segmentation through UNet.
- Spec1D: Segmentation without UNet
- DETR2DCNN: Use UNet to detect sleep as in [DETR](https://arxiv.org/abs/2005.12872). This model is still under development.
- CenterNet: Detect onset and offset, respectively, like [CenterNet](https://arxiv.org/abs/1904.07850) using UNet
- TransformerAutoModel: 
  - Segmentation using huggingface's [AutoModel](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html). don't use feature_extractor and decoder.
  -  Since the Internet is not available during inference, it is necessary to create a config dataset and specify the path in the model_name.

The correspondence table between each model and dataset is as follows.

| model     | dataset   | 
| --------- | --------- | 
| Spec1D    | seg       | 
| Spec2DCNN | seg       | 
| DETR2DCNN | detr      | 
| CenterNet | centernet | 
| TransformerAutoModel | seg |

The command to train CenterNet with feature_extractor=CNNSpectrogram, decoder=UNet1DDecoder is as follows

```bash
rye run python run/train.py model=CenterNet dataset=centernet feature_extractor=CNNSpectrogram decoder=UNet1DDecoder
```