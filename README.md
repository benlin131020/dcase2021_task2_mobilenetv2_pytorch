# dcase2021_task2_mobilenetv2_pytorch
## Usage

### 1. Clone repository
```
$ git clone https://github.com/benlin131020/dcase2021_task2_mobilenetv2_pytorch.git
```

### 2. Python 環境
建立 Python 環境
```
$ conda create -n myenv python=3.6
```

進入 Python 環境
```
$ conda activate myenv
```

離開 Python 環境
```
$ conda deactivate
```

### 3. 安裝 Python 套件
進入 Python 環境後
```
$ pip install -r requirements.txt
```

### 4. Run
指定使用第幾顆 GPU
```
$ CUDA_VISIBLE_DEVICES=1 bash run.sh
```
`run.sh` 包含了訓練、擷取 embedding、訓練 LOF 以及測試

### 5. yaml 設定
在 `baseline.yaml` 中:
- `exp_directory: ./exp/linear`
    - 可以改成 `./exp/my_name` 避免覆蓋掉之前的實驗
- `linear: 0`
    - `0` 為使用 mel-scale, `1` 為使用 linear-scale