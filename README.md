# Guidance of using the hand gesture demo

## Dendencies and Environment

- Python 3.7
- OpenCV 4.1.0
- tensorflow 1.14.0

## Model Path

the inference code has implemented two cascaded ML models. First, the MTCNN network and then a simple CNN network.

The network structure of MTCNN is defined in the `Train_Model` folder, while the CNN network structure is coded in the inference file.

MTCNN model weights are stored in the `MTCNN_Model` folder, P-Net, R-Net, O-Net separately.
CNN model weights are stored in the `CNN_Model` folder, please refer to the checkpoint file in it.

If you want to change the path, just change the value of 
```python
model_path = ['MTCNN_Model/PNet/PNet-30', 'MTCNN_Model/RNet/RNet-22', 'MTCNN_Model/ONet/ONet-22']
model = load_classification('CNN_Model/kcnn')
```
inside the inference code.

## Run camera inference

just run `camera.py` for hand detection (MTCNN only) or `camera_c.py` for hand gesture classification (both MTCNN and CNN).

hand gesture classes are defined below:

```python
class_names = ['0SingleOne', '1SingleTwo', '2SingleFour', '3SingleSix',
               '4SingleEight', '5SingleNine', '6SingleBad', '7SingleGood']
```

The classification is not of good performance in this beta version.