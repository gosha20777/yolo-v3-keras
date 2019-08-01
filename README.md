# yolo-v3-keras
Keras implementation of YOLO v3 https://pjreddie.com/media/files/papers/YOLOv3.pdf.

## Pretrained model
You can download offitial pascal pretrained model [here](https://github.com/gosha20777/yolo-v3-keras)

## Ussage
To make prediction use this command:

```
git clone https://github.com/gosha20777/yolo-v3-keras.git
cd yolo-v3-keras
wget -O voc.h5 https://github.com/gosha20777/yolo-v3-keras/releases/download/0.1.0/voc.h5
python predict.py -c config.json -i /path/to/image_dir -o /patch/to/output_dir
```

## Training
1. Load the VOC training set http://host.robots.ox.ac.uk/pascal/VOC/voc2012/.

2. Edit `config.json` file and chnge `train_image_folder`, `train_annot_folder`, `valid_image_folder` and `val_annot_folder` patches.

3. Run:
   `python train.py -c config.json`

## Training on your own data
1. Create your VOC-style dataset. (You can do it with [labelImg](https://github.com/tzutalin/labelImg) program).
2. Edit classes at `config.json`.
3. Train it!
*I highly recommend you to use default pretrained weights and fine-tune it*
