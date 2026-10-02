See train_yolo_colab.ipynb (the notebook used to train this model on
Google Colab's free T4 GPU). Copy that file into this notebooks/ folder
to keep a full record of the training run alongside the project.

Training summary (epoch 50, yolo26n, imgsz=640, batch=16):
- Test set (248 images, 896 instances):
    Overall:         P=0.550  R=0.652  mAP50=0.639  mAP50-95=0.339
    Mask:            P=0.566  R=0.768  mAP50=0.725  mAP50-95=0.388
    Mask Incorrect:  P=0.586  R=0.589  mAP50=0.618  mAP50-95=0.355
    No Mask:         P=0.499  R=0.598  mAP50=0.575  mAP50-95=0.275

- Training curves (see reports/figures/training_history.png) show
  validation mAP50 converging by ~epoch 15 and plateauing around
  0.65-0.69 for the remaining 35 epochs, with mild overfitting visible
  in the classification loss (train loss keeps falling, validation loss
  flattens/slightly rises after epoch ~20). This indicates the model
  has converged given the current data and resolution - more epochs
  would not meaningfully improve it.
