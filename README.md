
# Depth Conrollable SRNet-Tensorflow
Tensorflow implemetation of depth conrllable SRNet 

We propose a network which can controll the depth of layer without any changes of weights during runtime.
This network can be useful to adatp speed/performance trade-off in stochstic realtime super resolution system such like video streaming for HR TV.

Detail describtion will be added after paper acceptance or uploading to arxiv.


</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/model.png" width="800">
</p>


## Prerequisites
 * python 3.x
 * Tensorflow > 1.5
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * matplotlib
 * argparse
 * opencv

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3
 * This code supports only RGB color images (demo type) and Ychannel of YCbCr (eval type) 
 * This code supports data augmentation (rotation and mirror flip)
 * This code supports custom dataset


## Usage
```
=============================================================================
usage: main.py [-h] [--exp_tag EXP_TAG] [--gpu GPU] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--patch_size PATCH_SIZE]
               [--base_lr BASE_LR] [--lr_min LR_MIN]
               [--num_additional_inputlayer NUM_ADDITIONAL_INPUTLAYER]
               [--num_hiddens NUM_HIDDENS] [--num_innerlayer NUM_INNERLAYER]
               [--num_additional_branchlayer NUM_ADDITIONAL_BRANCHLAYER]
               [--lr_decay_rate LR_DECAY_RATE] [--lr_step_size LR_STEP_SIZE]
               [--scale SCALE] [--checkpoint_dir CHECKPOINT_DIR]
               [--cpkt_itr CPKT_ITR] [--save_period SAVE_PERIOD]
               [--result_dir RESULT_DIR] [--train_subdir TRAIN_SUBDIR]
               [--test_subdir TEST_SUBDIR] [--infer_subdir INFER_SUBDIR]
               [--infer_imgpath INFER_IMGPATH] [--type {eval,demo}]
               [--c_dim C_DIM] [--mode {train,test,inference,test_plot}]
               [--train_depth TRAIN_DEPTH] [--save_extension {jpg,png}]
=============================================================================
```

 * For training, `python3 main.py --mode train --check_itr 0` [set 0 for training from scratch, -1 for latest]
 * For testing, `python 3 main.py --mode test --check_itr 40`
 * For inference with cumstom dataset, `python3 main.py --mode inference --infer_imgpath monarch.bmp` [result will be generated in ./result/inference]
 * For running tensorboard, `tensorboard --logdir=./board` then access localhost:6006 with your browser
 * For demo `python 3 demo2.py`
## Result

</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/DASR%20test.png" width="800">
</p>

</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/result.png" width="800">
</p>



</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/result2.png" width="800">
</p>


</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/result2.png" width="800">
</p>


</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/tradeoff.png" width="800">
</p>


</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_Depth_Controllable_SRNet_Tensorflow/master/asset/demo.png" width="800">
</p>


## References
* Related works
* [Model adaptation]() : Low-Complexity Online Model Selection with Lyapunov Control for Reward Maximization in Stabilized Real-Time Deep Learning Platforms (Dohyun Kim, Joongheon Kim, Joonseok Kwon, SMC 2018)

## ToDo
* link pretrained models [done]
* link dataset [done]
* add describtion (not yet, please refer ppt in asset dir)

## Author
Dohyun Kim



