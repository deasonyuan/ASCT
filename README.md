## ASCT
***Visual object tracking with adaptive structural convolutional network***

**Abstract:** Convolutional Neural Networks (CNN) have been demonstrated to achieve state-of-the-art performance in visual object tracking task. However, existing CNN-based trackers usually use holistic target samples
to train their networks. Once the target undergoes complicated situations (e.g., occlusion, background clutter, and deformation), the tracking performance degrades badly. In this paper, we propose an adaptive structural convolutional filter model to enhance the robustness of deep regression trackers (named: ASCT). Specifically, we first design a mask set to generate local filters to capture local structures of the target. Meanwhile, we adopt an adaptive weighting fusion strategy for these local filters to adapt to the changes in the target appearance, which can enhance the robustness of the tracker effectively. Besides, we develop an end-to-end trainable network comprising feature extraction, decision making, and model updating modules for effective training. Extensive experimental results on large benchmark datasets demonstrate the effectiveness of the proposed ASCT tracker performs favorably against the state-of-the-art trackers.

 :star: Update:The code for ASCT tracker can be downloaded [here[baidu(password:)]]().

## Usage
### Tracking
1 Download the code and unzip it in your computer.

2 Run the demo.m to test a tracking sequence using a default model.
   

## Citation
If you find the code useful, please cite:
```
@article{yuan2020visual,
  title={Visual object tracking with adaptive structural convolutional network},
  author={Yuan, Di and Li, Xin and He, Zhenyu and Liu, Qiao and Lu, Shuwei},
  journal={Knowledge-Based Systems},
  pages={https://doi.org/10.1016/j.knosys.2020.105554},
  year={2020},
}

@inproceedings{song-iccv17-CREST,
    author={Song, Yibing and Ma, Chao and Gong, Lijun and Zhang, Jiawei and Lau, Rynson and Yang, Ming-Hsuan}, 
    title={CREST: Convolutional Residual Learning for Visual Tracking}, 
    booktitle={IEEE International Conference on Computer Vision},
    pages={2555-2564},
    year={2017}
}

```

## Contact
Feedbacks and comments are welcome! Feel free to contact us via dyuanhit@gmail.com


## Acknowledgements
Some of the parameter settings and functions are borrowed from CREST(https://github.com/ybsong00/CREST-Release). 
