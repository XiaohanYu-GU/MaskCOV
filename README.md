
# MaskCOV
![image](https://user-images.githubusercontent.com/9549469/119601776-36f89a80-be2d-11eb-8535-effb67a1f9e3.png)
<!-- ![image](https://user-images.githubusercontent.com/9549469/119601877-660f0c00-be2d-11eb-91c9-307b02e448dd.png) -->


## Basic information
This work is published in Pattern Recognition. Please cite the following paper should you consider to use this code.
* Xiaohan Yu, Yang Zhao, Yongsheng Gao, Shengwu Xiong (2021). MaskCOV: A Random Mask Covariance Network for Ultra-Fine-Grained Visual Categorization. In Pattern Recognition.

@article{yu2021maskcov,
  title={MaskCOV: A Random Mask Covariance Network for Ultra-Fine-Grained Visual Categorization},
  author={Yu, Xiaohan and Zhao, Yang and Gao, Yongsheng and Xiong, Shengwu},
  journal={Pattern Recognition},
  pages={108067},
  year={2021},
  publisher={Elsevier}
}

## Source Download
Please find our code in the folder PR_MaskCOV. The ultra-fine-grained image dataset, UFG, used in this paper can be downloaded via "https://github.com/XiaohanYu-GU/Ultra-FGVC".

### How to use
install pytorch 1.6.0, python 3.7, cuda 10.1, cudnn7.6.3 and any necessary python package that is required.

Use the following order to run the training code in a default setting.

"sh main.sh"

Or revise the hyper-parameters (batch size, learning rate) in config.py if needed and then run "sh main.sh".

### Note
For Cotton80 subset, the batch size is recommended to be 8. For the remaining subsets, the batch size is recommended to be 16.


### Acknowledgement
The code is revised based on source code provided by DCL (see "https://github.com/JDAI-CV/DCL"). We sincerely thank their contribution.


## Author contact info
*Xiaohan Yu*, *yuxiaohan112@gmail.com*
