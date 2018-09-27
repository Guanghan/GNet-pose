GNet-pose
=======
Project Page: [http://guanghan.info/projects/guided-fractal/](http://guanghan.info/projects/Guided-Fractal/)

UPDATE 9/27/2018:
Prototxts and model that achieved 93.9Pck on LSP dataset. 
[http://guanghan.info/download/Data/GNet_update.zip](http://guanghan.info/download/Data/GNet_update.zip)

When I was replying e-mails, it occurred to me that the models that I had uploaded was around May/June 2017 (performance in old arxiv version), and in August 2017 the performance was improved to 93.9 on LSP with a newer caffe version which fixed the downsampling and/or upsampling deprecation problem (Yeah, it "magically" improved the performance).
The best model was 94.0071 on LSP dataset, but it was not uploaded nor published on the benchmark. 

--------
## Overview
Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation.

Source code release of the paper for reproduction of experimental results, and to aid researchers in future research.

----
## Prerequisites
- Python 2.7 or Python 3.3+
- [Modified Caffe](http://github/Guanghan/GNet-caffe)

----
## Getting Started

### 1. Download Data and Pre-trained Models

- **Datasets ([MPII](http://human-pose.mpi-inf.mpg.de/#overview) [[1]], [LSP](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) [[2]])**

  ```
  bash ./get_dataset.sh
  ```

- **Models**
  ```
  bash ./get_models.sh
  ```

- **Predictions (optional)**
  ```
  bash ./get_preds.sh
  ```

### 2. Testing

- Generate cropped patches from the dataset for testing:
  ```
  cd testing/
  matlab gen_cropped_LSP_test_images.m
  matlab gen_cropped_MPII_test_images.m
  cd -
  ```
  This will generate images with 368-by-368 resolution.

- Reproduce the results with the pre-trained model:

  ```
  cd testing/
  python .test.py
  cd -
  ```
  You can choose different dataset to test on, with different models. You can also choose different settings in test.py, e.g., with or without flipping, scaling, cross-heatmap regression, etc.

### 3. Training

- Generate Annotations
  ```
  cd training/Annotations/
  matlab MPI.m LEEDS.m
  cd -
  ```
  This will generate annotations in json files.


- Generate LMDB
  ```
  python ./training/Data/genLMDB.py
  ```
  This will load images from dataset and annotations from json files, and generate lmdb files for caffe training.


- Generate Prototxt files (optional)

   ```
   python ./training/GNet/scripts/gen_GNet.py
   python ./training/GNet/scripts/gen_fractal.py
   python ./training/GNet/scripts/gen_hourglass.py
   ```

- Training:

	```
	bash ./training/train.sh
	```

### 4. Performance Evaluation

	cd testing/eval_LSP/; matlab test_evaluation_lsp.m; cd../
  cd testing/eval_MPII/; matlab test_evaluation_mpii_test.m

### 5. Results

More Qualitative results can be found in the project page.  Quantitative results please refer to the arxiv paper.

![](http://guanghan.info/projects/Guided-Fractal/mpii-results.png)

---
## License

GNet-pose is released under the Apache License Version 2.0 (refer to the LICENSE file for details).

---
## Citation
If you use the code and models, please cite the following paper:
[TMM 2017](http://ieeexplore.ieee.org/document/8064661/).

	@article{ning2017knowledge, 
	 author={G. Ning and Z. Zhang and Z. He}, 
         journal={IEEE Transactions on Multimedia}, 
         title={Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation}, 
         year={2017}, 
         doi={10.1109/TMM.2017.2762010}, 
         ISSN={1520-9210}, }


---
## Reference
[[1]] Andriluka M, Pishchulin L, Gehler P, et al. "2d human pose estimation: New benchmark and state of the art analysis." CVPR (2014).

[1]: https://www.d2.mpi-inf.mpg.de/sites/default/files/andriluka14cvpr.pdf "MPII"

[[2]] Sam Johnson and Mark Everingham. "Clustered Pose and Nonlinear Appearance
Models for Human Pose Estimation." BMVC (2010).

[2]: http://www.comp.leeds.ac.uk/mat4saj/publications/johnson10bmvc.pdf "LSP"
