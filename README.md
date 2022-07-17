# ART-SS
ART-SS: An Adaptive Rejection Technique for Semi-Supervised restoration for adverse weather-affected images

[Rajeev Yasarla](https://sites.google.com/view/rajeevyasarla/home), [Vishal M. Patel](https://engineering.jhu.edu/ece/faculty/vishal-m-patel/)

In recent years, convolutional neural network-based single image adverse weather removal methods have achieved significant performance improvements on many benchmark datasets. However, these methods require large amounts of clean-weather degraded image pairs for training, which is often difficult to obtain in practice. Although various weather degradation synthesis methods exist in the literature, the use of synthetically generated weather degraded images often results in sub-optimal performance on the real weather degraded images due to the domain gap between synthetic and real-world images. To deal with this problem, various semi-supervised restoration (SSR) methods have been proposed for deraining or dehazing which learn to restore the clean image using synthetically generated datasets while generalizing better using unlabeled real-world images. The performance of a semi-supervised method is essentially based on the quality of the unlabeled data. In particular, if the unlabeled data characteristics are very different from that of the labeled data, then the performance of a semi-supervised method degrades significantly. We theoretically study the effect of unlabeled data on the performance of an SSR method and develop a technique that rejects the unlabeled images that degrade the performance. Extensive experiments and ablation study show that the proposed sample rejection method increases the performance of existing SSR deraining and dehazing methods significantly. 

## ART-SS implementation
ART-SS is implemented in the ART_SS.py python file. ART-SS technique can be applied to any semi-supervised image restoration technique. Here we use the smilarlity index and aleatoric uncertainty to come up with rejection criteria. For computing aleatoric uncertainty we use rain and snow masks (which can be downloaded from the [dropbox link](https://www.dropbox.com/s/3jb2p3z9nt4oiu0/mask.zip?dl=0)). ART-SS technique is implemented in ART class of ART_SS.py file where it takes unlabeled and labeled dataset loaders and computes  smilarlity indices, corresponding aleatoric uncertainty sigma values, and threshold value (refer variable self.thrsh_ang) in the  gen_featmaps_unlbl and  gen_featmaps functions. Using the computed values and threshold ART class reject the unlabeled images indicated in the variable self.reject_unlabl where "1" indicates unlabeled image is not rejected and "0" indicates unlabeled image is rejected.

## Applying ART-SS to Syn2Real
Here we apply ART-SS to Syn2Real
Training command 
```
python train_new_comb.py  -train_batch_size 2  -category derain -exp_name <path_to_save_model>  -lambda_GP 0.0015 -epoch_start 0 -version version1
```
Testing command
```
python test.py -category derain -exp_name <path_to_saved_model>
```
