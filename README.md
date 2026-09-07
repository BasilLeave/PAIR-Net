# Rethinking Multilabel Remote Sensing Image Classification under Dual-Distribution Instance-Dependent Noise

## Abstract

Multilabel scene classification (MLC) is a fundamental task in remote sensing (RS). However, its performance is often limited by noisy annotations, where subtractive noise occurs when inconspicuous targets are missed and additive noise arises when visually confusing categories are falsely added. To simulate such annotation errors, the recently proposed Instance-Dependent Multilabel Noise (IDMN) method introduces the first instance-dependent noise generation strategy for RS MLC, driven by the intuition that hard label instances are more susceptible to mislabeling. Specifically, it leverages a single probability distribution of present labels to derive thresholds for both subtractive and additive noise generation. This design determines additive noise by requiring absent-label occurrence probabilities to exceed an irrelevant present-label threshold, leaving its quantity difficult to control and overlooking potentially confusing absent labels. To address this limitation, we propose Dual-Distribution Instance-Dependent Multilabel Noise (DD-IDMN), which decouples present-label and absent-label probability distributions for calibrated noise generation. Empirical analysis under DD-IDMN reveals that additive noise yields more severe performance degradation than subtractive noise. To combat this DD-IDMN challenge, we propose the Prompt-guided Activation Interference Rectification Network (PAIR-Net). Through cross-modal attention with learnable text prompts, it learns more discriminative RS feature representations. Furthermore, we introduce an activation-aware dynamic filtering module to identify unreliable present labels for suppressing the more destructive additive noise under DD-IDMN. Experiments on UCMerced, AID, and MLRSNet demonstrate that PAIR-Net consistently outperforms representative noise-robust multilabel methods under complex instance-dependent noise.

## Usage

### Step 1. Install python dependencies.
```
pip install -r requirements.txt
```

### Step 2. Download datasets images and multilabel annotations.

- UCMerced : http://weegee.vision.ucmerced.edu/datasets/landuse.html

- MLRSNet : https://data.mendeley.com/datasets/7j9bv9vwsx/3

- AID : https://github.com/Hua-YS/AID-Multilabel-Dataset

The resulting directory hierarchy is: 

```
data
|--- UCMerced_LandUse
|    |--- Images
|    |--- CAMs
|    |--- formatted_train_labels.npy
|    |--- formatted_train_images.npy
|    |--- formatted_test_labels.npy
|    |--- formatted_test_images.npy
|    |--- clip_logits.npy
|    |--- clip_adjacency.npy
|--- AID_multilabel
|    |--- images_tr
|    |--- images_test
|    |--- CAMs
|    |--- formatted_train_labels.npy
|    |--- formatted_train_images.npy
|    |--- formatted_test_labels.npy
|    |--- formatted_test_images.npy
|    |--- clip_logits.npy
|    |--- clip_adjacency.npy
|--- MLRSNet
|    |--- Images
|    |--- CAMs
|    |--- formatted_train_labels.npy
|    |--- formatted_train_images.npy
|    |--- formatted_test_labels.npy
|    |--- formatted_test_images.npy
|    |--- clip_logits.npy
|    |--- clip_adjacency.npy
```

### Step 4. Run experiments.

For PAIR-Net:
```
python train_PAIR-Net.py --dataset [dataset] \
                         --noise_type [noise_type] \
                         --noise_rate [noise_rate]
```
For BCE, LCR, SAT, and ELR:
```
python train_BCE.py --dataset [dataset] \
                    --scheme [scheme] \
                    --noise_type [noise_type] \
                    --noise_rate [noise_rate]
```

For JoCoR:
```
python train_jocor.py --dataset [dataset] \
                      --noise_type [noise_type] \
                      --noise_rate [noise_rate]
```

For RCML:
```
python train_rcml.py --dataset [dataset] \
                     --noise_type [noise_type] \
                     --noise_rate [noise_rate]
```
where [dataset] in {UCMerced, MLRSNet, AID}, [scheme] in {BCE, LCR, SAT, ELR}, [noise_type] in {additive, subtractive, mix}, [noise_rate] in {10, 20, 30, 40}.


## Acknowledgements
Our code is heavily built upon [Instance-Dependent Multilabel Noise Generation for Multilabel Remote Sensing Image Classification](https://github.com/youngwk/IDMN).
