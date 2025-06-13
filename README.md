### Source Code for the Paper **SMAE-Fusion**

This repository contains the implementation of our paper **SMAE-Fusion:** Integrating saliency-aware masked autoencoder with hybrid attention transformer for infrared–visible image fusion , which has been published in **[Information Fusion]**.

You can access the paper at the following link:  
[**SMAE-Fusion**](https://doi.org/10.1016/j.inffus.2024.102841)

---

## ⚙️ Installation

This project was tested under the following environment:

- Python 3.10  
- PyTorch ≥ 2.0 with CUDA 12.x  
- Other dependencies will be automatically installed via `requirements.txt`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 1. Pre-training Phase

Download the required dataset and start pre-training with:

```bash
sh bash/train.sh
```

> The training configuration is specified in:  
> `option/train/MSRS/train_smaepre.yaml`

---

### 🔹 2. Fine-tuning Phase

After pre-training, fine-tune the model by running:

```bash
sh bash/train.sh
```

> The fine-tuning configuration is specified in:  
> `option/train/MSRS/train_smaeft.yaml`

---

### 🔹 3. Testing Phase

Evaluate the trained or fine-tuned model using:

```bash
sh bash/test.sh
```

> The test configuration is specified in:  
> `option/test/test_smaeft.yaml`

---

## 📊 Evaluation

For quantitative evaluation of fusion quality, please refer to the official evaluation toolkit:  
🔗 **[VIF-Benchmark](https://github.com/Linfeng-Tang/VIF-Benchmark)**

---

## 🙏 Acknowledgements

This project builds upon and is inspired by the following excellent works:

- 🔗 [DecompositionForFusion](https://github.com/erfect2020/DecompositionForFusion)  
- 🔗 [BasicSR](https://github.com/XPixelGroup/BasicSR)  
- 🔗 [SS-MAE](https://github.com/summitgao/SS-MAE)

We sincerely thank the authors for their contributions to the community.

---

## 📖 Citation

If you use this work in your research, please cite it as follows:

```bibtex
@article{wang2025smae,
  title={SMAE-Fusion: Integrating saliency-aware masked autoencoder with hybrid attention transformer for infrared--visible image fusion},
  author={Wang, Qinghua and Li, Ziwei and Zhang, Shuqi and Luo, Yuhong and Chen, Wentao and Wang, Tianyun and Chi, Nan and Dai, Qionghai},
  journal={Information Fusion},
  volume={117},
  pages={102841},
  year={2025},
  publisher={Elsevier}
}
```

---

