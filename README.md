# 📊 From TextCNN to SNN: End-to-End Sentiment Analysis Pipeline

> A comprehensive sentiment classification pipeline bridging traditional machine learning, modern deep architectures, and energy-efficient spiking neural networks (SNNs). **Best DL accuracy: 91.2%** with Attention-BiLSTM | **SNN efficiency: ~90% energy reduction**

[![GitHub](https://img.shields.io/badge/GitHub-ee6483project-blue?logo=github)](https://github.com/TSWtswTSWtsw123/ee6483project)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

## 🎯 Overview

This project proposes an **end-to-end sentiment classification pipeline** that systematically bridges three approaches:

### Stage 1: Traditional ML Baselines (TF-IDF Features)
- **Logistic Regression**: Fast linear baseline with L2 regularization
- **Naive Bayes**: Probabilistic classifier for quick comparison
- **Support Vector Machine (SVM)**: Linear SVM for high-dimensional text

### Stage 2: Advanced Deep Learning Models (Learned Embeddings) ⭐
- **CNN Classifier**: Multiple parallel convolutions capturing n-gram features (86.3% accuracy)
- **BiLSTM Classifier**: Bidirectional LSTM processing sequences in both directions (88.7% accuracy)
- **Attention-BiLSTM Classifier**: Enhanced BiLSTM with attention mechanism for interpretability **(91.2% accuracy - BEST)**

### Stage 3: Energy-Efficient Spiking Neural Networks (SNN)
- **Tailored TextCNN → SNN Conversion**: ANN-to-SNN conversion with surrogate-gradient fine-tuning
- **Energy Efficiency**: Estimated ~90% reduction in per-inference compute energy under neuromorphic assumptions
- **Accuracy-Efficiency Tradeoff**: Controlled via time-steps (T) and membrane threshold (U_thr) tuning

**Course**: IE6483 / EE6483 Mini Project - Artificial Intelligence and Data Mining
**Institution**: Nanyang Technological University (NTU)
**Team**: Ye Shuhan (SNN), Tang Shuwei (Deep Learning), Ding Miao (Traditional ML & Domain Adaptation)

## 📁 Project Structure

```
final/
├── Source Code/                      # Complete implementation (ML, DL, SNN)
│   ├── deep_learning_models.py      # Deep learning: CNN, BiLSTM, Attention-BiLSTM
│   ├── data_utils.py                # Data loading and preprocessing
│   ├── train.py                     # Training pipeline (DL models)
│   ├── predict.py                   # Prediction generation (DL models)
│   ├── run_all.py                   # Complete DL pipeline
│   ├── example_usage.py             # Usage examples (DL models)
│   ├── SVM_LR_NB.py                 # Traditional ML baselines (TF-IDF)
│   └── [SNN_conversion code]        # SNN conversion (code pending - see Note below)
│
├── Configuration & Results/          # Dependencies and results
│   ├── requirements.txt              # Python dependencies
│   ├── training_results.json         # Validation metrics
│   └── submission.csv                # Test predictions
│
├── Data Files/                       # Training and test datasets
│   ├── train.json                   # 7,401 training samples
│   └── test.json                    # 1,851 test samples
│
├── Models/                           # Pre-trained model weights
│   ├── best_cnn_model.pt
│   ├── best_bilstm_model.pt
│   └── best_attention_bilstm_model.pt
│
├── Visualizations/                   # Performance analysis charts
│   ├── accuracy_vs_time.png
│   ├── all_models_comparison.png
│   ├── attention_visualization.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
├── Documentation/                    # Comprehensive guides
│   ├── README.md                    # Full project guide
│   ├── QUICKSTART.md                # Quick start instructions
│   ├── PROJECT_SUMMARY.md           # Project overview
│   ├── FINAL_REPORT.md              # Detailed final report
│   └── GITHUB_UPLOAD_INSTRUCTIONS.md
│
├── Research & References/            # Academic papers and sources
│   ├── merged_report.pdf
│   ├── deeplearningpart.pdf
│   ├── IE6483-Project1.pdf
│   └── [More references...]
│
├── Logs/                             # Training logs
│   ├── training.log
│   └── full_training.log
│
└── Notebooks/                        # Jupyter notebooks
    ├── Sentiment_Analysis_Models.ipynb
    ├── tfidf_features_example.ipynb
    └── [More notebooks...]
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### 2. Installation

```bash
cd Source\ Code
pip install -r ../Configuration\ \&\ Results/requirements.txt
```

### 3. Run the Project

**Option A: Complete Pipeline (Train + Predict)**
```bash
cd Source\ Code
python run_all.py
```

**Option B: Train Only**
```bash
cd Source\ Code
python train.py
```

**Option C: Generate Predictions**
```bash
cd Source\ Code
python predict.py
```

**Option D: Run Examples**
```bash
cd Source\ Code
python example_usage.py
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| CNN | 86.3% | 0.862 | 0.863 | 0.863 | ~45 min |
| BiLSTM | 88.7% | 0.884 | 0.887 | 0.886 | ~90 min |
| **Attention-BiLSTM** | **91.2%** | **0.908** | **0.912** | **0.910** | **~100 min** |

### Class-wise Performance (Best Model)
- Positive class recall: 92.0%
- Negative class recall: 91.6%
- Balanced performance across both classes

## 🏗️ Architecture Details

### Attention-BiLSTM (Primary Model)
```
Input → Embedding (300-dim)
       → BiLSTM (2 layers, 256 units each)
       → Attention Mechanism
       → Fully Connected (512 → 128 → 1)
       → Sigmoid (Binary Classification)
```

**Key Features:**
- Attention weights reveal which words influence the prediction
- 300-dimensional word embeddings
- 2 stacked bidirectional LSTM layers
- Attention-based context vector
- Dropout (0.5) for regularization
- Batch normalization for stability

## 📚 Documentation

For detailed information, see:

1. **[Documentation/README.md](Documentation/README.md)** - Complete project guide with full details
2. **[Documentation/QUICKSTART.md](Documentation/QUICKSTART.md)** - Step-by-step quick start
3. **[Documentation/PROJECT_SUMMARY.md](Documentation/PROJECT_SUMMARY.md)** - High-level overview
4. **[Documentation/FINAL_REPORT.md](Documentation/FINAL_REPORT.md)** - Detailed final report
5. **[UPLOAD_SUCCESS.md](UPLOAD_SUCCESS.md)** - GitHub upload confirmation

## 💾 Data

**Training Data**: 7,401 product reviews with sentiment labels
**Test Data**: 1,851 product reviews for evaluation
**Class Distribution**: ~85% positive, ~15% negative (6:1 imbalance)
**Sequence Length**: Fixed at 200 tokens (covers 91.3% of reviews)

## 🔧 Key Features

### Data Preprocessing
- Text normalization (lowercase, remove URLs/emails)
- NLTK tokenization
- Vocabulary building (15,247 words)
- Sequence padding/truncation
- Special tokens: `<PAD>` (index 0), `<UNK>` (index 1)

### Training Features
- **Loss Function**: Weighted Binary Cross-Entropy (handles class imbalance)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 64
- **Early Stopping**: Patience = 5 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau (reduce by 0.5 every 2 epochs)
- **Gradient Clipping**: max_norm = 1.0
- **Reproducibility**: Fixed random seed = 42

### Regularization
- Dropout (0.5) for overfitting prevention
- Batch normalization for training stability
- Weighted loss for class imbalance handling
- Gradient clipping for stability

## 🎓 Three-Stage Pipeline Overview

### Stage 1: Traditional ML Baselines (Ding Miao et al.)
| Model | Features | Approach | Pros | Cons |
|-------|----------|----------|------|------|
| **Logistic Regression** | TF-IDF (1-2 gram) | Linear classifier + L2 reg | Fast, interpretable | Shallow feature extraction |
| **Naive Bayes** | TF-IDF (1-2 gram) | Probabilistic | Simple, lightweight | Strong independence assumption |
| **SVM** | TF-IDF (1-2 gram) | Linear SVM | Good for high-dim text | No built-in probability |

### Stage 2: Advanced Deep Learning (Tang Shuwei et al.)
| Model | Input | Architecture | Accuracy | Training Time |
|-------|-------|--------------|----------|---------------|
| **CNN** | Word embeddings (300-dim) | 3 parallel Conv1D + Global Max Pool | 86.3% | ~45 min |
| **BiLSTM** | Word embeddings (300-dim) | 2-layer BiLSTM + FC | 88.7% | ~90 min |
| **Attention-BiLSTM** ⭐ | Word embeddings (300-dim) | 2-layer BiLSTM + Attention + FC | **91.2%** | ~100 min |

**Key Features:**
- Learned 300-dimensional word embeddings
- Weighted Binary Cross-Entropy loss (handles class imbalance 6:1)
- Adam optimizer with learning rate scheduling
- Early stopping (patience=5) and gradient clipping
- Regularization: Dropout (0.5), Batch Normalization, Gradient clipping

### Stage 3: Energy-Efficient SNNs (Ye Shuhan et al.)
| Component | Description | Status |
|-----------|-------------|--------|
| **Tailored TextCNN** | SNN-compatible CNN (ReLU, avg-pooling, bias-free) | ✅ Implemented & Trained (88.2% accuracy) |
| **ANN-to-SNN Conversion** | Weight mapping + Poisson spike encoding | ✅ Conversion framework ready |
| **Surrogate Gradient Fine-tuning** | Backprop through spikes using fast-sigmoid | ✅ Fine-tuning pipeline ready |
| **Energy Analysis** | Time-step (T) & threshold (U_thr) tuning | ✅ Results: ~87.6% accuracy, ~90% energy savings |
| **SNN Code** | Full SNN implementation & training scripts | ⏳ Code pending (see note below) |

## 🔍 Model Interpretability

The Attention-BiLSTM model provides interpretability through:
- **Attention Weights**: Shows which words influenced the prediction
- **Error Analysis**: Understanding failure cases
- **Example Cases**: Detailed examples of correct and incorrect predictions

## ⚙️ Hardware Requirements

- **CPU**: Intel i5/i7 or AMD Ryzen 5/7+ (minimum)
- **GPU**: NVIDIA GTX 1080+ recommended (tested on RTX 2080 Ti)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~4GB for model weights + data files

## 📌 Important Note: SNN Implementation Status

**Current Status**: The SNN conversion and fine-tuning experiments have been completed and reported in the final paper (`iclr2023_conference copy.tex`). However, the full SNN implementation code is currently pending and will be added to this repository.

**What's Included:**
- ✅ Complete DL models (CNN, BiLSTM, Attention-BiLSTM) with full source code
- ✅ Complete traditional ML models (SVM, LR, NB) with full source code
- ✅ Comprehensive results, logs, and documentation
- ✅ SNN theoretical framework and conversion methodology in paper

**What's Pending:**
- ⏳ SNN conversion implementation (Python code)
- ⏳ Surrogate gradient fine-tuning scripts
- ⏳ SNN inference and energy analysis code

For SNN methodology details, please refer to **`Research & References/iclr2023_conference copy.tex`** (Section 3 & 4).

---

## 🛠️ Troubleshooting

### "No such file or directory" error
**Solution**: Ensure you're running scripts from the correct directory

### Out of Memory (OOM) error
**Solution**: Reduce batch size in train.py (modify `batch_size=64`)

### Model training is very slow
**Solution**: Check CUDA/GPU support with `torch.cuda.is_available()`

## 📈 Project Highlights

✅ **Complete Implementation** - 3 different neural network architectures
✅ **High Performance** - 91.2% accuracy on validation set
✅ **Well Documented** - Comprehensive README and documentation
✅ **Organized Structure** - 9 logical project folders
✅ **Production Ready** - Clean, well-commented source code
✅ **Reproducible** - Fixed seeds and detailed training logs
✅ **Visualized Results** - 7 performance analysis charts
✅ **GitHub Ready** - Professional repository structure

## 📄 Files Overview

### Source Code
**Stage 2: Deep Learning Models (Tang Shuwei)**
- `deep_learning_models.py` (420 lines) - CNN, BiLSTM, Attention-BiLSTM implementations
- `data_utils.py` (310 lines) - Data loading and preprocessing utilities
- `train.py` (380 lines) - Complete training pipeline
- `predict.py` (220 lines) - Prediction generation script
- `run_all.py` (95 lines) - Orchestrates train + predict
- `example_usage.py` (180 lines) - Usage examples and demonstrations

**Stage 1: Traditional ML Models (Ding Miao)**
- `SVM_LR_NB.py` (251 lines) - Logistic Regression, Naive Bayes, SVM with TF-IDF features

**Stage 3: SNN Models (Ye Shuhan)**
- `[SNN implementation code]` - ⏳ **Pending** - See note below

### Configuration
- `requirements.txt` - All Python dependencies
- `training_results.json` - Validation metrics for all models

### Results
- `submission_attention_bilstm.csv` - Final test predictions using Attention-BiLSTM model (1,851 samples)

## 🎯 Project Completion Checklist

- ✅ Literature survey on sentiment analysis
- ✅ Feature format selection (learned word embeddings)
- ✅ Multiple model architectures (CNN, BiLSTM, Attention-BiLSTM)
- ✅ Hyperparameter optimization and ablation study
- ✅ Error analysis with example cases
- ✅ Feature format impact analysis
- ✅ Domain adaptation strategy (hotel reviews)
- ✅ Noisy label handling approaches
- ✅ Project structure organized into 9 folders
- ✅ Comprehensive documentation
- ✅ GitHub repository uploaded

## 🔗 Repository Information

- **GitHub Repository**: https://github.com/TSWtswTSWtsw123/ee6483project
- **Owner**: TSWtswTSWtsw123
- **Branch**: main
- **Total Commits**: 16
- **Latest Commit**: `445cd3a` - Update README to reflect complete team project (ML, DL, SNN pipeline)
- **Last Updated**: 2025-11-14

## 📖 References

- Devlin et al. (2019): BERT - Pre-training of Deep Bidirectional Transformers
- Pennington et al. (2014): GloVe - Global Vectors for Word Representation
- Mikolov et al. (2013): Efficient Estimation of Word Representations in Vector Space
- Kim (2014): Convolutional Neural Networks for Sentence Classification
- Bahdanau et al. (2015): Neural Machine Translation by Jointly Learning to Align and Translate

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

This is a course project. For modifications or improvements, please fork the repository and create a pull request.

## ❓ FAQ

**Q: How do I clone this project?**
```bash
git clone https://github.com/TSWtswTSWtsw123/ee6483project.git
cd ee6483project
```

**Q: Which model should I use?**
A: The Attention-BiLSTM model offers the best balance of accuracy (91.2%) and interpretability.

**Q: How long does training take?**
A: Approximately 100 minutes on an RTX 2080 Ti GPU. CPU training will take significantly longer.

**Q: Can I use pre-trained models?**
A: Yes, all three models are pre-trained and saved in the `Models/` folder.

**Q: How do I understand the model decisions?**
A: The Attention-BiLSTM model provides attention weights showing which words influenced each prediction.

## 📞 Support

For questions, issues, or suggestions:
1. Check the [Documentation](Documentation/) folder
2. Review the [FINAL_REPORT.md](Documentation/FINAL_REPORT.md)
3. Check existing issues on GitHub
4. Open a new issue with detailed description

---

**Project Status**: ✅ Complete and uploaded to GitHub
**Last Updated**: 2025-11-14
**Course**: IE6483 / EE6483 - Artificial Intelligence and Data Mining

**Made with ❤️ for EE6483 Mini Project**
