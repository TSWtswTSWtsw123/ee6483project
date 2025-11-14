# 📊 Sentiment Analysis - Deep Learning Models

> A comprehensive deep learning project implementing three neural network architectures for sentiment classification of product reviews. Achieves 91.2% accuracy with Attention-BiLSTM model.

[![GitHub](https://img.shields.io/badge/GitHub-ee6483project-blue?logo=github)](https://github.com/TSWtswTSWtsw123/ee6483project)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

## 🎯 Overview

This project implements comprehensive sentiment classification models:

### Deep Learning Models (Neural Networks)
- **CNN Classifier**: Multiple parallel convolutions with different filter sizes to capture n-gram features (86.3% accuracy)
- **BiLSTM Classifier**: Bidirectional LSTM processing sequences in both directions (88.7% accuracy)
- **Attention-BiLSTM Classifier** ⭐: Enhanced BiLSTM with attention mechanism for interpretability **(91.2% accuracy)**

### Traditional Machine Learning Models (Baselines)
- **Support Vector Machine (SVM)**: Linear SVM with TF-IDF features
- **Logistic Regression**: Efficient baseline with L2 regularization
- **Naive Bayes**: Probabilistic classifier for quick comparison

**Course**: IE6483 / EE6483 Mini Project - Artificial Intelligence and Data Mining
**Institution**: Nanyang Technological University (NTU)

## 📁 Project Structure

```
final/
├── Source Code/                      # Python implementation files
│   ├── deep_learning_models.py      # Deep learning model implementations
│   ├── data_utils.py                # Data loading and preprocessing
│   ├── train.py                     # Deep learning training script
│   ├── predict.py                   # Prediction script
│   ├── run_all.py                   # Complete pipeline (DL models)
│   ├── example_usage.py             # Usage examples
│   └── SVM_LR_NB.py                 # Traditional ML baselines
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

## 🎓 Model Architecture Comparison

### CNN
- Parallel Conv1D layers (filters: 3, 4, 5)
- Global max pooling
- Fast training (~45 min)
- Good for local feature extraction

### BiLSTM
- Bidirectional LSTM (2 stacked layers)
- Captures long-range dependencies
- Medium training time (~90 min)
- Better context understanding

### Attention-BiLSTM ⭐
- BiLSTM + Attention mechanism
- Interpretable predictions
- Longest training time (~100 min)
- Best performance (91.2%)
- Attention weights show influential words

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
- `deep_learning_models.py` (420 lines) - All three deep learning model implementations (CNN, BiLSTM, Attention-BiLSTM)
- `data_utils.py` (310 lines) - Data loading and preprocessing utilities
- `train.py` (380 lines) - Complete training pipeline for deep learning models
- `predict.py` (220 lines) - Prediction generation script
- `run_all.py` (95 lines) - Orchestrates train + predict for DL models
- `example_usage.py` (180 lines) - Usage examples and demonstrations
- `SVM_LR_NB.py` (251 lines) - Traditional ML baseline models (SVM, Logistic Regression, Naive Bayes)

### Configuration
- `requirements.txt` - All Python dependencies
- `training_results.json` - Validation metrics for all models

### Results
- `submission.csv` - Test predictions (2,851 samples)

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
- **Email**: 1072202885@qq.com
- **Branch**: main
- **Total Commits**: 8+
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
