import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

SEED = 42
np.random.seed(SEED)


def load_data(train_path, test_path):
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)
    train_df["reviews"] = train_df["reviews"].astype(str).fillna("")
    train_df["sentiments"] = train_df["sentiments"].astype(int)
    test_df["reviews"] = test_df["reviews"].astype(str).fillna("")
    return train_df, test_df


def get_models():
    models = {}
    
    # Logistic Regression
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(norm="l2")),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", 
                                   max_iter=200, random_state=SEED))
    ])
    lr_params = {
        "tfidf__ngram_range": [(1,2)],
        "tfidf__min_df": [2],
        "clf__C": [1.0, 2.0]
    }
    models["Logistic Regression"] = (lr_pipe, lr_params)
    
    # Naive Bayes
    nb_pipe = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB())
    ])
    nb_params = {
        "vect__ngram_range": [(1,2)],
        "vect__min_df": [2],
        "clf__alpha": [0.5, 1.0]
    }
    models["Naive Bayes"] = (nb_pipe, nb_params)
    
    # Linear SVM
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LinearSVC(random_state=SEED, max_iter=1000))
    ])
    svm_params = {
        "tfidf__ngram_range": [(1,2)],
        "tfidf__min_df": [2],
        "clf__C": [1.0, 2.0]
    }
    models["Linear SVM"] = (svm_pipe, svm_params)
    
    return models


def train_model(name, pipe, params, X_train, y_train, X_val, y_val):
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    start = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, params, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"Best params: {gs.best_params_}")
    print(f"CV F1: {gs.best_score_:.4f}")
    
    y_pred = gs.predict(X_val)
    
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, average='macro'),
        "Recall": recall_score(y_val, y_pred, average='macro'),
        "F1-Score": f1_score(y_val, y_pred, average='macro'),
        "Time_min": train_time / 60
    }
    
    print(f"Val Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Val F1: {metrics['F1-Score']:.4f}")
    print(f"Time: {metrics['Time_min']:.2f} min")
    
    return metrics, gs.best_estimator_


def plot_results(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Performance metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, df[metric], width, label=metric, color=color, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: Traditional ML Methods', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    fig.tight_layout(pad=1.5)
    fig.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/performance_comparison.png")
    plt.close(fig)
    
    # Plot 2: Training time
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_time = sns.color_palette("viridis", len(df))
    bars = ax.barh(df['Model'], df['Time_min'], color=colors_time, alpha=0.8)
    
    for bar in bars:
        width_bar = bar.get_width()
        ax.text(width_bar + 0.05, bar.get_y() + bar.get_height()/2,
               f'{width_bar:.2f}', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.tight_layout(pad=1.5)
    fig.savefig(output_dir / 'training_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_time.png")
    plt.close(fig)
    
    # Plot 3: Accuracy vs Time
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['Time_min'], df['Accuracy'], 
                        s=300, alpha=0.6, c=range(len(df)), cmap='viridis', 
                        edgecolors='black', linewidth=2)
    
    for _, row in df.iterrows():
        ax.annotate(row['Model'], 
                   (row['Time_min'], row['Accuracy']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. Training Time Trade-off', fontsize=13)
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout(pad=1.5)
    fig.savefig(output_dir / 'accuracy_vs_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_vs_time.png")
    plt.close(fig)


def main(args):
    print("="*80)
    print("TRADITIONAL ML BASELINE COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data(args.train, args.test)
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Split
    X = train_df["reviews"].values
    y = train_df["sentiments"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Get models
    models = get_models()
    
    # Train all models
    results = []
    for name, (pipe, params) in models.items():
        metrics, best_model = train_model(name, pipe, params, X_train, y_train, X_val, y_val)
        results.append(metrics)
        
        # Generate submission
        test_pred = best_model.predict(test_df["reviews"].values)
        sub = pd.DataFrame({"id": np.arange(len(test_pred)), "sentiments": test_pred})
        output_name = name.lower().replace(" ", "_")
        sub.to_csv(f"submission_{output_name}.csv", index=False)
        print(f"Saved: submission_{output_name}.csv")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("F1-Score", ascending=False)
    
    # Save results CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'results.csv', index=False)
    print(f"\nSaved: {output_dir}/results.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n{df.to_string(index=False)}\n")
    
    best = df.iloc[0]
    print(f"Best Model: {best['Model']} - {best['Accuracy']:.1%} accuracy")
    print(f"Fastest: {df.loc[df['Time_min'].idxmin(), 'Model']} - {df['Time_min'].min():.2f} min")
    print(f"\nAll results saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traditional ML Baseline Comparison")
    parser.add_argument("--train", type=str, default="train.json")
    parser.add_argument("--test", type=str, default="test.json")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
