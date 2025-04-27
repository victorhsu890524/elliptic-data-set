from sklearn.model_selection import BaseCrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)

class TimeStepSplit(BaseCrossValidator):
    def __init__(self, df, n_splits=4):
        self.df = df
        self.n_splits = n_splits
        self.time_steps = sorted(df['timeStep'].unique())

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        n_time_steps = len(self.time_steps)
        fold_size = n_time_steps // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            train_steps = self.time_steps[:train_end]
            test_steps = self.time_steps[test_start:test_end]

            train_idx = self.df[self.df['timeStep'].isin(train_steps)].index.to_numpy()
            test_idx = self.df[self.df['timeStep'].isin(test_steps)].index.to_numpy()

            yield train_idx, test_idx

def plot_classification_results(y_true, y_prob, y_pred, model_name="Model"):
    """
    y_true: ground truth labels
    y_prob: predicted probabilities for class 1
    y_pred: predicted labels (0 or 1)
    model_name: string to use in plot titles (default: 'Model')
    """

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Classification Metrics: {model_name}", fontsize=16)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['licit', 'illicit'],
                yticklabels=['licit', 'illicit'],
                ax=axs[0, 0])
    axs[0, 0].set_xlabel('Predicted')
    axs[0, 0].set_ylabel('Actual')
    axs[0, 0].set_title('Confusion Matrix')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    axs[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    axs[0, 1].plot([0, 1], [0, 1], '--', label='Random')
    axs[0, 1].set_xlabel('False Positive Rate')
    axs[0, 1].set_ylabel('True Positive Rate')
    axs[0, 1].set_title('ROC Curve')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    axs[1, 0].plot(recall, precision, lw=2, label=f'PR Curve (AP = {ap:.3f})')
    axs[1, 0].set_xlabel('Recall')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].set_title('Precision-Recall Curve')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Precision vs Recall vs Threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    axs[1, 1].plot(thresholds, precision[:-1], label='Precision')
    axs[1, 1].plot(thresholds, recall[:-1], label='Recall')
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].set_title('Precision vs. Recall vs. Threshold')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print classification report and ROC AUC
    print(classification_report(y_true, y_pred, target_names=["licit", "illicit"]))
    print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.6f}")