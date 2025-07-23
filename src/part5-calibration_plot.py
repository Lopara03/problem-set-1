'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

#My code:

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

# === Read in test data ===
df_lr = pd.read_csv("data/df_arrests_test_with_lr.csv")
df_dt = pd.read_csv("data/df_arrests_test_with_dt.csv")

# calibrate
print("Logistic Regression Calibration Plot:")
calibration_plot(df_dt['y'], df_dt['pred_lr'], n_bins=5)

print("Decision Tree Calibration Plot:")
calibration_plot(df_dt['y'], df_dt['pred_dt'], n_bins=5)

# compare calibration 
print("Which model is more calibrated?")
print("Answer: Based on visual inspection of the calibration plots, the model whose curve is closer to the 45-degree dashed line is more calibrated.")

# extra credit 

# PPV = Precision = TP / (TP + FP)
def compute_ppv(y_true, y_prob, top_n=50):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    top_preds = df.sort_values('y_prob', ascending=False).head(top_n)
    ppv = top_preds['y_true'].sum() / top_n
    return ppv

# logistic regression
ppv_lr = compute_ppv(df_lr['y'], df_lr['pred_lr_proba'], top_n=50)
auc_lr = roc_auc_score(df_lr['y'], df_lr['pred_lr_proba'])

# decision tree
ppv_dt = compute_ppv(df_dt['y'], df_dt['pred_dt'], top_n=50)
auc_dt = roc_auc_score(df_dt['y'], df_dt['pred_dt'])

print(f"PPV for Logistic Regression (Top 50): {ppv_lr:.3f}")
print(f"PPV for Decision Tree (Top 50): {ppv_dt:.3f}")
print(f"AUC for Logistic Regression: {auc_lr:.3f}")
print(f"AUC for Decision Tree: {auc_dt:.3f}")

# agreement on better model
print("Do both metrics agree that one model is more accurate than the other?")
if (ppv_lr > ppv_dt) and (auc_lr >
                           auc_dt):
    print("Yes, Logistic Regression performs better in both PPV and AUC.")
elif (ppv_lr < ppv_dt) and (auc_lr < auc_dt):
    print("Yes, Decision Tree performs better in both PPV and AUC.")
else:
    print("No, the metrics disagree — one model has better PPV, the other better AUC.")
