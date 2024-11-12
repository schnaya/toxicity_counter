import os
import re

from matplotlib import pyplot as plt

from toxic_coms_task.final_module import nlp, stop_words


def preprocess_text( text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s\u00C0-\u017F\u4e00-\u9fff\u0400-\u04FF]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if
              token.lemma_ not in stop_words and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def plot_loss_curve( log_path):
    if not os.path.exists(log_path):
        raise ValueError("Log file path is not provided or does not exist.")
    try:
        # Read the log file and extract loss values
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()

        loss_values = [float(line.split()[1]) for line in lines if "iter:" in line]
        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Failed to plot loss curve: {str(e)}")

