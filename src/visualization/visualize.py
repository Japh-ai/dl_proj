import matplotlib.pyplot as plt


def plot_waveform_with_prediction(waveform, pred_prob, index):
    """
    Plot waveform data along with prediction probability.

    Parameters:
        waveform (np.ndarray): 3-channel waveform data.
        pred_prob (float): Prediction probability.
        index (int): Index of the waveform.
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size

    channels = waveform.shape[0]  # Assuming waveform is (channels, samples)

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(waveform[i], label=f"Channel {i+1}")
        plt.axhline(y=0, color="k", linestyle="--")
        plt.title(f"Channel {i+1}", fontsize=10)
        plt.xlabel("Sample Index", fontsize=8)
        plt.ylabel("Amplitude", fontsize=8)

    plt.suptitle(
        f"Waveform with Prediction - Index {index}\nPrediction Probability: {pred_prob:.4f}",
        fontsize=14,
    )
    plt.tight_layout(pad=2.0)  # Adds padding between plots for better readability
    plt.show()
