import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io, img_as_float
from tkinter import Tk, filedialog


def select_image():
    """Function to select an image file using a file dialog."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
    root.destroy()
    return file_path

def load_image(image_path):
    """Function to load an image and convert grayscale to RGB if necessary."""
    try:
        A = img_as_float(io.imread(image_path))
        if A.ndim == 2:
            print("Grayscale image loaded. Converting to RGB.")
            A = np.stack((A, A, A), axis=-1)  # Convert grayscale to RGB by stacking channels
        return A
    except Exception as e:
        print(f"Error loading the image: {e}")
        return None

def fft_filter(A, D0_values, filter_type='highpass', D0_low=None, D0_high=None):
    """Function to perform frequency domain filtering on an image."""
    if A is None or D0_values is None:
        return None

    [a, b, c] = A.shape
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
    axs[0, 0].set_title('Original Image')

    # Perform 2D FFT for each color channel with zero-padding
    F = fft2(A, s=(2*a, 2*b), axes=(0, 1))
    F3 = fftshift(F, axes=(0, 1))

    for idx, D0 in enumerate(D0_values, start=1):
        W = np.zeros((2*a, 2*b, c))
        u, v = np.mgrid[-a:a, -b:b]  # Use NumPy indexing for performance
        D_square = (u)**2 + (v)**2
        if filter_type == 'highpass':
            W[:, :, :] = 1 - np.exp(-D_square[:, :, np.newaxis] / (2 * D0 **2))
        elif filter_type == 'lowpass':
            W[:, :, :] = np.exp(-D_square[:, :, np.newaxis] / (2 * D0**2))
        elif filter_type == 'bandpass':
            if D0_low is None or D0_high is None:
                print("D0_low and D0_high must be provided for bandpass filter")
                return None
            W_low = np.exp(-D_square[:, :, np.newaxis] / (2 * D0_low**2))
            W_high = np.exp(-D_square[:, :, np.newaxis] / (2 * D0_high**2))
            W[:, :, :] = W_high - W_low

        # Apply filter in the frequency domain for each channel
        G = F3 * W
        F4 = ifftshift(G, axes=(0, 1))
        F1 = ifft2(F4, axes=(0, 1))
        F1 = np.real(F1[:a, :b, :])

        # Clip filtered image values to [0, 1] range
        F1 = np.clip(F1, 0, 1)

        # Display the filtered image
        row, col = divmod(idx, 2)
        axs[row, col].imshow(F1)
        if filter_type == 'highpass':
            axs[row, col].set_title(f'High-Pass Filter D0={D0}')
            file_name = f'HighPassFiltered_D0_{D0}.png'
        elif filter_type == 'lowpass':
            axs[row, col].set_title(f'Low-Pass Filter D0={D0}')
            file_name = f'LowPassFiltered_D0_{D0}.png'
        elif filter_type == 'bandpass':
            axs[row, col].set_title(f'Band-Pass Filter D0_low={D0_low}, D0_high={D0_high}')
            file_name = f'BandPassFiltered_D0_low_{D0_low}_D0_high_{D0_high}.png'
        
        file_path = os.path.join("./", file_name)
        plt.imsave(file_path, F1)

    plt.tight_layout()
    plt.show()


def main():
    # Clear all previous plots
    plt.close('all')

    # Select and load the image
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        return

    A = load_image(image_path)
    if A is None:
        return

    # Define filter cutoff frequencies
    D0_values = [3, 10, 20]

    # Perform FFT high-pass, low-pass, and band-pass filtering and display results
    filter_types = ['highpass', 'lowpass']
    for filter_type in filter_types:
        fft_filter(A, D0_values, filter_type)

    # Bandpass filter example
    D0_low = 3
    D0_high = 10
    fft_filter(A, [10], 'bandpass', D0_low=D0_low, D0_high=D0_high)

if __name__ == "__main__":
    main()
