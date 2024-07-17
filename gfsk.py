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
    A = img_as_float(io.imread(image_path))
    if A.ndim == 2:
        print("Grayscale image loaded. Converting to RGB.")
        A = np.stack((A, A, A), axis=-1)  # Convert grayscale to RGB by stacking channels
    return A

def fft_highpass_filter(A, D0_values):
    """Function to perform frequency domain high-pass filtering on an image."""
    [a, b, c] = A.shape
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
    axs[0, 0].set_title('Original Image')

    # Perform 2D FFT for each color channel with zero-padding
    F = fft2(A, s=(2*a, 2*b), axes=(0, 1))
    F3 = fftshift(F, axes=(0, 1))

    for idx, D0 in enumerate(D0_values, start=1):
        W = np.zeros((2*a, 2*b, c))
        for u in range(2*a):
            for v in range(2*b):
                D_square = (u-a)**2 + (v-b)**2
                W[u, v, :] = 1 - np.exp(-D_square / (2 * D0 * D0))

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
        axs[row, col].set_title(f'High-Pass Filter D0={D0}')
        file_name = f'Filtered_D0_{D0}.png'
        file_path = os.path.join("./", file_name)
        plt.imsave(file_path, F1)

    plt.tight_layout()
    plt.show()



# Function to apply Gaussian low-pass filter
def fft_lowpass_filter(A, D0_values):
    """Function to perform frequency domain low-pass filtering on an image."""
    [a, b, c] = A.shape
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
    axs[0, 0].set_title('Original Image')

    # Perform 2D FFT for each color channel with zero-padding
    F = fft2(A, s=(2*a, 2*b), axes=(0, 1))
    F3 = fftshift(F, axes=(0, 1))

    for idx, D0 in enumerate(D0_values, start=1):
        W = np.zeros((2*a, 2*b, c))
        for u in range(2*a):
            for v in range(2*b):
                D_square = (u-a)**2 + (v-b)**2
                W[u, v, :] = np.exp(-D_square / (2 * D0**2))

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
        axs[row, col].set_title(f'Low-Pass Filter D0={D0}')
        file_name = f'Filtered_D0_{D0}.png'
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

    # Define high-pass filter cutoff frequencies
    D0_values = [3, 10, 20]

    # Perform FFT high-pass filtering and display results
    
    fft_highpass_filter(A, D0_values)
    fft_lowpass_filter(A, D0_values)

if __name__ == "__main__":
    main()
