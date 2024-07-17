import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from tkinter import Tk, filedialog
import torch

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

def save_image(image, filter_type, D0, D0_low, D0_high):
    """Function to save an image to the ./result_imgs directory with appropriate naming."""
    if not os.path.exists('./result_imgs'):
        os.makedirs('./result_imgs')

    if filter_type == 'bandpass':
        filename = f"./result_imgs/{filter_type}_D0_low_{D0_low}_D0_high_{D0_high}.png"
    else:
        filename = f"./result_imgs/{filter_type}_D0_{D0}.png"

    plt.imsave(filename, image)

def fft_filter(A, D0_values, filter_type='highpass', D0_low=None, D0_high=None):
    """Function to perform frequency domain filtering on an image using PyTorch."""
    print("Starting FFT filter...")
    if A is None or D0_values is None:
        return None

    [a, b, c] = A.shape

    # Try to use CUDA if available and sufficient memory, otherwise fall back to CPU
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    except RuntimeError as e:
        print("CUDA memory insufficient, falling back to CPU.")
        device = torch.device('cpu')
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    print(f"Using device: {device}")

    # Perform 2D FFT for each color channel with zero-padding using PyTorch
    F = torch.fft.fftn(A_tensor, s=(2*a, 2*b))
    F_shifted = torch.fft.fftshift(F, dim=(0, 1))

    filtered_images = []

    for D0 in D0_values:
        W = torch.zeros((2*a, 2*b, c), dtype=torch.float32, device=device)
        u, v = torch.meshgrid(torch.arange(-a, a, device=device), torch.arange(-b, b, device=device))
        D_square = u**2 + v**2
        if filter_type == 'highpass':
            W[:, :, :] = 1 - torch.exp(-D_square[:, :, None] / (2 * D0**2))
        elif filter_type == 'lowpass':
            W[:, :, :] = torch.exp(-D_square[:, :, None] / (2 * D0**2))
        elif filter_type == 'bandpass':
            if D0_low is None or D0_high is None:
                print("D0_low and D0_high must be provided for bandpass filter")
                return None
            W_low = torch.exp(-D_square[:, :, None] / (2 * D0_low**2))
            W_high = torch.exp(-D_square[:, :, None] / (2 * D0_high**2))
            W[:, :, :] = W_high - W_low

        # Apply filter in the frequency domain for each channel
        G = F_shifted * W
        G_ifftshift = torch.fft.ifftshift(G, dim=(0, 1))
        F1 = torch.fft.ifftn(G_ifftshift, dim=(0, 1))
        F1 = torch.real(F1[:a, :b, :])

        # Clip filtered image values to [0, 1] range
        F1 = torch.clamp(F1, 0, 1).cpu().numpy()

        filtered_images.append((F1, filter_type, D0, D0_low, D0_high))
        
        save_image(F1, filter_type, D0, D0_low, D0_high)

    return filtered_images

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
    highpass_D0_values = [5, 10, 20]
    lowpass_D0_values = [20, 10, 5]
    bandpass_D0_values = [(5, 10), (10, 30), (5, 30)]

    # Perform high-pass filtering
    high_pass_filtered_images = fft_filter(A, highpass_D0_values, 'highpass')

    # Perform low-pass filtering
    low_pass_filtered_images = fft_filter(A, lowpass_D0_values, 'lowpass')

    # Perform band-pass filtering
    band_pass_filtered_images = []
    for D0_low, D0_high in bandpass_D0_values:
        band_pass_filtered_images.extend(fft_filter(A, [10], 'bandpass', D0_low, D0_high))

    # Process and plot images for each page
    for page in range(3):
        fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))

        if page == 0:
            # Plot original image and low-pass filtered images
            axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
            axs[0, 0].set_title('Original Image')

            idx = 1  # Start indexing from 1 for subsequent images
            for F1, filter_type, D0, D0_low, D0_high in low_pass_filtered_images:
                row, col = divmod(idx, 2)
                if row < 2 and col < 2:
                    axs[row, col].imshow(F1)
                    axs[row, col].set_title(f'Low-Pass Filter D0={D0}')
                    idx += 1

        elif page == 1:
            # Plot original image and high-pass filtered images
            axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
            axs[0, 0].set_title('Original Image')

            idx = 1  # Start indexing from 1 for subsequent images
            for F1, filter_type, D0, D0_low, D0_high in high_pass_filtered_images:
                row, col = divmod(idx, 2)
                if row < 2 and col < 2:
                    axs[row, col].imshow(F1)
                    axs[row, col].set_title(f'High-Pass Filter D0={D0}')
                    idx += 1

        elif page == 2:
            # Plot original image and band-pass filtered images
            axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
            axs[0, 0].set_title('Original Image')

            idx = 1  # Start indexing from 1 for subsequent images
            for F1, filter_type, D0, D0_low, D0_high in band_pass_filtered_images:
                row, col = divmod(idx, 2)
                if row < 2 and col < 2:
                    axs[row, col].imshow(F1)
                    axs[row, col].set_title(f'Band-Pass Filter D0_low={D0_low}, D0_high={D0_high}')
                    idx += 1

        plt.tight_layout()
        plt.show(block=False)  # Show the current page without blocking further execution

    # Wait for user input before closing the figures
    input("Press Enter to close all figures...")

    plt.close('all')  # Close all figures

if __name__ == "__main__":
    main()
