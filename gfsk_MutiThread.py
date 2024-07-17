import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io, img_as_float
from tkinter import Tk, filedialog
import threading
import queue

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
    # print(f"Saved: {filename}")

def fft_filter(A, D0_values, filter_type='highpass', D0_low=None, D0_high=None):
    """Function to perform frequency domain filtering on an image."""
    if A is None or D0_values is None:
        return None

    [a, b, c] = A.shape

    # Perform 2D FFT for each color channel with zero-padding
    F = fft2(A, s=(2*a, 2*b), axes=(0, 1))
    F3 = fftshift(F, axes=(0, 1))

    filtered_images = []

    for D0 in D0_values:
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

        filtered_images.append((F1, filter_type, D0, D0_low, D0_high))
        
        save_image(F1, filter_type, D0, D0_low, D0_high)

    return filtered_images


# Define queues for storing filtered images
high_pass_queue = queue.Queue()
low_pass_queue = queue.Queue()
band_pass_queue = queue.Queue()

def run_filter(A, D0_values, filter_type, D0_low=None, D0_high=None, q=None):
    filtered_images = fft_filter(A, D0_values, filter_type, D0_low, D0_high)
    q.put(filtered_images)

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

    # Create threads for high-pass, low-pass, and band-pass filters
    threads = []
    threads.append(threading.Thread(target=run_filter, args=(A, highpass_D0_values, 'highpass', None, None, high_pass_queue)))
    threads.append(threading.Thread(target=run_filter, args=(A, lowpass_D0_values, 'lowpass', None, None, low_pass_queue)))
    for D0_low, D0_high in bandpass_D0_values:
        threads.append(threading.Thread(target=run_filter, args=(A, [10], 'bandpass', D0_low, D0_high, band_pass_queue)))

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Process and plot images for each page
    for page in range(3):
        fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))

        if page == 0:
            # Plot original image and low-pass filtered images
            axs[0, 0].imshow(A.clip(0, 1))  # Clip values to [0, 1] range
            axs[0, 0].set_title('Original Image')

            idx = 1  # Start indexing from 1 for subsequent images
            while not low_pass_queue.empty() and idx < 4:
                filtered_images = low_pass_queue.get()
                for F1, filter_type, D0, D0_low, D0_high in filtered_images:
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
            while not high_pass_queue.empty() and idx < 4:
                filtered_images = high_pass_queue.get()
                for F1, filter_type, D0, D0_low, D0_high in filtered_images:
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
            while not band_pass_queue.empty() and idx < 4:
                filtered_images = band_pass_queue.get()
                for F1, filter_type, D0, D0_low, D0_high in filtered_images:
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
