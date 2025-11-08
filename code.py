import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os

# Load model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(current_dir, '..', '..', 'models', 'colorization_deploy_v2.prototxt')
model_path = os.path.join(current_dir, '..', '..', 'models', 'colorization_release_v2.caffemodel')
kernel_path = os.path.join(current_dir, '..', '..', 'models', 'pts_in_hull.npy')

# UI preview size (max width/height for each preview). Increase these to show
# larger images by default. Each preview will be centered on a neutral bg.
DISPLAY_MAX_W = 640
DISPLAY_MAX_H = 640

# Check files exist
if not all(os.path.exists(f) for f in [prototxt_path, model_path, kernel_path]):
    print("Error: Missing model files. Ensure they are in the current directory.")
    exit()

# Load network and kernel
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path):
    # Load and preprocess
    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0] - 50

    # Predict ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

    # Combine L and ab
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    return bw_image, colorized

def _resize_preserve_aspect(pil_img, max_w, max_h):
    """Resize a PIL image preserving aspect ratio so it fits within max_w x max_h."""
    w, h = pil_img.size
    if w == 0 or h == 0:
        return pil_img
    ratio = min(max_w / w, max_h / h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return pil_img.resize(new_size, Image.Resampling.LANCZOS)

def _make_centered_preview(pil_img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H, bg_color=(240,240,240)):
    """Return a new Image of size (max_w,max_h) with pil_img resized and centered on a bg."""
    img_resized = _resize_preserve_aspect(pil_img, max_w, max_h)
    bg = Image.new("RGB", (max_w, max_h), bg_color)
    x = (max_w - img_resized.width) // 2
    y = (max_h - img_resized.height) // 2
    bg.paste(img_resized, (x, y))
    return bg

def open_image():
    path = filedialog.askopenfilename(initialdir=current_dir,
                                      title="Select an image",
                                      filetypes=(
                                          ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                                          ("All files", "*.*"),
                                      ))
    if path:
        bw_img, color_img = colorize_image(path)

        # Convert BGR to RGB for display
        bw_rgb = cv2.cvtColor(bw_img, cv2.COLOR_BGR2RGB)
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        bw_pil = Image.fromarray(bw_rgb)
        color_pil = Image.fromarray(color_rgb)

        # Create fixed-size previews (centered on a neutral background)
        bw_preview = _make_centered_preview(bw_pil, DISPLAY_MAX_W, DISPLAY_MAX_H)
        color_preview = _make_centered_preview(color_pil, DISPLAY_MAX_W, DISPLAY_MAX_H)

        bw_tk = ImageTk.PhotoImage(bw_preview)
        color_tk = ImageTk.PhotoImage(color_preview)

        # Update labels
        label_bw.config(image=bw_tk)
        label_color.config(image=color_tk)
        # Keep references to avoid garbage collection
        label_bw.image = bw_tk
        label_color.image = color_tk

# GUI setup
root = tk.Tk()
root.title("Image Colorizer")

# Prefer a larger default window so previews are visible. User can resize further.
root.geometry("1400x900")
root.minsize(1000, 700)

# Configure grid to expand
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=1)

btn = Button(root, text="Select Image", command=open_image, font=("Arial", 12), width=20)
btn.grid(row=0, column=0, columnspan=2, pady=12)

label_bw = Label(root, bd=2, relief="sunken")
label_bw.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
label_color = Label(root, bd=2, relief="sunken")
label_color.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

root.mainloop()