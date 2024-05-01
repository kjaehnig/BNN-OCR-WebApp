import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('/Users/karljaehnig/Downloads/Drawing.sketchpad2.png')

# Check if image is loaded properly
if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)

# Apply GaussianBlur
blur = cv2.GaussianBlur(gray, (5,5), 0, 0)
# print(help(cv2.GaussianBlur))

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare to crop the contours
digit_images = []

# Sort contours by their left most x-coordinate, left to right
contours = sorted(
    contours, 
    key=lambda ctr: (cv2.boundingRect(ctr)[0]))

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    print(x,y,w,h)

    # Make sure the contour area is a likely digit (optional, adjust sizes as needed)
    if w > 50 and h > 100:
        digit = gray[y:y+h, x:x+w]
        # digit_images.append(digit)

    if h > w:  # More height than width, pad width
        pad_size = abs((h - w)) // 2
        digit = cv2.copyMakeBorder(digit, pad_size//2, pad_size//2, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:  # More width than height, pad height
        pad_size = abs(w - h) // 2
        digit = cv2.copyMakeBorder(digit, pad_size, pad_size, pad_size//2, pad_size//2, cv2.BORDER_CONSTANT, value=[0,0,0])
    # Resize to 28x28
    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    digit_images.append(resized)
    # Draw rectangle around each digit on the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Resize and Normalize Digits
mnist_digits = []
for digit in digit_images:
    # Resize to 28x28
    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize pixel values to 0-1
    normalized = resized / 255.0
    mnist_digits.append(normalized)

# Show the original image with contours and the processed digits

fig,ax = plt.subplots(figsize=(12,6))
# Show the original image with bounding rectangles
ax.plot(1, len(mnist_digits) + 1, 1)
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert color for matplotlib
ax.set_title('Original with contours')
ax.axis('off')
plt.show()

fig = plt.figure(figsize=(12,6))
# Optionally visualize the processed digits
for i, digit in enumerate(mnist_digits):
    plt.subplot(1, len(mnist_digits) + 1, i + 2)
    plt.imshow(digit, cmap='gray' if i%2==0 else 'Blues')
    # 1]lt.title(f'Digit {i+1}')
    plt.axis('off')

plt.show()
