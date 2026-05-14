"""Create a synthetic fundus image for testing."""
import cv2
import numpy as np
import os

os.makedirs('sample_data', exist_ok=True)

# Create a dark background
img = np.zeros((600, 600, 3), dtype=np.uint8)

# Draw the fundus circle (orange-red retina)
cv2.circle(img, (300, 300), 250, (20, 40, 80), -1)
cv2.circle(img, (300, 300), 248, (30, 70, 140), -1)

# Optic disc (bright yellowish circle)
cv2.circle(img, (380, 300), 40, (60, 130, 200), -1)
cv2.circle(img, (380, 300), 35, (80, 160, 230), -1)

# Macula (dark center)
cv2.circle(img, (260, 300), 30, (15, 35, 70), -1)

# Blood vessels (dark red lines radiating from optic disc)
np.random.seed(42)
for i in range(30):
    angle = np.random.uniform(0, 2 * np.pi)
    length = np.random.randint(80, 200)
    x1, y1 = 380, 300
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))
    thickness = np.random.randint(1, 4)
    color = (10, 20, 50)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Add some microaneurysms (small bright dots) to simulate DR
for i in range(8):
    x = np.random.randint(150, 450)
    y = np.random.randint(150, 450)
    r = np.random.randint(2, 5)
    cv2.circle(img, (x, y), r, (40, 50, 170), -1)

cv2.imwrite('sample_data/test_fundus.jpg', img)
print('[OK] Synthetic fundus test image created at sample_data/test_fundus.jpg')
