import math
import random
import cv2
import numpy as np
from PIL import Image

class WeatherAugmentation:
    def __init__(
        self, 
        rain_prob=0.3, snow_prob=0.3, fog_prob=0.3,
        rain_intensity=1.0, snow_intensity=1.0, fog_intensity=1.0,
        blur_strength=201  # Larger values = stronger blur
    ):
        self.rain_prob = rain_prob
        self.snow_prob = snow_prob
        self.fog_prob = fog_prob
        self.rain_intensity = rain_intensity
        self.snow_intensity = snow_intensity
        self.fog_intensity = fog_intensity
        self.blur_strength = blur_strength  # kernel size for Gaussian blur

    def __call__(self, img):
        # Convert PIL to OpenCV BGR
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if random.random() < self.rain_prob:
            img_cv = self.add_blurred_raindrops(img_cv, self.rain_intensity)
        if random.random() < self.snow_prob:
            img_cv = self.add_snow(img_cv, self.snow_intensity)
        if random.random() < self.fog_prob:
            img_cv = self.add_fog(img_cv, self.fog_intensity)

        # Convert back to PIL
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_cv)

    def add_blurred_raindrops(self, img, intensity=1.0):
        """
        Simulate raindrops by placing circular areas of heavy Gaussian blur on the image.
        Droplets do not overlap.
        """
        distorted_img = img.copy()
        rows, cols, _ = distorted_img.shape

        num_drops = int(15 * intensity)
        droplets = []
        max_attempts = 100

        # Generate non-overlapping droplet positions
        for _ in range(num_drops):
            for attempt in range(max_attempts):
                center_x = random.randint(0, cols - 1)
                center_y = random.randint(0, rows - 1)
                radius = random.randint(8, 35)

                # Check collision with existing droplets
                collision = False
                for (ex, ey, er) in droplets:
                    dist = math.hypot(ex - center_x, ey - center_y)
                    if dist < (radius + er):
                        collision = True
                        break

                if not collision:
                    droplets.append((center_x, center_y, radius))
                    break

        # Apply gaussian blur within each droplet region
        for (cx, cy, radius) in droplets:
            x_start = max(cx - radius, 0)
            y_start = max(cy - radius, 0)
            x_end = min(cx + radius, cols)
            y_end = min(cy + radius, rows)

            region = distorted_img[y_start:y_end, x_start:x_end]
            if region.size == 0:
                continue

            blurred_region = cv2.GaussianBlur(
                region, 
                (self.blur_strength, self.blur_strength), 
                0
            )

            h, w, _ = region.shape
            droplet_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(droplet_mask, (w // 2, h // 2), radius, 1, thickness=-1)
            droplet_mask = cv2.GaussianBlur(droplet_mask, (7, 7), 0)
            droplet_mask = droplet_mask[..., None]  # shape (h, w, 1)

            alpha = 0.9
            blended = region * (1 - alpha * droplet_mask) + blurred_region * (alpha * droplet_mask)
            blended = blended.astype(np.uint8)

            distorted_img[y_start:y_end, x_start:x_end] = blended

        return distorted_img

    def add_snow(self, img, intensity=1.0):
        snow_layer = np.zeros_like(img, dtype=np.uint8)
        rows, cols, _ = img.shape
        num_flakes = int(200 * intensity)
        for _ in range(num_flakes):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            radius = random.randint(5, 20)
            cv2.circle(snow_layer, (x, y), radius, (255, 255, 255), -1)
        snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)
        blended = cv2.addWeighted(img, 0.7, snow_layer, 0.3, 0)
        return blended

    def add_fog(self, img, intensity=1.0):
        fog_layer = np.full(img.shape, 255, dtype=np.uint8)
        alpha = 0.3 * intensity
        blended = cv2.addWeighted(img, 1 - alpha, fog_layer, alpha, 0)
        blended = cv2.GaussianBlur(blended, (7, 7), 0)
        return blended
