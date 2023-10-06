import math
import random
from PIL import Image

# Voronoi Random Vector
def voronoi_random_vector(uv, angle_offset):
    m = [[15.27, 47.63], [99.41, 89.98]]
    uv = [math.sin(sum(uv[i] * m[i][j] for i in range(2))) % 1 for j in range(2)]
    return [math.sin(uv[1] + angle_offset) * 0.5 + 0.5, math.cos(uv[0] * angle_offset) * 0.5 + 0.5]

# Voronoi Noise
def voronoi_noise(uv, angle_offset, cell_density):
    g = [math.floor(coord * cell_density) for coord in uv]
    f = [coord * cell_density - g[i] for i, coord in enumerate(uv)]
    t = 8.0
    res = [8.0, 0.0, 0.0]
    
    for y in range(-1, 2):
        for x in range(-1, 2):
            lattice = [x, y]
            offset = voronoi_random_vector([lattice[i] + g[i] for i in range(2)], angle_offset)
            d = math.sqrt(sum((lattice[i] + offset[i] - f[i]) ** 2 for i in range(2)))
            
            if d < res[0]:
                res = [d, offset[0], offset[1]]
    return res

# Gradient Noise
def gradient_noise(uv, scale):
    return random.random()

# Main function
def main():
    width, height = 256, 256
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    
    for y in range(height):
        for x in range(width):
            uv = [x / width, y / height]
            voronoi = voronoi_noise(uv, 0.5, 14)
            grad_noise = gradient_noise(uv, 10)
            color = [int(255 * c) for c in [1.0, 1.0, 0.0]]
            
            # Multiply gradient noise with color
            color = [int(c * grad_noise) for c in color]
            
            # Multiply Voronoi with color
            color = [int(c * voronoi[0]) for c in color]
            
            pixels[x, y] = tuple(color)
    
    image.show()

if __name__ == "__main__":
    main()
