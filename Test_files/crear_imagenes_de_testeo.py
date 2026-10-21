from PIL import Image, ImageDraw
import os
import random

# --- Lista de resoluciones ---
resolutions = [
    (128, 128), (256, 256), (512, 512), (640, 480), (800, 600),
    (1024, 768), (1024, 1024), (1280, 960), (1600, 1200), (1920, 1440),
    (2048, 2048), (2560, 1920), (3840, 2880), (4096, 4096), (2*4096, 2*4096)
]

# --- Carpeta base de salida ---
base_folder = "imagenes_grises"
os.makedirs(base_folder, exist_ok=True)

# --- Generar 3 imágenes por cada resolución ---
for width, height in resolutions:
    # Crear carpeta para esta resolución
    folder_name = f"{width}x{height}"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, 4):
        # Crear imagen en escala de grises
        img = Image.new("L", (width, height))

        # Rellenar con un patrón aleatorio de grises
        pixels = img.load()
        for x in range(width):
            for y in range(height):
                pixels[x, y] = random.randint(0, 255)  # valor de gris entre 0 (negro) y 255 (blanco)

        # Agregar texto con resolución
        draw = ImageDraw.Draw(img)
        text = f"{width}x{height} #{i}"
        draw.text((10, 10), text, fill=255)

        # Guardar imagen en la subcarpeta correspondiente
        filename = f"img{i}.png"
        img.save(os.path.join(output_folder, filename))

print("✅ Imágenes generadas por carpetas en:", base_folder)
