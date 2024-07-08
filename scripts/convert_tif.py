import os
from PIL import Image
import shutil

PATH = 'data/CASIA2'

old_tampered_dir = os.path.join(PATH, 'Tp')
new_tampered_dir = os.path.join(PATH, 'Tp2')


def process_images(input_folder, output_folder):
    # Asegúrate de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Recorre todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        # Verifica si es un archivo
        if os.path.isfile(input_path):
            # Si es un archivo .tif, conviértelo a .jpeg
            if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
                # Cambia la extensión a .jpeg
                output_filename = os.path.splitext(filename)[0] + '.jpeg'
                output_path = os.path.join(output_folder, output_filename)       
                try:
                    with Image.open(input_path) as img:
                        # Convierte y guarda la imagen como .jpeg
                        img.convert("RGB").save(output_path, "JPEG")
                    print(f"Convertido: {filename} -> {output_filename}")
                except Exception as e:
                    print(f"Error al convertir {filename}: {str(e)}")      
            # Si no es .tif, simplemente cópialo
            else:
                output_path = os.path.join(output_folder, filename)
                try:
                    shutil.copy2(input_path, output_path)
                    print(f"Copiado: {filename}")
                except Exception as e:
                    print(f"Error al copiar {filename}: {str(e)}")


def main():
    process_images(old_tampered_dir, new_tampered_dir)


if __name__ == "__main__":
    main()
