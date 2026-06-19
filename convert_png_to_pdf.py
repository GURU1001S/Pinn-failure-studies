import os
from PIL import Image
import glob

def convert_all_pngs_to_pdf(directory):
    png_files = glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)
    converted_count = 0
    for png_path in png_files:
        pdf_path = png_path.rsplit('.', 1)[0] + '.pdf'
        if not os.path.exists(pdf_path):
            try:
                img = Image.open(png_path).convert('RGB')
                img.save(pdf_path)
                print(f"Converted: {png_path} -> {pdf_path}")
                converted_count += 1
            except Exception as e:
                print(f"Error converting {png_path}: {e}")
        else:
            print(f"Already exists: {pdf_path}")
    print(f"Total converted: {converted_count}")

if __name__ == "__main__":
    results_dir = r"d:\Games\FAILURE STUDIES\results"
    convert_all_pngs_to_pdf(results_dir)
