import argparse
import subprocess
import os

def download_kaggle_output(kernel_slug: str, output_dir: str):
    """
    Downloads the output of a Kaggle kernel to the specified directory.
    Uses the Kaggle CLI under the hood.
    """
    print(f"Downloading output for kernel: {kernel_slug}")
    print(f"Destination: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Using Kaggle CLI to pull kernel output
    command = [
        "kaggle", "kernels", "output", kernel_slug, "-p", output_dir
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Download completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading kernel output: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle kernel output files.")
    parser.add_argument("kernel_slug", help="The Kaggle kernel slug (e.g., fadhilelrizandamicr/my-kernel-name)")
    parser.add_argument("--output-dir", "-p", default="./output", help="Destination directory for the downloaded files")
    
    args = parser.parse_args()
    download_kaggle_output(args.kernel_slug, args.output_dir)
