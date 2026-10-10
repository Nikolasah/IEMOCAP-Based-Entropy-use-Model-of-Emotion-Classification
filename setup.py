import subprocess
import re
import sys

def get_cuda_version():
    try:
        # Membaca informasi dari driver NVIDIA bawaan komputermu
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print("GPU NVIDIA tidak terdeteksi atau nvidia-smi tidak tersedia.")
        return None
    return None

def main():
    print("Mengecek spesifikasi GPU lokal...")
    cuda_version = get_cuda_version()
    
    # Menentukan versi PyTorch CUDA yang cocok
    if cuda_version is None:
        print("Beralih ke mode CPU (Tidak ada GPU terdeteksi).")
        pytorch_cmd = "pytorch torchvision torchaudio cpuonly -c pytorch"
    elif cuda_version >= 12.1:
        print(f"CUDA {cuda_version} terdeteksi. Menggunakan pytorch-cuda=12.1")
        pytorch_cmd = "pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    elif cuda_version >= 11.8:
        print(f"CUDA {cuda_version} terdeteksi. Menggunakan pytorch-cuda=11.8")
        pytorch_cmd = "pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    else:
        print(f"CUDA {cuda_version} terlalu lawas. Harap update driver NVIDIA kamu.")
        sys.exit(1)

    env_name = "transisi_emosi_311"
    
    # Command 1: Buat Env Python 3.11 & Install PyTorch yang cocok
    cmd_create = f"conda create -n {env_name} python=3.11 {pytorch_cmd} -y"
    
    # Command 2: Install core library (termasuk pandas, scikit-learn, librosa, opencv)
    cmd_conda_libs = f"conda install -n {env_name} -c conda-forge pandas numpy scikit-learn matplotlib seaborn librosa soundfile opencv jupyter tqdm -y"
    
    # Command 3: Pip install untuk NLP (IndoBERT) & Visual (MediaPipe) agar tidak kehalang dependensi
    cmd_pip_libs = f"conda run -n {env_name} pip install transformers mediapipe"

    print("\n--- Memulai proses otomatisasi instalasi ---")
    commands = [cmd_create, cmd_conda_libs, cmd_pip_libs]
    
    for cmd in commands:
        print(f"\nMenjalankan: {cmd}")
        subprocess.run(cmd, shell=True)

    print("\n✅ SELESAI! Environment berhasil dibuat dengan Python 3.11 dan GPU yang matching.")
    print(f"Silakan aktifkan dengan: conda activate {env_name}")

if __name__ == "__main__":
    main()