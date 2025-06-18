import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import os
import json
import piq
import skimage.metrics
from scipy import ndimage


class WaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')
    
    def forward(self, x):
        yl, yh = self.dwt(x)
        cA = yl.unsqueeze(2)
        cH, cV, cD = torch.unbind(yh[0], dim=2)
        combined = torch.cat([cA, cH.unsqueeze(2), cV.unsqueeze(2), cD.unsqueeze(2)], dim=2)
        batch, channels, _, h, w = combined.shape
        return combined.view(batch, channels * 4, h, w)

class InverseWaveletLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
    
    def forward(self, x):
        batch, total_channels, h, w = x.shape
        channels = total_channels // 4
        x = x.view(batch, channels, 4, h, w)
        yl = x[:, :, 0]
        yh = [x[:, :, 1:]]
        return self.idwt((yl, yh))

class DeepCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if i > 0 else 4, 64, 3, padding=1),
                nn.ReLU(inplace=True))
            for i in range(10)
        ])
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 4, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x0 = x
        x1 = self.conv_layers[0](x)
        x2 = self.conv_layers[1](x1)
        out = x2
        for i in range(2, 10):
            out = self.conv_layers[i](out)
        out = self.conv11(out + x2)
        out = self.conv12(out + x1)
        out = self.conv13(out) + x0
        return out

class SubpixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.dec1_upsample = SubpixelUpsample(128, 64)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.dec2_upsample = SubpixelUpsample(64, 32)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True))
        self.dec3_upsample = SubpixelUpsample(32, 16)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(20, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        x_wave_input = x
        enc1_out = self.enc1(x_wave_input)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        dec1 = self.dec1_upsample(enc3_out)
        dec1 = torch.cat([dec1, enc2_out], dim=1)
        dec1 = self.dec1_conv(dec1)
        dec2 = self.dec2_upsample(dec1)
        dec2 = torch.cat([dec2, enc1_out], dim=1)
        dec2 = self.dec2_conv(dec2)
        dec3 = self.dec3_upsample(dec2)
        dec3 = torch.cat([dec3, x_wave_input], dim=1)
        dec3 = self.dec3_conv(dec3)
        return self.final_conv(dec3)

class WaCAEN(nn.Module):
    def __init__(self, in_channels=1, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.wavelet_layer = WaveletLayer(wavelet)
        self.inverse_wavelet_layer = InverseWaveletLayer(wavelet)
        self.branch1 = DeepCNNBranch()
        self.branch2 = ConvAutoencoder()

    def forward(self, x):
        x_wave = self.wavelet_layer(x)
        branch1_out = self.branch1(x_wave)
        branch2_out = self.branch2(x_wave)
        combined = branch1_out + branch2_out
        return self.inverse_wavelet_layer(combined)

def load_model(checkpoint_path, device):
    model = WaCAEN(in_channels=1, wavelet='haar').to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_and_preprocess_image(image_path, size=(256, 256)):
    
    image = read_image(image_path, ImageReadMode.GRAY).float()
    
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Lambda(lambda x: x / 255.0)
    ])
    
    return transform(image).unsqueeze(0)  

def calculate_metrics(original, restored, blurry):
    """Calcula todas las métricas de calidad de imagen"""
    results = {}
    
    
    original_np = original.squeeze().cpu().numpy() * 255
    restored_np = restored.squeeze().cpu().numpy() * 255
    blurry_np = blurry.squeeze().cpu().numpy() * 255
    
    
    original_t = torch.tensor(original_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    restored_t = torch.tensor(restored_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    
    results['MSE'] = skimage.metrics.mean_squared_error(original_np, restored_np)
    
    
    results['RMSE'] = np.sqrt(results['MSE'])
    
    
    results['RMSE-SW'] = calculate_rmse_sw(original_np, restored_np)
    
    
    results['PSNR'] = skimage.metrics.peak_signal_noise_ratio(
        original_np, restored_np, data_range=255)
    
    
    results['UQI'] = piq.uqi(original_t, restored_t, data_range=255).item()
    
    
    ssim_score = skimage.metrics.structural_similarity(
        original_np, restored_np, win_size=7, data_range=255)
    results['SSIM'] = ssim_score
    
    
    results['ERGAS'] = piq.ergas(original_t, restored_t, data_range=255).item()
    
    
    results['SCC'] = calculate_scc(original_np, restored_np)
    
    
    results['RASE'] = calculate_rase(original_np, restored_np)
    
    
    results['SAM'] = calculate_sam(original_np, restored_np)
    
    
    results['MS-SSIM'] = piq.multi_scale_ssim(
        original_t, restored_t, data_range=255).item()
    
    
    results['VIFP'] = piq.vif_p(original_t, restored_t, data_range=255).item()
    
    
    results['PSNR-B'] = calculate_psnr_b(original_np, restored_np)
    
    
    results['CED'] = np.sqrt(np.mean((original_np - restored_np)**2))
    
    
    results['NED'] = np.linalg.norm(original_np - restored_np) / np.linalg.norm(original_np)
    
    return results


def calculate_rmse_sw(original, restored, window_size=8):
    """Calcula RMSE con ventana deslizante"""
    h, w = original.shape
    rmse_values = []
    for i in range(0, h - window_size + 1, window_size//2):
        for j in range(0, w - window_size + 1, window_size//2):
            orig_patch = original[i:i+window_size, j:j+window_size]
            res_patch = restored[i:i+window_size, j:j+window_size]
            mse = np.mean((orig_patch - res_patch) ** 2)
            rmse_values.append(np.sqrt(mse))
    return np.mean(rmse_values)

def calculate_scc(original, restored):
    """Coeficiente de Correlación Espacial"""
    orig_flat = original.flatten()
    res_flat = restored.flatten()
    return np.corrcoef(orig_flat, res_flat)[0, 1]

def calculate_rase(original, restored):
    """Relative Average Spectral Error"""
    mse = np.mean((original - restored) ** 2)
    ref_mean = np.mean(original)
    return 100 / ref_mean * np.sqrt(mse)

def calculate_sam(original, restored):
    """Spectral Angle Mapper (adaptado para escala de grises)"""
    
    orig_flat = original.flatten()
    res_flat = restored.flatten()
    dot_product = np.dot(orig_flat, res_flat)
    norm_orig = np.linalg.norm(orig_flat)
    norm_res = np.linalg.norm(res_flat)
    return np.arccos(dot_product / (norm_orig * norm_res))

def calculate_psnr_b(original, restored):
    """PSNR con corrección de bloque"""
    
    block_artifacts = ndimage.sobel(restored) > 50
    
    weighted_mse = np.mean((original - restored)**2 * (1 + block_artifacts*0.5))
    if weighted_mse == 0:
        return float('inf')
    return 20 * np.log10(255 / np.sqrt(weighted_mse))

def save_results_to_pdf(blurry, restored, original, pdf_path):
    """Guarda las imágenes en un PDF"""
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        
        blurry_np = blurry.squeeze().numpy()
        restored_np = restored.squeeze().numpy()
        original_np = original.squeeze().numpy()
        
        
        axes[0].imshow(blurry_np, cmap='gray')
        axes[0].set_title('Imagen Borrosa')
        axes[0].axis('off')
        
        
        axes[1].imshow(restored_np, cmap='gray')
        axes[1].set_title('Imagen Recuperada')
        axes[1].axis('off')
        
        
        axes[2].imshow(original_np, cmap='gray')
        axes[2].set_title('Imagen Original')
        axes[2].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

def metrics_to_markdown(metrics, output_path):
    """Guarda las métricas en una tabla Markdown"""
    md_content = "| Métrica | Valor |\n|---------|-------|\n"
    for metric, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        md_content += f"| {metric} | {value} |\n"
    
    with open(output_path, 'w') as f:
        f.write(md_content)

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    
    CHECKPOINT_PATH = "../MRI/Checkpoints/best_model_epoch_300_loss_0.0004_20250612_005824.pt" 
    BLURRY_IMAGE_PATH = "./pictures/IM000000a.png"
    TARGET_IMAGE_PATH = "./pictures/IM000000t.png"
    OUTPUT_DIR = "resultados_evaluacion"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    model = load_model(CHECKPOINT_PATH, device)
    print("Modelo cargado correctamente")
    
    
    blurry_img = load_and_preprocess_image(BLURRY_IMAGE_PATH)
    target_img = load_and_preprocess_image(TARGET_IMAGE_PATH)
    
    
    start_time = time.time()
    with torch.no_grad():
        restored_img = model(blurry_img.to(device))
    inference_time = time.time() - start_time
    
    
    metrics = calculate_metrics(target_img, restored_img, blurry_img)
    metrics['TIME'] = f"{inference_time:.4f} segundos"
    
    
    
    pdf_path = os.path.join(OUTPUT_DIR, "comparacion_imagenes.pdf")
    save_results_to_pdf(
        blurry_img.cpu(), 
        restored_img.cpu(), 
        target_img.cpu(),
        pdf_path
    )
    
    
    md_path = os.path.join(OUTPUT_DIR, "metricas_calidad.md")
    metrics_to_markdown(metrics, md_path)
    
    print(f"\nResultados guardados en: {OUTPUT_DIR}")
    print("PDF con imágenes generado:", pdf_path)
    print("Tabla de métricas generada:", md_path)
    
    
    print("\nMétricas de Calidad:")
    print("---------------------")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()