import math
import torch
import torch.nn as nn
import numpy as np
import csv
from torchvision.io import read_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob
from PIL import Image
import argparse
from pytorch_wavelets import DWTForward, DWTInverse
from skimage.metrics import structural_similarity as ssim_sk
from sewar.full_ref import vifp


TEST_SOURCE_DIR = '../MRI/Test/source_a'
TEST_TARGET_DIR = '../MRI/Test/target'
CHECKPOINT_PATH = '../MRI/Checkpoints/best_model.pt'
PDF_PATH = 'comparison_results_a.pdf'
INPUT_SIZE = (256, 256)  


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



def preprocess_image(image_path, target_size):
    img = read_image(image_path)
    img = img.float()
    
    if img.shape[0] == 3:
        img = img.mean(dim=0, keepdim=True)
    
    img = img / 255.0
    img = img.unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
    
    return img.squeeze(0)


def calculate_psnr(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_snr(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - restored) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * math.log10(signal_power / noise_power)
    return snr

def calculate_mse(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((original - restored) ** 2)
    return mse

def calculate_rmse(original, restored):
    mse = calculate_mse(original, restored)
    rmse = math.sqrt(mse)
    return rmse

def calculate_psnr(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0 
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, restored, multichannel=False):
    
    
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    
    
    data_range = restored.max() - restored.min()
    ssim_val = ssim_sk(original, restored, data_range=data_range, multichannel=multichannel)
    return ssim_val

def calculate_ms_ssim(original, restored, multichannel=False):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    
    data_range = restored.max() - restored.min()
    ms_ssim_val = ssim_sk(original, restored, data_range=data_range, multichannel=multichannel, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03, dynamic_range=255) 
    return ms_ssim_val


def calculate_vifp(original, restored):
    
    
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    vifp_val = vifp(original, restored)
    return vifp_val



def load_model(model_path):
    device = torch.device('cpu')
    model = WaCAEN(in_channels=1, wavelet='haar')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def process_images_and_generate_pdf(test_source_dir, test_target_dir, checkpoint_path, pdf_path):
    model = load_model(checkpoint_path)
    print(f"Modelo cargado desde {checkpoint_path}")
    
    source_paths = sorted(glob.glob(os.path.join(test_source_dir, '*.png')))
    print(f"Encontradas {len(source_paths)} imágenes fuente para procesar")
    
    with PdfPages(pdf_path) as pdf:
        for i, source_path in enumerate(source_paths):
            filename = os.path.basename(source_path)
            target_path = os.path.join(test_target_dir, filename)
            
            if not os.path.exists(target_path):
                print(f"⚠️ No se encontró target para {filename}. Saltando...")
                continue
                
            print(f"Procesando imagen {i+1}/{len(source_paths)}: {filename}")
            
            try:
                source_tensor = preprocess_image(source_path, INPUT_SIZE).unsqueeze(0)
                target_tensor = preprocess_image(target_path, INPUT_SIZE).unsqueeze(0)
                
                with torch.no_grad():
                    restored_tensor = model(source_tensor)
                
                source_np = (source_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                restored_np = (restored_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                target_np = (target_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                
                psnr_value = calculate_psnr(target_np, restored_np)
                snr_value = calculate_snr(target_np, restored_np)
                mse_value = calculate_mse(target_np, restored_np)
                rmse_value = calculate_rmse(target_np, restored_np)
                ssim_value = calculate_ssim(target_np, restored_np, multichannel=True)
                ms_ssim_value = calculate_ms_ssim(target_np, restored_np, multichannel=True)
                vifp_value = calculate_vifp(target_np, restored_np)

                
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                
                axs[0].imshow(source_np, cmap='gray')
                
                axs[0].axis('off')
                
                axs[1].imshow(restored_np, cmap='gray')
                
                axs[1].axis('off')
                
                axs[2].imshow(target_np, cmap='gray')
                
                axs[2].axis('off')
                
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")
                continue
    
    print(f"\n✅ Proceso completado! Resultados guardados en: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restauración de imágenes MRI usando WaCAEN')
    parser.add_argument('--test_source', type=str, default=TEST_SOURCE_DIR, 
                        help='Directorio con imágenes fuente a restaurar')
    parser.add_argument('--test_target', type=str, default=TEST_TARGET_DIR,
                        help='Directorio con imágenes target (objetivo)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Ruta al checkpoint del modelo (.pt)')
    parser.add_argument('--output_pdf', type=str, default=PDF_PATH,
                        help='Ruta de salida para el PDF de resultados')
    
    args = parser.parse_args()
    
    process_images_and_generate_pdf(
        test_source_dir=args.test_source,
        test_target_dir=args.test_target,
        checkpoint_path=args.checkpoint,
        pdf_path=args.output_pdf
    )