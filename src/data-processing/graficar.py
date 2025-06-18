import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE_NAME = 'loss_batch12_ssim.log'


data = {
    'Epoch': [],
    'Train Loss': [],
    'Val Loss': []
}


with open(FILE_NAME, 'r') as file:
    for line in file:
        
        match = re.search(r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val Loss: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            # Agrega los datos al diccionario
            data['Epoch'].append(epoch)
            data['Train Loss'].append(train_loss)
            data['Val Loss'].append(val_loss)

df = pd.DataFrame(data)


# for better visualization, change dt downj
# df_filtered = df[df['Epoch'].isin([1, 2, 3, 5, 10, 25, 50, 75, 100, 125, 150])]


sns.set(style='whitegrid')


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Epoch', y='Train Loss', label='Train Loss', marker='o')
sns.lineplot(data=df, x='Epoch', y='Val Loss', label='Val Loss', marker='o')


plt.title('Train Loss and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')




plt.legend()


plt.savefig(f'{FILE_NAME}.png')
plt.close()
