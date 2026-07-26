import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  
        self.fc2 = nn.Linear(512, 38)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание болезней растений")
        self.root.geometry("960x540") 

        self.model = Net()
        self.model = nn.DataParallel(self.model)
        try:
            self.model.load_state_dict(torch.load("model8.pth", map_location=torch.device('cpu')))
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror("Ошибка", "Файл весов model8.pth не найден! Поместите его в папку со скриптом.")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        
        self.image_label = tk.Label(self.root, text="Не выбрано изображение", font=("Arial", 14))
        self.image_label.pack(pady=20)
        
        self.canvas = tk.Canvas(self.root, width=256, height=256, bg="lightgray")
        self.canvas.pack(pady=20)

        self.classify_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"))
        self.classify_label.pack(pady=20)
        
        self.load_button = tk.Button(self.root, text="Загрузить изображение", font=("Arial", 12), command=self.load_image)
        self.load_button.pack(pady=10)
        
        self.classify_button = tk.Button(self.root, text="Распознать", font=("Arial", 12), command=self.classify)
        self.classify_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path).convert('RGB')
            self.tk_image = ImageTk.PhotoImage(self.image.resize((256, 256)))
            self.canvas.create_image(128, 128, image=self.tk_image)
            self.image_label.config(text=f"Загружено: {file_path.split('/')[-1]}")
            self.classify_label.config(text="")
        
    def classify(self):
        if not hasattr(self, 'image'):
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return

        with torch.no_grad():
            input_tensor = self.transform(self.image).unsqueeze(0)   
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            
            raw_class_name = self.classes[predicted.item()]
            clean_class_name = raw_class_name.replace('___', ' - ').replace('_', ' ')
            
            self.classify_label.config(text=f"Результат: {clean_class_name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()