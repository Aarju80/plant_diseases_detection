"""
Plant Disease Detection Application
Enhanced UI Design with Modern Styling
Author: AI Plant Disease Detector
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import json
import numpy as np
from PIL import Image
import gradio as gr
import pandas as pd
import os

# Configuration
MODEL_PATH = "plant_diseases_detect/model/resnet_crop_disease_best.pth"
LABEL_MAP_PATH = "plant_diseases_detect/label_map.json"

class PlantDiseaseDetector:
    def __init__(self, model_path, label_map_path):
        """Initialize the detector"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Convert string keys to integers if needed
        if isinstance(list(self.label_map.keys())[0], str):
            self.label_map = {int(k): v for k, v in self.label_map.items()}
        
        self.num_classes = len(self.label_map)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded successfully!")
        print(f"  Number of classes: {self.num_classes}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image, top_k=3):
        """Predict disease from image"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = probabilities.topk(top_k, dim=1)
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
            
            # Format results
            predicted_class = self.label_map[int(top_indices[0])]
            confidence = float(top_probs[0] * 100)
            
            top_predictions = []
            for idx, prob in zip(top_indices, top_probs):
                top_predictions.append({
                    'class': self.label_map[int(idx)],
                    'confidence': float(prob * 100)
                })
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize detector
print("\n" + "="*60)
print("🌿 Initializing Plant Disease Detection System...")
print("="*60 + "\n")

detector = PlantDiseaseDetector(MODEL_PATH, LABEL_MAP_PATH)

def predict_disease(image):
    """Gradio prediction function"""
    if image is None:
        return "Please upload an image!", None, None
    
    result = detector.predict(image, top_k=3)
    
    if 'error' in result:
        return f"Error: {result['error']}", None, None
    
    # Format main result with enhanced styling
    disease_name = result['predicted_class'].replace('_', ' ').replace('__', ' - ')
    confidence_color = "🟢" if result['confidence'] > 80 else "🟡" if result['confidence'] > 60 else "🟠"
    
    main_result = f"""
<div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 30px; border-radius: 20px; color: white; box-shadow: 0 10px 30px rgba(5,150,105,0.4);">
    <h2 style="margin: 0 0 15px 0; font-size: 22px; font-weight: 600; opacity: 0.95;">🔬 Diagnosis Result</h2>
    <h1 style="margin: 0 0 25px 0; font-size: 36px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">{disease_name}</h1>
    <div style="background: rgba(255,255,255,0.25); padding: 18px; border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3);">
        <p style="margin: 0; font-size: 20px; font-weight: 600;">{confidence_color} Confidence: {result['confidence']:.2f}%</p>
    </div>
</div>
"""
    
    # Format top 3 predictions with modern card design
    top3_text = """
<div style="margin-top: 20px;">
    <h3 style="color: #047857; margin-bottom: 20px; font-size: 22px; font-weight: 700;">📊 Detailed Prediction Analysis</h3>
"""
    
    colors = ['#059669', '#10b981', '#34d399']
    for i, pred in enumerate(result['top_predictions'], 1):
        disease = pred['class'].replace('_', ' ').replace('__', ' - ')
        confidence = pred['confidence']
        
        top3_text += f"""
    <div style="background: white; padding: 20px; margin-bottom: 15px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid {colors[i-1]};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-weight: 700; color: #1f2937; font-size: 17px;">{i}. {disease}</span>
            <span style="font-weight: 800; color: {colors[i-1]}; font-size: 20px;">{confidence:.2f}%</span>
        </div>
        <div style="background: #f3f4f6; height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="background: {colors[i-1]}; height: 100%; width: {confidence}%; border-radius: 5px; transition: width 0.3s ease;"></div>
        </div>
    </div>
"""
    
    top3_text += "</div>"
    
    # Create pandas DataFrame for chart
    chart_data = pd.DataFrame({
        'Disease': [pred['class'].replace('_', ' ').replace('__', ' - ') 
                   for pred in result['top_predictions']],
        'Confidence (%)': [pred['confidence'] for pred in result['top_predictions']]
    })
    
    return main_result, top3_text, chart_data

# Custom CSS for enhanced styling
custom_css = """
#component-0 {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

.main-header {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    padding: 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(5,150,105,0.3);
}

.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.stat-card {
    background: rgba(255,255,255,0.25);
    padding: 18px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.3);
}

button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    border: none !important;
    font-weight: bold !important;
    transition: all 0.3s !important;
    font-size: 16px !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4) !important;
}

.tips-box {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(16,185,129,0.2);
}

.disease-list {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    padding: 25px;
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border-radius: 15px;
    margin-top: 30px;
}
"""

# Create Gradio Interface with Enhanced Design
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="🌿 Plant Disease Detector") as demo:
    
    gr.HTML(
        """
        <div class="main-header">
            <h1 style="margin: 0; font-size: 42px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">🌿 AI Plant Disease Detection</h1>
            <p style="margin: 18px 0 0 0; font-size: 19px; opacity: 0.95; font-weight: 500;">Advanced Deep Learning for Agricultural Diagnostics</p>
            
            <div class="stats-container">
                <div class="stat-card">
                    <div style="font-size: 32px; font-weight: 800;">98.69%</div>
                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 32px; font-weight: 800;">100%</div>
                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Top-3</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 32px; font-weight: 800;">15</div>
                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Classes</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 32px; font-weight: 800;">⚡</div>
                    <div style="font-size: 14px; opacity: 0.9; font-weight: 500;">Real-time</div>
                </div>
            </div>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Upload Plant Leaf Image",
                type="numpy",
                height=400
            )
            
            predict_btn = gr.Button(
                "🔍 Analyze Disease",
                variant="primary",
                size="lg"
            )
            
            gr.HTML(
                """
                <div class="tips-box">
                    <h3 style="margin: 0 0 18px 0; font-size: 20px; font-weight: 700;">💡 Tips for Best Results</h3>
                    <ul style="margin: 0; padding-left: 20px; line-height: 2; font-size: 15px;">
                        <li>Use clear, well-lit images</li>
                        <li>Focus on the affected leaf area</li>
                        <li>Avoid blurry or dark images</li>
                        <li>JPEG, JPG, or PNG formats</li>
                    </ul>
                </div>
                """
            )
        
        with gr.Column(scale=1):
            output_result = gr.HTML(label="Results")
            output_top3 = gr.HTML(label="Detailed Analysis")
            output_chart = gr.BarPlot(
                x="Disease",
                y="Confidence (%)",
                title="📈 Confidence Distribution",
                y_lim=[0, 100],
                height=300,
                width=500,
                color="#10b981"
            )
    
    gr.HTML(
        """
        <div class="disease-list">
            <h3 style="color: #065f46; margin-bottom: 25px; font-size: 26px; text-align: center; font-weight: 800;">📋 Supported Plant Diseases</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div style="background: #f0fdf4; padding: 25px; border-radius: 15px; border: 2px solid #86efac;">
                    <h4 style="margin: 0 0 12px 0; color: #065f46; font-size: 20px; font-weight: 700;">🫑 Pepper</h4>
                    <p style="margin: 0; color: #047857; line-height: 1.8; font-size: 15px;">Bacterial spot, Healthy</p>
                </div>
                
                <div style="background: #f0fdf4; padding: 25px; border-radius: 15px; border: 2px solid #4ade80;">
                    <h4 style="margin: 0 0 12px 0; color: #065f46; font-size: 20px; font-weight: 700;">🥔 Potato</h4>
                    <p style="margin: 0; color: #047857; line-height: 1.8; font-size: 15px;">Early blight, Late blight, Healthy</p>
                </div>
                
                <div style="background: #f0fdf4; padding: 25px; border-radius: 15px; border: 2px solid #10b981;">
                    <h4 style="margin: 0 0 12px 0; color: #065f46; font-size: 20px; font-weight: 700;">🍅 Tomato</h4>
                    <p style="margin: 0; color: #047857; line-height: 1.8; font-size: 15px;">Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p style="margin: 0; font-size: 17px; font-weight: 600;">
                <strong>Model:</strong> ResNet50 | <strong>Framework:</strong> PyTorch | <strong>Dataset:</strong> PlantVillage
            </p>
            <p style="margin: 12px 0 0 0; font-size: 15px; opacity: 0.95;">
                Powered by Advanced Deep Learning Technology
            </p>
        </div>
        """
    )
    
    predict_btn.click(
        fn=predict_disease,
        inputs=input_image,
        outputs=[output_result, output_top3, output_chart]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌿 Plant Disease Detection System")
    print("="*60)
    print("\n🚀 Starting application...")
    print("  Opening in your default browser...")
    print("\n💡 Press Ctrl+C to stop the application")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )