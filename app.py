from flask import Flask
from flask import render_template, request, redirect, url_for, jsonify
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import base64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50()
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
model.load_state_dict(torch.load(
    'AI/model/model-resnet50.pth', map_location=device))
model.eval().to(device)

app = Flask(__name__)


def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = Transform(image)
    image = image.unsqueeze(0)
    return image.to(device)


def predict(image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    return preds.item()


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if request.files:
            res = ""
            for fi in request.files:
                file = request.files[fi]
                image = Image.open(file)
                prediction = predict(image, model)
                res = prediction
            return jsonify(res)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
