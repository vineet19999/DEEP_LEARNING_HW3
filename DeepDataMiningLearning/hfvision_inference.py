from transformers import AutoModelForObjectDetection, AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import requests
import numpy as np
import cv2
import os
import evaluate
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ColorJitter,
    RandomRotation,
    GaussianBlur
)
import albumentations  # pip install albumentations
from time import perf_counter
from sklearn.metrics import precision_recall_fscore_support

from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
from DeepDataMiningLearning.hfvisionmain import load_visionmodel, load_dataset
from DeepDataMiningLearning.detection.models import create_detectionmodel

# tasks: "depth-estimation", "image-classification", "object-detection"
class MyVisionInference():
    def __init__(self, model_name, model_path="", model_type="huggingface", task="image-classification", cache_dir="./output", gpuid='0', scale='x') -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.transforms = None
        self.id2label = None
        if isinstance(model_name, str) and model_type == "huggingface":
            if model_path and os.path.exists(model_path):
                model_name_or_path = model_path
            else:
                model_name_or_path = model_name
            self.model, self.image_processor = load_visionmodel(model_name_or_path=model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=cache_dir, trust_remote_code=True)
            self.id2label = self.model.config.id2label
        elif isinstance(model_name, str) and task == "image-classification":  # torch model
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)  # 'resnet18'
            labels = load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task == "object-detection":  # torch model
            self.model, self.image_processor, labels = create_detectionmodel(modelname=model_name, num_classes=None, ckpt_file=model_path, device=self.device, scale=scale)
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task == "depth-estimation":  # torch model
            self.model = torch.hub.load('intel-isl/MiDaS', model_name, pretrained=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transforms = transforms.dpt_transform

        self.model = self.model.to(self.device)
        self.model.eval()

    def batch_inference(self, images):
        try:
            inputs = [self.image_processor(image, return_tensors="pt") for image in images]
            pixel_values = torch.cat([input['pixel_values'] for input in inputs], dim=0).to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values)
            return outputs
        except Exception as e:
            print(f"An error occurred during batch inference: {str(e)}")
            return None

    def mypreprocess(self, inp):
        manual_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.4, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.transforms is not None:
            inp = self.transforms(inp)
        else:
            inp = manual_transforms(inp).unsqueeze(0)
        return inp

    def __call__(self, image_url):
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
            self.image, self.org_sizeHW = read_image(image, use_pil=True, use_cv2=False, output_format='numpy', plotfig=False)
            print(f"Shape of the NumPy array: {self.image.shape}")

            if self.image_processor is not None:
                inputs = self.image_processor(self.image, return_tensors="pt").pixel_values
            else:
                inputs = self.mypreprocess(self.image)

            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)

            if self.task == "image-classification":
                results = self.classification_postprocessing(outputs)
            elif self.task == "depth-estimation":
                results = self.depth_postprocessing(outputs, recolor=True)
            elif self.task == "object-detection":
                results = self.objectdetection_postprocessing(outputs)

            return results

        except Exception as e:
            print(f"An error occurred while processing the image from {image_url}: {str(e)}")
            return None

    def objectdetection_postprocessing(self, outputs, threshold=0.3):
        target_sizes = torch.tensor([self.org_sizeHW])
        if self.model_type == "huggingface":
            results = self.image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        else:  # torch model
            imgsize = self.inputs.shape[2:]
            results_list = self.image_processor.postprocess(preds=outputs, newimagesize=imgsize, origimageshapes=target_sizes)
            results = results_list[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {self.id2label[str(int(label.item()))]} with confidence {round(score.item(), 3)} at location {box}")
        pilimage = Image.fromarray(self.image)
        draw = ImageDraw.Draw(pilimage)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), self.id2label[str(int(label.item()))], fill="white")
        pilimage.save("output/ImageDraw.png")
        return pilimage

    def depth_postprocessing(self, outputs, recolor=True):
        try:
            if self.model_type == "huggingface":
                predicted_depth = outputs.predicted_depth
            else:
                predicted_depth = outputs

            if predicted_depth.dim() == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            target_size = self.org_sizeHW

            prediction = torch.nn.functional.interpolate(
                predicted_depth,
                size=target_size,
                mode="bicubic",
                align_corners=False
            )

            depth = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
            formatted = depth.squeeze().cpu().numpy().astype(np.uint8)

            if recolor:
                formatted = cv2.applyColorMap(formatted, cv2.COLORMAP_HOT)[:, :, ::-1]

            depth_image = Image.fromarray(formatted)
            depth_image.save("data/depth_testresult.jpg")
            depth_image.show()

            return depth_image

        except Exception as e:
            print(f"Error processing depth image: {str(e)}")
            return None

    def classification_postprocessing(self, outputs):
        if self.model_type == "huggingface":
            logits = outputs.logits
        else:
            logits = outputs
        predictions = torch.nn.functional.softmax(logits[0], dim=0)
        predictions = predictions.cpu().numpy()
        confidences = {self.id2label[i]: float(predictions[i]) for i in range(len(self.id2label))}

        predmax_idx = np.argmax(predictions, axis=-1)
        predmax_label = self.id2label[predmax_idx]
        predmax_confidence = float(predictions[predmax_idx])
        print(f"predmax_idx: {predmax_idx}, predmax_label: {predmax_label}, predmax_confidence: {predmax_confidence}")
        return confidences

def compute_additional_metrics(preds, labels):
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def vision_inferencetest(model_name_or_path, task="image-classification", mycache_dir=None):
    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)

    dataset = load_dataset("huggingface/cats-image", cache_dir=mycache_dir)
    image = dataset["test"]["image"][0]
    image.save("test.jpg")

    myinference = MyVisionInference(model_name=model_name_or_path, task=task, model_type="huggingface", cache_dir=mycache_dir)
    confidences = myinference(image)
    print(confidences)

    model, image_processor = load_visionmodel(model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=mycache_dir, trust_remote_code=True)
    id2label = model.config.id2label

    inputs = image_processor(image.convert("RGB"), return_tensors="pt")
    print(inputs.pixel_values.shape)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    predictions = torch.nn.functional.softmax(logits[0], dim=0)
    confidences = {id2label[i]: float(predictions[i]) for i in range(len(id2label))}

    pred = logits.argmax(dim=-1)
    predicted_class_idx = pred[0].item()
    print("Predicted class:", id2label[predicted_class_idx])
    return confidences

def clip_test(mycache_dir=None):
    url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
    candidate_labels = ["tree", "car", "bike", "cat"]
    image = Image.open(requests.get(url, stream=True).raw)
    checkpoint = "openai/clip-vit-large-patch14"
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint, cache_dir=mycache_dir)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)

    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": candidate_label}
        for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
    ]

    print(result)

def MyVisionInference_depthtest(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("data/depth_test.jpg")

    checkpoint = "Intel/dpt-large"
    depthinference = MyVisionInference(model_name=checkpoint, model_type="huggingface", task="depth-estimation", cache_dir=mycache_dir)
    results = depthinference(image=image)

def depth_test(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("data/depth_test.jpg")

    checkpoint = "Intel/dpt-large"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=mycache_dir)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    predicted_depth_input = predicted_depth.unsqueeze(1)
    target_size = image.size[::-1]
    prediction = torch.nn.functional.interpolate(
        predicted_depth_input,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save("data/depth_testresult.jpg")

def object_detection(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)

    checkpoint = "facebook/detr-resnet-50"
    myinference = MyVisionInference(model_name=checkpoint, task="object-detection", model_type="huggingface", cache_dir=mycache_dir)
    results = myinference(image)

    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint, cache_dir=mycache_dir)

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
    image.save("output/ImageDraw.png")

def test_inference():
    mycache_dir = r"D:\Cache\huggingface"
    if os.environ.get('HF_HOME') is not None:
        mycache_dir = os.environ['HF_HOME']
    object_detection(mycache_dir=mycache_dir)

    depth_test(mycache_dir=mycache_dir)
    clip_test(mycache_dir=mycache_dir)
    confidences = vision_inferencetest(model_name_or_path="google/bit-50", task="image-classification", mycache_dir=mycache_dir)

def MyVisionInferencetest(task="object-detection", mycache_dir=None):
    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    imagepath = './sampledata/bus.jpg'
    im0 = cv2.imread(imagepath)
    imgs = [im0]

    myinference = MyVisionInference(model_name="yolov8", model_path="/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt", task=task, model_type="torch", cache_dir=mycache_dir, gpuid='2', scale='n')
    confidences = myinference(imagepath)
    print(confidences)

if __name__ == "__main__":
    image_urls = ['http://example.com/image1.jpg', 'http://example.com/image2.jpg']
    my_inference = MyVisionInference(model_name='your_model_name', task='your_task')
    images = [my_inference.load_and_process_image(url) for url in image_urls if my_inference.load_and_process_image(url) is not None]
    if images:
        results = my_inference.batch_inference(images)
        print(results)
