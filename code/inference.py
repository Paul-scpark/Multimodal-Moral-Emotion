# examples for use
# python inference.py -l korean -g 0 -e other_condemning -s test -d test_data.parquet
# python inference.py -l english -g 0 -e other_praising -s test -d test_data.parquet

import os
import io
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

# internal module
from code.utils import *
from model import MoralEmotionVLClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        required=True,
        help='type only korean or english'
    )

    parser.add_argument(
        '-g',
        '--gpu',
        type=str,
        default='auto',
        help='type only number(e.g., cuda:0 -> 0)'
    )

    parser.add_argument(
        '-e',
        '--emotion',
        type=str,
        required=True,
        help='type the target moral emotion(e.g., other_condemning, other_praising, other_suffering, self_conscious, non_moral_emotion, neutral)'
    )

    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        required=True,
        help='type the directory path of save results'
    )

    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        required=True,
        help='type the data path for inference input'
    )

    return parser.parse_args()

def bytes_to_image(byte_data):
    return Image.open(io.BytesIO(byte_data['bytes']))

def predict(model, test_dataloader, threshold):
    results = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            ids = batch.pop('ids')
            outputs = model(**batch)
            logits = outputs.logits
            
            probabilities = torch.sigmoid(logits)
            predicted_class = (probabilities > threshold).long()
            
            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "prediction": model.id2label[predicted_class[i].item()],
                    "probabilities": probabilities[i].item()
                })

    return pd.DataFrame(results)

if __name__ == "__main__":
    args = parse_args()
    base_model_name = 'Qwen/Qwen2-VL-7B-Instruct'
    device = args.gpu if args.gpu == "auto" else "cuda:" + args.gpu
    moral_emotion = args.emotion
    data_path = args.data_path
    language = args.language
    save_dir = os.path.join(args.save_dir, language)
    korean = True if language == 'korean' else False
    
    moral_emotion_model_name = KOREAN_MODEL_REPO_MAPPING[moral_emotion] if korean else ENGLISH_MODEL_REPO_MAPPING[moral_emotion]

    model_path = hf_hub_download(
        repo_id=moral_emotion_model_name,
        filename="model.pth"
    )

    save_path = os.path.join(save_dir, moral_emotion + '_results.parquet')
    threshold = KOREAN_MORAL_EMOTION_THRESHOLDDS[moral_emotion] if korean else ENGLISH_MORAL_EMOTION_THRESHOLDDS[moral_emotion]

    os.makedirs(save_dir, exist_ok=True)

    test_dataset = pd.read_parquet(data_path)

    # make byte format to PIL Image
    test_dataset["thumbnail"] = test_dataset["thumbnail"].apply(bytes_to_image)

    binary_labels = [
        "False",
        "True"
    ]

    model = MoralEmotionVLClassifier(
        base_model_name, 
        num_labels=1,
        device=device,
        label_names=binary_labels
    )

    model.to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)

    processor = AutoProcessor.from_pretrained(base_model_name)
    
    # if english -> korean=False
    formatted_test_dataset = [
        format_data(sample, moral_emotion, korean=korean) for sample in test_dataset.itertuples(index=False)]

    test_dataloader = DataLoader(
        formatted_test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, processor, device),
    )

    model.eval()
    outputs = predict(model, test_dataloader, threshold)
    outputs.to_parquet(save_path, index=False)