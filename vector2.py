import argparse, json, os, requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import numpy as np
import faiss, torch, open_clip

def main(input_json, out_index, out_meta):
    with open(input_json, 'r', encoding='utf-8') as f:
        images = json.load(f)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    dim = model.visual.output_dim
    index = faiss.IndexFlatL2(dim)
    meta = []
    os.makedirs('images', exist_ok=True)

    for item in tqdm(images, desc="Embedding"):
        name, url = item['name'], item['url']
        path = os.path.join('images', name)
        if not os.path.exists(path):
            r = requests.get(url, timeout=10)
            Image.open(BytesIO(r.content)).convert('RGB').save(path)
        img = Image.open(path).convert('RGB')
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(x).cpu().numpy()
        faiss.normalize_L2(feat)
        index.add(feat)
        meta.append(item)

    faiss.write_index(index, out_index)
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Wrote", out_index, out_meta)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='drive_images.json')
    p.add_argument('--out-index', default='image_index.faiss')
    p.add_argument('--out-meta', default='image_metadata.json')
    args = p.parse_args()
    main(args.input, args.out_index, args.out_meta)
