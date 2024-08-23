import time
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
   
model_id="openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id,clean_up_tokenization_spaces=True)

# if you have cuda set it to the active device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# to display CUDA device name
# print(torch.cuda.get_device_name(torch.cuda.current_device()))  
# move the model to the device
model.to(device)


def generate_text_embedding(phrase):
    label_tokens = processor(
        text=phrase,
        padding=True,
        images=None,
        return_tensors='pt'
    ).to(device)
    
    # encode tokens to sentence embeddings
    label_embeddings = model.get_text_features(**label_tokens)
    # detach from pytorch gradient computation 
    label_embeddings = label_embeddings.detach().cpu().numpy()
    return label_embeddings

def benchmark_text():
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "The sun rises in the east and sets in the west.",
        "He enjoys hiking in the mountains during the summer.",
        "Music can be a great source of relaxation and inspiration.",
        "Reading books expands your knowledge and imagination.",
        "The cat sat on the mat and stared at the window.",
        "She baked a delicious chocolate cake for the party.",
        "Traveling to new places broadens your horizons."

    ]
    # print(len(sentences))
    times=[]
    rounds=5
    for j in range(rounds):
        start_time = time.time()
        for i in sentences:
            generate_text_embedding(i)
        total_time = time.time()-start_time
        times.append(total_time)
    print("The program while running on",device,"took:"+str(sum(times)/rounds))

def benchmark_image():
    
    images = [
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg",
        "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp",
        "https://miro.medium.com/v2/resize:fit:800/1*n4NaE_6qCzUaKUPTNbfpcA.jpeg",
        "https://m.media-amazon.com/images/I/51KrxEKN58L.jpg"        
    ]
    times=[]
    rounds=10
    for j in range(rounds):
        start_time = time.time()
        for i in images:
            generate_image_embedding(i)
        total_time = time.time()-start_time
        times.append(total_time)
    print("The program while running on",device,"took:"+str(sum(times)/rounds))


def generate_image_embedding(img):
    url = "https://www.androidauthority.com/wp-content/uploads/2022/11/twitter-1-scaled-1000w-563h.jpg.webp"
    img  = Image.open(requests.get(url, stream=True).raw)
    image = processor(
        text=None,
        images=img,
        return_tensors='pt'
    ).to(device)['pixel_values']
    image_embeddings = model.get_image_features(image)
    print(image_embeddings  )
    return image_embeddings


benchmark_text()
# benchmark_image()

    