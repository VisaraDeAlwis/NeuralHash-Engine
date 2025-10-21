# neuralhash_utils.py

from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image
from torchvision.transforms import ToPILImage

# Initialize MTCNN to keep all detected faces so we can select the largest
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Helper to convert tensor to a savable PIL Image
to_pil = ToPILImage()

def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']


def load_hyperplanes(path):
    arr = np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]
    return arr


def generate_neuralhash(image_path, pca, hyperplanes, cropped_image_save_path=None):
    """
    Generates a NeuralHash for the largest face in an image and optionally saves
    the cropped face.
    """
    img = Image.open(image_path).convert('RGB')
    
    # Detect all faces to find their bounding boxes
    boxes, _ = mtcnn.detect(img)
    
    # Get a list of face tensors
    face_tensors = mtcnn(img)

    if face_tensors is None:
        raise ValueError('No face detected')

    # If multiple faces are detected, select the one with the largest bounding box
    if len(face_tensors) > 1:
        box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_box_index = np.argmax(box_areas)
        face_tensor = face_tensors[largest_box_index]
    else:
        # If only one face is detected
        face_tensor = face_tensors[0]

    # --- New: Save the cropped face if a path is provided ---
    if cropped_image_save_path:
        # The tensor values are in the range [-1, 1]. Convert them to [0, 1]
        # and then to a PIL Image that can be saved.
        face_image = to_pil((face_tensor + 1) / 2)
        face_image.save(cropped_image_save_path)
    # --------------------------------------------------------

    emb_512 = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().squeeze()
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits


def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0,len(bits),8))

def calculate_hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))  # ensure Python int
