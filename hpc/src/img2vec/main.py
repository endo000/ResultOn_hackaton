__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np

if __name__ == "__main__":
    # Initialize Img2Vec with GPU
    img2vec = Img2Vec(cuda=True, model="efficientnet_b7")

    # Read in an image (rgb format)
    img = Image.open("/exec/cat.jpg")
    # Get a vector from img2vec, returned as a torch FloatTensor
    vec = img2vec.get_vec(img)
    client = chromadb.HttpClient(
        host="86.58.82.202",
        port=8000,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
            chroma_client_auth_credentials="hackathon:ieh2Ohfade",
        )
    )

    print(client.heartbeat())
    print(client.get_version())
    print(client.list_collections())

    collection = client.get_or_create_collection(name="hackathon")
    collection.add(
        embeddings=[vec.tolist()],
        ids=["id1"],
    )

    while True:
        data = input("Please enter the message:\n")
        if "Exit" == data:
            break

        exec(data)
