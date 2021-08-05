import argparse
import json

import numpy as np
import requests
import tensorflow
import PIL
from tensorflow.keras.preprocessing import image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
ap.add_argument("-u", "--uri", required=True,
                help="URI of model server")

args = vars(ap.parse_args())

image_path = args['image']
uri = args['uri']

img = image.img_to_array(image.load_img(image_path, target_size=(128, 128))) / 255.

payload = {
    "instances": [{'conv2d_3_input': img.tolist()}]
}

r = requests.post(uri+'/v1/models/dogs-vs-cats:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))
predict=np.asarray(pred['predictions']).argmax(axis=1)[0]
print( "Dog" if predict==1 else "Cat" )
