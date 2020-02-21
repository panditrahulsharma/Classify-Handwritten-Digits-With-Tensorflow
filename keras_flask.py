# Import flask which will handle our communication with the frontend
# Also import few other libraries

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import numpy as np
import re
import sys
import base64
import os
import pickle
import tensorflow as tf
from keras.models import load_model
# Path to our saved model
# sys.path.append(os.path.abspath("./model"))


# Initialize flask app
app = Flask(__name__)

#Initialize some global variables
#global graph
#model, graph = init()
#graph = tf.get_default_graph()
#model = load_model('static/model/model.h5')


def convertImage(imgData1):
	 imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
	 with open('output.png', 'wb') as output:
		   output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
 return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
	 # Predict method is called when we push the 'Predict' button 
	 # on the webpage. We will feed the user drawn image to the model
	 # perform inference, and return the classification
	 digit_detect_pkl = open('static/model/digit_predict.pkl', 'rb')
	 model = pickle.load(digit_detect_pkl)
	 
	 imgData = request.get_data()
	 convertImage(imgData)
	 # read the image into memory
	 #x = imread('output.png', mode='L')
	 image = Image.open("output.png")

	 # make it the right size
	 image_file = image.convert('L')
	 new_image = image_file.resize((28,28))
	 image_file.save('result.png')
	 #x = imresize(x, (28, 28))/255
	 #You can save the image (optional) to view later
	 #imsave('final_image.jpg', x)
	 #x = x.reshape(1, 28, 28, 1)
	 x = np.expand_dims(new_image, axis=0)
	 x=np.reshape(x,(1,28,28,1))
	 #with graph.as_default():
	 out = model.predict(x)
	 response = np.argmax(out, axis=1)
	 
	 return str(response)

if __name__ == "__main__":
# run the app locally on the given port
	app.run(host='127.0.0.1', port=5000,debug=True)


