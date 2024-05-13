import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define class names with descriptions
class_names = {
    0: {'name': 'Apple__Apple_scab', 'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit. Treat with copper-based fungicides or sulfur-based fungicide'},
    1: {'name': 'Apple___Black_rot', 'description': 'Fungal disease causing dark, sunken lesions on fruit. Treat with copper-based fungicides or mancozeb'},
    2: {'name': 'Apple___Cedar_apple_rust', 'description': 'Fungal disease causing orange spots on leaves. Treat with copper-based fungicides or sulfur-based fungicide'},
    3: {'name': 'Apple___healthy', 'description': 'No specific disease, maintain tree health through proper irrigation and fertilization'},
    4: {'name': 'Blueberry___healthy', 'description': 'No specific disease, maintain plant health through proper irrigation and fertilization'},
    5: {'name': 'Cherry_(including_sour)___Powdery_mildew', 'description': 'Fungal disease causing white powdery growth on leaves. Treat with sulfur-based fungicides or neem oil'},
    6: {'name': 'Cherry_(including_sour)___healthy', 'description': 'No specific disease, maintain plant health through proper irrigation and fertilization'},
    7: {'name': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'description': 'Fungal disease causing gray spots with brown margins on leaves. Treat with chlorothalonil or azoxystrobin'},
    8: {'name': 'Corn_(maize)___Common_rust_', 'description': ' Fungal disease causing reddish-brown pustules on leaves. Treat with mancozeb or propiconazole'},
    9: {'name': 'Corn_(maize)___Northern_Leaf_Blight', 'description': 'Fungal disease causing cigar-shaped lesions on leaves. Treat with chlorothalonil or azoxystrobin'},
    10: {'name': 'Corn_(maize)___healthy', 'description': 'No specific disease, maintain plant health through crop rotation and balanced fertilization'},
    11: {'name': 'Orange___Haunglongbing_(Citrus_greening)', 'description': 'Bacterial disease causing yellowing and mottling of leaves. Treat with copper-based compounds or tetracycline antibiotics'},
    12: {'name': 'Peach___Bacterial_spot', 'description': 'Bacterial disease causing dark spots on leaves and fruit. Treat with copper-based bactericides or streptomycin'},
    13: {'name': 'Peach___healthy', 'description': 'No specific disease, maintain tree health through regular pruning and sanitation'},
    14: {'name': 'Potato___Early_blight', 'description': ' A fungal disease on potato leaves, identified by dark lesions with yellow halos; treat preventively with fungicides or choose resistant cultivars'},
    15: {'name': 'Potato___Late_blight', 'description': 'Fungal disease causing dark lesions on leaves and stems. Treat with copper-based fungicides or chlorothalonil.'},
    16: {'name': 'Potato___healthy', 'description': ' No specific disease, maintain plant health through crop rotation and sanitation'},
    17: {'name': 'Strawberry___Leaf_scorch', 'description': 'Fungal disease causing brown spots on leaves. Treat with copper-based fungicides or chlorothalonil'},
    18: {'name': 'Strawberry___healthy', 'description': 'No specific disease, maintain plant health through weed control and proper irrigation'},
    19: {'name': 'Tomato___Bacterial_spot', 'description': 'Bacterial disease causing dark spots with yellow halos on leaves. Treat with copper-based bactericides or streptomycin'},
    20: {'name': 'Tomato___Early_blight', 'description': 'Fungal disease causing dark concentric rings on leaves. Treat with copper-based fungicides or chlorothalonil.'},
    21: {'name': 'Tomato___Late_blight', 'description': 'Fungal disease causing water-soaked lesions on leaves and fruit. Treat with copper-based fungicides or chlorothalonil'},
    22: {'name': 'Tomato___Leaf_Mold', 'description': 'Fungal disease causing yellowing and browning of leaves. Treat with chlorothalonil or mancozeb'},
    23: {'name': 'Tomato___Septoria_leaf_spot', 'description': ' Fungal disease causing small, dark spots on leaves. Treat with copper-based fungicides or chlorothalonil'},
    24: {'name': 'Tomato___Spider_mites Two-spotted_spider_mite', 'description': 'These tiny arachnids feed on tomato leaves, causing stippling and yellowing; control with horticultural oils or insecticidal soaps'},
    25: {'name': 'Tomato___Target_Spot', 'description': 'Fungal disease causing dark concentric rings on leaves. Treat with copper-based fungicides or chlorothalonil'},
    26: {'name': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'description': 'Viral disease causing yellowing and curling of leaves. Control whiteflies with insecticides and practice cultural methods to reduce virus spread'},
    27: {'name': 'Tomato___Tomato_mosaic_virus', 'description': 'Viral disease causing mottling and distortion of leaves. Control aphids and other vectors, remove infected plants'},
    28: {'name': 'Tomato___healthy', 'description': 'No specific disease, maintain plant health through crop rotation and sanitation'},
    29: {'name': 'Unknown', 'description': 'Unknown disease'}
    # Add descriptions for other classes as needed
}

# Define the list of students with their name and roll number
students = [
    {"name": "R.Monika ", "roll_number": "2038010039"},
    {"name": "P.Gayathri ", "roll_number": "2038010015"},
    {"name": "S.Kowsalya ", "roll_number": "2038010030"}   
]

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model('C:/Users/Keran Anne/Downloads/Moni_Project/Moni_Project/my_model.h5')

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part', students=students)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file', students=students)
        if file and allowed_file(file.filename):
            try:
                # Read the image file
                img_bytes = file.read()
                img = image.load_img(io.BytesIO(img_bytes), target_size=(48, 48))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalize image
                # Make prediction
                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                if class_index in class_names:
                    predicted_class = class_names[class_index]
                    predicted_class_name = predicted_class['name']
                    predicted_class_description = predicted_class['description']
                else:
                    predicted_class_name = 'Unknown'
                    predicted_class_description = 'Description not available'
                return render_template('index.html', message='Prediction: {}'.format(predicted_class_name),
                                       description=predicted_class_description, filename=None, students=students)
                if class_index in class_names:
                    predicted_class_name = class_names[class_index]
                else:
                    predicted_class_name = 'Unknown'
                return render_template('index.html', message='Prediction: {}'.format(predicted_class_name), filename=None, students=students)
            except Exception as e:
                # Handle prediction error
                return render_template('index.html', message='Prediction error: {}'.format(str(e)), filename=None, students=students)
        else:
            return render_template('index.html', message='Invalid file type', filename=None, students=students)
    return render_template('index.html', students=students)
    

if __name__ == "__main__":
    app.run(debug=True)
