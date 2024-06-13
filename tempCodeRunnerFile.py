def about_disease(filtered_df, column):
#     cause_values = filtered_df[column]

#     f = ""
#     for cause in cause_values:
#         f = f + cause + ", "  # Separate multiple causes with a comma
#     return f.rstrip(", ")  # Remove trailing comma and spaces

# def load_prep(img_path):
#   img = tf.io.read_file(img_path)

#   img = tf.image.decode_image(img)

#   img = tf.image.resize(img,size=(224,224))

#   return img

# def model_predict(img_path, model2):
#     image=load_prep(img_path)
#     preds = model2.predict(tf.expand_dims(image,axis=0))
#     return preds


# @app.route('/disease', methods=['GET'])
# def disease():
#     # Main page
#     return render_template('disease.html')


# @app.route('/disease_pred', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model2)
#         print(preds)

#         # x = x.reshape([64, 64]);
#         disease_class = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Blueberry___healthy',
#  'Cherry_(including_sour)___Powdery_mildew',
#  'Cherry_(including_sour)___healthy',
#  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn_(maize)___Common_rust_',
#  'Corn_(maize)___Northern_Leaf_Blight',
#  'Corn_(maize)___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']

#         result = disease_class[preds.argmax()]

#         filtered_df = data[data['Type'] == result]

#         symptoms = about_disease(filtered_df, 'Symptoms')
#         cause = about_disease(filtered_df, 'Cause')
#         prevention = about_disease(filtered_df, 'Prevention')

#         response = {
#             'disease': result,
#             'cause': cause,
#             'symptoms': symptoms,
#             'prevention': prevention
#         }

#         return jsonify(response)