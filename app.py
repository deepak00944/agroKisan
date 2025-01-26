from flask_mail import Mail, Message
import numpy as np
import pandas as pd
import pickle
import torch
from flask import Flask, flash, redirect, url_for, request, render_template, jsonify
from flask import jsonify
import requests
from bs4 import BeautifulSoup as bs
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import model as enet
import torch
from efficientnet_pytorch import model as enet
from torch import nn
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options



app = Flask(__name__)

app.config['SECRET_KEY'] = '262044xx'  # Change this to a random secret key

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'deepak.s.ashta@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'liujhsoaqrcrllnr'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'deepak.s.ashta@gmail.com'  # Replace with your email

mail = Mail(app)

def get_default_device():
    """Pick GPU if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_default_device()

def load_model():
    """Load and quantize the model dynamically for efficient memory usage."""
    model = enet.EfficientNet.from_name('efficientnet-b0', num_classes=6)
    # Load weights into the model
    model.load_state_dict(torch.load('checking.pth', map_location=device))
    # Apply dynamic quantization to reduce memory usage during inference
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


# Load other necessary data
data = pd.DataFrame(pd.read_csv("final.csv"))
model = pickle.load(open(r"RandomForest.pkl", "rb"))
area = pd.read_csv(r"final_data.csv")
commodity = pd.read_csv(r"commodities1.csv")

@app.route('/')
def index():
    return render_template('index.html')

# News
@app.route('/get-news', methods=['GET'])
def get_news():
    # Your existing Selenium code to scrape the website
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    
    url = 'https://agristack.gov.in/#/'
    driver.get(url)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.slide')))
    html_content = driver.page_source
    driver.quit()
    
    soup = bs(html_content, 'html.parser')
    slides = soup.find_all('div', class_='slide')
    
    extracted_data = []
    seen_images = set()
    base_url = 'https://agristack.gov.in'
    
    for slide in slides:
        label = slide.find('span', class_='agri-business-label')
        if label:
            label_text = label.get_text(strip=True)

            img_div = slide.find('div', class_='agri-gallery-section')
            img_url = ""
            if img_div:
                style_attr = img_div.get('style', '')
                img_url = style_attr.split('url("')[1].split('")')[0] if 'url(' in style_attr else ''
                if img_url and not img_url.startswith('http'):
                    img_url = base_url + '/' + img_url.lstrip('/')
                if img_url in seen_images:
                    continue
                seen_images.add(img_url)

            text = slide.find('div', class_='agri-gallery-text')
            text_content = text.get_text(strip=True) if text else ''

            extracted_data.append({
                'label': label_text,
                'image_url': img_url,
                'text': text_content
            })
    
    return jsonify(extracted_data)

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

def now_final(state_name, district_name, market, commodity_name, trend, Datefrom, DateTo):
    commodity_code = commodity[commodity['Commodities'] == commodity_name]['code'].unique()[0]
    state_short_name = area[area['State'] == state_name]['State_code'].unique()[0]
    district_code = area[area['District'] == district_name]['District_code'].unique()[0]
    market_code = area[area['Market'] == market]['Market_code'].unique()[0]
    date_from = Datefrom
    date_to = DateTo
    commodity_name = commodity
    state_full_name = state_name
    district_full_name = district_name
    market_full_name = market

    r = requests.get(f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodity_code}&Tx_State={state_short_name}&Tx_District={district_code}&Tx_Market={market_code}&DateFrom={date_from}&DateTo={date_to}&Fr_Date={date_from}&To_Date={date_to}&Tx_Trend={trend}&Tx_CommodityHead={commodity_name}&Tx_StateHead={state_full_name}&Tx_DistrictHead={district_full_name}&Tx_MarketHead={market_full_name}")
    soup = bs(r.text, "html.parser")
    title = soup.find("h4")

    tables = soup.find_all("table", class_="tableagmark_new")
    for tn in range(len(tables)):
        table = tables[tn]

        # preinit list of lists
        rows = table.findAll("tr")
        row_lengths = [len(r.findAll(['th', 'td'])) for r in rows]
        ncols = max(row_lengths)
        nrows = len(rows)
        data = []

        print(ncols, nrows)
        for i in range(nrows):
            rowD = []
            for j in range(ncols):
                rowD.append('')
            data.append(rowD)

        # process html
        for i in range(len(rows)):
            row = rows[i]
            cells = row.findAll(["td", "th"])
            j = 0  # Column index for data list

            if trend == "2":
                for cell in cells:
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    cell_text = cell.text.strip()

                    while data[i][j]:
                        j += 1

                    for r in range(rowspan):
                        for c in range(colspan):
                            data[i + r][j + c] = cell_text

                    j += colspan
            if trend == "0":
                if (i <= 50):
                    for cell in cells:
                        rowspan = int(cell.get('rowspan', 1))
                        colspan = int(cell.get('colspan', 1))
                        cell_text = cell.text.strip()

                        while data[i][j]:
                            j += 1

                        for r in range(rowspan):
                            for c in range(colspan):
                                data[i + r][j + c] = cell_text

                        j += colspan

        #     print(data)
        df = pd.DataFrame(data)
        df.columns = df.iloc[0]
        df = df[1:]
        if trend == "0":
            df = df.drop(df.index[-2:], axis=0)

        if df.empty:
            df.loc[0] = "No Data Found"
        if trend == "0":
            df.drop(columns={"Sl no."}, inplace=True)
        # df.to_csv("pta.csv",index=False)
        return (df, title.text)

@app.route('/market')
def market():
    state1 = sorted(area['State'].unique().astype(str))
    # print(states)  # Add this line to print the states

    area["District_state"] = area["State"] + "_" + area["District"]
    district1 = sorted(area['District_state'].unique().astype(str))
    # print(district1)
    area["market_district"] = area["State"]+"_" +area['District'] + "_" + area['Market']
    markets = sorted(area['market_district'].unique().astype(str))
    commodities= commodity["Commodities"].unique().astype(str)
    return render_template('market.html', states=state1, districts=district1, commodities=commodities, markets = markets)

@app.route('/price',methods=['POST'])
def price():
    state_name = request.form.get('state1')
    district_name = request.form.get('district1')
    market = request.form.get('Market')
    commodity_name = request.form.get('commodity')
    trend = request.form.get('Price/Arrival')
    Datefrom = request.form.get('from')
    DateTo = request.form.get('To')
    print(trend)
    final_data,heading=now_final(state_name,district_name,market,commodity_name,trend,Datefrom,DateTo)
    table_data = final_data.to_dict(orient='records')

    return jsonify(table_data)


label_dic = {
    0: 'healthy', 
    1: 'scab',
    2: 'rust',
    3: 'frog_eye_leaf_spot',
    4: 'complex', 
    5: 'powdery_mildew'
}

@app.route('/predict', methods=['POST'])
def predict():
    pH = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    temperature = float(request.form.get('temperature'))
    nitrogen = int(request.form.get('nitrogen'))
    phosphorus = int(request.form.get('phosphorus'))
    potassium = int(request.form.get('potassium'))
    humidity = float(request.form.get('humidity'))
    output = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]]),

    return ("{} Crop can be Grown".format(str(output[0])))

def about_disease(filtered_df, column):
    cause_values = filtered_df[column]
    f = ", ".join(cause_values)
    return f.rstrip(", ") # Remove trailing comma and spaces


def transform_valid():
    augmentation_pipeline = A.Compose(
        [
            A.SmallestMaxSize(224),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    )
    return lambda img: augmentation_pipeline(image=np.array(img))['image']



@app.route('/disease', methods=['GET'])
def disease():
    # Main page
    return render_template('disease.html')

@app.route('/disease_pred', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Open the image file directly in memory using BytesIO
        img = Image.open(BytesIO(f.read()))

        # Apply transformations (resize, crop, normalize)
        img = transform_valid()(img).unsqueeze(0)  # Add batch dimension
        img = img.to(device)

        # Load model dynamically for the prediction request
        model = load_model()

        # Make prediction with the loaded and quantized model
        with torch.no_grad():  # Disable gradient calculation
            output = model(img)
            predicted_indices = torch.argmax(output, dim=1)

        # Convert predicted numerical labels to string labels
        predicted_labels_str = [label_dic[label.item()] for label in predicted_indices]

        # Assuming you have a DataFrame named 'data' with relevant information
        filtered_df = data[data['Type'] == predicted_labels_str[0]]

        symptoms = about_disease(filtered_df, 'Symptoms')
        cause = about_disease(filtered_df, 'Cause')
        prevention = about_disease(filtered_df, 'Prevention')

        response = {
            'disease': predicted_labels_str[0],
            'cause': cause,
            'symptoms': symptoms,
            'prevention': prevention
        }

        # Clear model from memory after inference
        del model
        torch.cuda.empty_cache()  # Optional: Clear GPU cache if using GPU

        return jsonify(response)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        
        try:
            msg = Message(subject,
                          recipients=['deepak.s.ashta@gmail.com'])  # Replace with your email
            msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
            mail.send(msg)
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            flash('An error occurred while sending your message. Please try again later.', 'error')
        
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/rainfall', methods = ['GET', 'POST'])
def rainfall():
    return render_template('rainfall.html')

if __name__ == '__main__':
    app.run()