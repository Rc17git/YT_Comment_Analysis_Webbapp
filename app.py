from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import json

app = Flask(__name__, template_folder='')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=chrome_options)

    try:
        video_url = request.form.get("video_url")
        scroll_count = int(request.form.get("scroll_count"))  # Convert to integer
        if not video_url or not scroll_count:
            return jsonify({"error": "Video URL and scroll count are required."})
        driver.get(video_url)

        for _ in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        comments = soup.find_all('yt-formatted-string', {'id': 'content-text'})

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        driver.quit()

    test_list = [comment.get_text(strip=True) for comment in comments]

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model("SentModel1.h5")

    max_sequence_length = 80
    seq = tokenizer.texts_to_sequences(test_list)
    ans = pad_sequences(seq, maxlen=max_sequence_length)
    preds = model.predict(ans)

    count_0 = 0
    count_1 = 0

    for i in preds:
        if np.around(i, decimals=0).argmax() == 0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1

    percentage_0 = (count_0 / len(test_list)) * 100
    percentage_1 = (count_1 / len(test_list)) * 100
    data_per = np.array([percentage_0, percentage_1])
    labels = ["Negative {:.2f}%".format(percentage_0), "Positive {:.2f}%".format(percentage_1)]
    data_per = data_per.tolist()

    return render_template("results.html", data_per=data_per, labels=labels)

if __name__ == "__main__":
    app.run(debug=True)
