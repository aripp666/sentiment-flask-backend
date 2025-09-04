from flask import Flask, request, jsonify
from flask_cors import CORS
from GoogleNews import GoogleNews
from transformers import pipeline
from datetime import datetime, timedelta
import re

app = Flask(__name__)
CORS(app)

# Fungsi konversi tanggal
def convert_to_date(date_str):
    today = datetime.today()
    date_str = (date_str or "").lower()
    if "jam" in date_str:
        match = re.search(r'(\d+)\s*jam', date_str)
        hours = int(match.group(1)) if match else 1
        converted = today - timedelta(hours=hours)
    elif "hari" in date_str:
        match = re.search(r'(\d+)\s*hari', date_str)
        days = int(match.group(1)) if match else 1
        converted = today - timedelta(days=days)
    elif "minggu" in date_str:
        match = re.search(r'(\d+)\s*minggu', date_str)
        weeks = int(match.group(1)) if match else 1
        converted = today - timedelta(weeks=weeks)
    else:
        try:
            converted = datetime.strptime(date_str, "%d/%m/%Y")
        except:
            converted = today
    return converted.strftime("%d/%m/%Y")

# Load model sekali di awal
classifier = pipeline(
    "sentiment-analysis",
    model="mdhugol/indonesia-bert-sentiment-classification"
)
label_map = {"LABEL_0": "Positif", "LABEL_1": "Netral", "LABEL_2": "Negatif"}

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    keywords = data.get("keywords", ["Bank Riau Kepri"])
    start = data.get("start", "08/01/2025")
    end = data.get("end", "08/31/2025")

    all_results = []
    for keyword in keywords:
        googlenews = GoogleNews(lang="id", start=start, end=end)
        googlenews.search(keyword)
        results = googlenews.result()
        all_results.extend(results)

    # Hapus duplikasi
    seen_titles, unique_results = set(), []
    for r in all_results:
        if r.get("title") not in seen_titles:
            unique_results.append(r)
            seen_titles.add(r.get("title"))

    berita_final = []
    for news in unique_results:
        title = news.get("title", "")
        link = news.get("link", "")
        raw_date = news.get("date", "")
        date = convert_to_date(raw_date)
        try:
            result = classifier(title)[0]
            label = label_map.get(result["label"], "Netral")
            score = round(result["score"], 2)
            if label == "Netral":
                if any(x in title.lower() for x in ["gagal", "utang", "tak lolos", "belum layak", "gugur"]):
                    sentimen = "Negatif"
                else:
                    sentimen = "Positif"
            else:
                sentimen = label

            berita_final.append({
                "Tanggal": date,
                "Judul": title,
                "Link": link,
                "Skor": score,
                "Sentimen": sentimen
            })
        except Exception as e:
            print(f"Gagal analisis: {title}, error {e}")

    return jsonify({"jumlah": len(berita_final), "berita": berita_final})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
