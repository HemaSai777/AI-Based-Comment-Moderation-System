import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_NAME = "unitary/toxic-bert"
TOXIC_THRESHOLD = 0.6
ABUSIVE_THRESHOLD = 0.6
HATE_THRESHOLD = 0.6

# ---------------- LOAD MODEL (once) ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

MODEL_LABELS = model.config.id2label

# ---------------- YOUTUBE ----------------
def build_youtube(api_key):
    return build("youtube", "v3", developerKey=api_key)

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]

    return None

def fetch_youtube_comments(youtube, video_url, max_comments=20):
    video_id = extract_video_id(video_url)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments[:max_comments]

# ---------------- CLASSIFICATION ----------------
def classify_comment(comment):
    inputs = tokenizer(
        comment,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0]

    scores = dict(zip(MODEL_LABELS.values(), probs.tolist()))

    if scores["identity_hate"] > HATE_THRESHOLD or scores["threat"] > HATE_THRESHOLD:
        label = "Hate"
        confidence = max(scores["identity_hate"], scores["threat"])

    elif scores["insult"] > ABUSIVE_THRESHOLD or scores["obscene"] > ABUSIVE_THRESHOLD:
        label = "Abusive"
        confidence = max(scores["insult"], scores["obscene"])

    elif scores["toxic"] > TOXIC_THRESHOLD:
        label = "Toxic"
        confidence = scores["toxic"]

    else:
        label = "Safe"
        confidence = max(scores.values())

    return label, round(confidence, 2)

def suggest_action(label):
    if label == "Hate":
        return "Report"
    elif label == "Abusive":
        return "Hide"
    elif label == "Toxic":
        return "Review"
    else:
        return "Allow"

def process_comments(youtube, url):
    comments = fetch_youtube_comments(youtube, url)

    results = []

    for comment in comments:
        label, confidence = classify_comment(comment)
        action = suggest_action(label)

        results.append({
            "Comment": comment,
            "Label": label,
            "Confidence": confidence,
            "Suggested Action": action
        })

    return pd.DataFrame(results)
