
import os, re, json
from pathlib import Path
from datetime import date, datetime

import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
import cv2
import easyocr
from sklearn.ensemble import RandomForestClassifier
from transformers import CLIPModel, CLIPProcessor

FRIDGE_FILE = Path("fridge_items.json")

FOOD_CLASSES = [
    "yogurt","milk","cheese","butter","cream","egg","chicken","beef","pork","fish","shrimp","tofu",
    "bread","rice","noodles","pasta","leftover cooked food","sandwich","juice","soft drink","bottled tea","sauce","jam",
    "apple","banana","orange","grapes","mango","pineapple","watermelon","kiwi","pear","lemon",
    "tomato","carrot","potato","onion","lettuce","cabbage","cucumber","spinach","bell pepper","eggplant","corn","garlic","ginger",
    "unknown food"
]

SHELF_LIFE = {
    "yogurt":10,"milk":7,"cheese":21,"butter":30,"cream":7,"egg":21,"chicken":2,"beef":3,"pork":3,"fish":2,"shrimp":2,"tofu":5,
    "bread":5,"rice":4,"noodles":4,"pasta":4,"leftover cooked food":3,"sandwich":2,"juice":7,"soft drink":30,"bottled tea":14,"sauce":30,"jam":30,
    "apple":21,"banana":5,"orange":21,"grapes":7,"mango":7,"pineapple":5,"watermelon":5,"kiwi":14,"pear":10,"lemon":21,
    "tomato":7,"carrot":21,"potato":30,"onion":30,"lettuce":5,"cabbage":14,"cucumber":7,"spinach":5,"bell pepper":10,"eggplant":7,"corn":5,"garlic":60,"ginger":30,
    "unknown food":5
}

EMOJI = {
    "apple":"🍎","banana":"🍌","orange":"🍊","lemon":"🍋","grapes":"🍇","watermelon":"🍉","yogurt":"🥣","milk":"🥛","cheese":"🧀",
    "egg":"🥚","bread":"🍞","chicken":"🍗","beef":"🥩","pork":"🥩","fish":"🐟","shrimp":"🍤","tofu":"⬜","rice":"🍚","noodles":"🍜",
    "tomato":"🍅","carrot":"🥕","potato":"🥔","onion":"🧅","lettuce":"🥬","cucumber":"🥒","corn":"🌽","juice":"🧃"
}

def norm(x):
    return str(x).lower().replace("_"," ").replace("-"," ").strip()

def emoji(food):
    food = norm(food)
    for k,v in EMOJI.items():
        if k in food or food in k:
            return v
    return "🍽️"

def shelf(food):
    food = norm(food)
    if food in SHELF_LIFE:
        return SHELF_LIFE[food]
    for k,v in SHELF_LIFE.items():
        if k in food or food in k:
            return v
    return 5

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def predict_food(image):
    if image is None:
        return "unknown food", 0.0
    prompts = [f"a photo of {f}" for f in FOOD_CLASSES]
    inputs = clip_processor(text=prompts, images=image.convert("RGB"), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
    i = int(np.argmax(probs))
    return FOOD_CLASSES[i], float(probs[i])

def parse_date(text):
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text.replace(".", "/").replace(",", " ")).strip()
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", clean)
    if match:
        y,m,d = map(int, match.groups())
        try:
            return date(y,m,d)
        except Exception:
            pass
    for a,b,y in re.findall(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", clean):
        a,b,y = int(a), int(b), int(y)
        for d,m in [(a,b),(b,a)]:
            try:
                return date(y,m,d)
            except Exception:
                pass
    words = clean.split()
    for i in range(len(words)):
        chunk = " ".join(words[i:i+3])
        for fmt in ["%d %b %Y","%d %B %Y","%b %d %Y","%B %d %Y","%d %b %y","%b %d %y"]:
            try:
                return datetime.strptime(chunk, fmt).date()
            except Exception:
                pass
    return None

def read_ocr(image):
    if image is None:
        return "", "No expiry-date image uploaded. Manual input was used.", None
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    text = " ".join(ocr_reader.readtext(gray, detail=0))
    dt = parse_date(text)
    if dt is None:
        return text, "OCR found text but no clear expiry date. Manual input was used.", None
    days = (dt - date.today()).days
    return text, f"Detected expiry date: {dt.isoformat()} | {days} day(s) left", days

def label_rule(food, days_stored, opened, days_left, storage):
    life = shelf(food)
    if storage == "freezer":
        life *= 3
    if storage == "room temperature":
        life *= 0.5
    if opened:
        life *= 0.65
    storage_ratio = days_stored / max(life, 1)
    expiry_score = 1.0 if days_left <= 0 else 0.85 if days_left <= 2 else 0.55 if days_left <= 5 else 0.2
    score = 0.65 * storage_ratio + 0.35 * expiry_score
    if norm(food) in ["chicken","beef","pork","fish","shrimp","tofu","milk","cream","leftover cooked food","sandwich"] and days_stored >= 2 and storage != "freezer":
        score += 0.15
    return "High Risk" if score >= 0.8 else "Medium Risk" if score >= 0.45 else "Low Risk"

def train_risk_model():
    np.random.seed(42)
    rows = []
    stores = ["fridge","freezer","room temperature"]
    for _ in range(6000):
        food = np.random.choice(FOOD_CLASSES)
        days = int(np.random.randint(0,31))
        opened = int(np.random.choice([0,1]))
        left = int(np.random.randint(-3,31))
        store = np.random.choice(stores, p=[0.75,0.1,0.15])
        rows.append({"food_type":food,"days_stored":days,"opened":opened,"days_until_expiry":left,"storage_type":store,
                     "risk":label_rule(food, days, opened, left, store)})
    df = pd.DataFrame(rows)
    X = pd.get_dummies(df.drop("risk", axis=1), columns=["food_type","storage_type"])
    y = df["risk"]
    model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, class_weight="balanced").fit(X,y)
    return model, X.columns

risk_model, risk_cols = train_risk_model()

def predict_risk(food, days, opened_status, days_left, storage):
    row = pd.DataFrame([{"food_type":norm(food),"days_stored":int(days),"opened":1 if opened_status=="Opened" else 0,
                         "days_until_expiry":int(days_left),"storage_type":storage}])
    X = pd.get_dummies(row, columns=["food_type","storage_type"]).reindex(columns=risk_cols, fill_value=0)
    risk = risk_model.predict(X)[0]
    return risk, dict(zip(risk_model.classes_, risk_model.predict_proba(X)[0]))

def reminder(food, risk, days_left, storage):
    if days_left < 0:
        return f"{food} is already expired. Please check it immediately."
    if risk == "High Risk":
        return f"{food} is high risk. Eat or use it as soon as possible."
    if risk == "Medium Risk":
        return f"{food} may expire soon. Plan to eat it within the next few days."
    return f"{food} is low risk. Keep it stored properly in the {storage}."

def load_fridge():
    if FRIDGE_FILE.exists():
        try:
            return json.loads(FRIDGE_FILE.read_text())
        except Exception:
            return []
    return []

def save_fridge(items):
    FRIDGE_FILE.write_text(json.dumps(items, indent=2))

def result_card(a):
    food, risk = a["food"], a["risk"]
    color = {"Low Risk":"#059669","Medium Risk":"#f59e0b","High Risk":"#dc2626"}.get(risk,"#6b7280")
    probs = ""
    for k in ["High Risk","Medium Risk","Low Risk"]:
        v = float(a["risk_probs"].get(k,0))
        probs += f"<div style='display:flex;gap:10px;align-items:center;margin:5px 0'><div style='width:100px'>{k}</div><div style='background:#e5e7eb;border-radius:8px;width:170px;height:16px'><div style='background:{color};width:{max(v*100,4):.0f}%;height:16px;border-radius:8px'></div></div><div>{v*100:.0f}%</div></div>"
    return f"""
    <div style="background:white;color:#111827;border-left:8px solid {color};border-radius:16px;padding:22px;max-width:520px">
      <div style="font-size:48px">{emoji(food)}</div><h2 style="margin:0">{food.title()}</h2>
      <p style="color:#6b7280">Recognized with {a['confidence']*100:.0f}% confidence</p>
      <span style="background:{color};color:white;padding:8px 14px;border-radius:20px;font-weight:bold">{risk}</span>
      <p>📅 {a['days_until_expiry']} day(s) until expiry</p>{probs}
      <p style="color:#6b7280">OCR read: {a['ocr_text'] or 'No OCR text'}</p>
      <div style="background:#ecfdf5;border-radius:12px;padding:14px;margin-top:12px">💡 <b>Recommendation:</b> {a['reminder_message']}</div>
    </div>"""

def render_fridge_html():
    items = load_fridge()
    if not items:
        return "<div style='background:#ecfdf5;border-radius:12px;padding:18px'>✅ Your fridge list is empty. Scan food to add reminders.</div>"
    high = sum(x.get("risk")=="High Risk" for x in items)
    soon = sum(int(x.get("days_until_expiry",99)) <= 2 for x in items)
    html = f"""<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px'>
    <div style='background:white;border-radius:12px;padding:18px;text-align:center'><h2>{len(items)}</h2><p>Items in fridge</p></div>
    <div style='background:white;border-radius:12px;padding:18px;text-align:center'><h2>{high}</h2><p>High risk</p></div>
    <div style='background:white;border-radius:12px;padding:18px;text-align:center'><h2>{soon}</h2><p>Expiring within 2 days</p></div></div>
    <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:14px'>"""
    for i, x in enumerate(items, 1):
        color = {"Low Risk":"#059669","Medium Risk":"#f59e0b","High Risk":"#dc2626"}.get(x.get("risk"),"#6b7280")
        html += f"""<div style='background:white;color:#111827;border-top:6px solid {color};border-radius:12px;padding:18px'>
        <div style='display:flex;justify-content:space-between'><div style='font-size:38px'>{emoji(x.get('food'))}</div><div style='color:#6b7280'>#{i}</div></div>
        <h3>{x.get('food','Unknown').title()}</h3><span style='background:{color};color:white;padding:6px 10px;border-radius:12px;font-weight:bold'>{x.get('risk')}</span>
        <p>📅 {x.get('days_until_expiry')} day(s) left</p><p>❄️ {x.get('storage_type')} · {x.get('opened_status')}</p><p>⏱️ stored {x.get('days_stored')} day(s)</p></div>"""
    return html + "</div>"

def analyze(food_image, expiry_image, days_stored, opened_status, manual_days_until_expiry, storage_type):
    food, conf = predict_food(food_image)
    ocr_text, ocr_msg, ocr_days = read_ocr(expiry_image)
    days_left = int(ocr_days) if ocr_days is not None else int(manual_days_until_expiry)
    risk, probs = predict_risk(food, days_stored, opened_status, days_left, storage_type)
    rem = reminder(food, risk, days_left, storage_type)
    a = {"food":food, "confidence":conf, "ocr_text":ocr_text, "ocr_message":ocr_msg, "days_stored":int(days_stored),
         "opened_status":opened_status, "days_until_expiry":days_left, "storage_type":storage_type, "risk":risk,
         "risk_probs":probs, "reminder_message":rem}
    return result_card(a), a

def add_to_fridge(last_analysis):
    if last_analysis is None:
        return "<div style='color:#dc2626'>Please analyze food first.</div>", render_fridge_html()
    items = load_fridge()
    items.append(last_analysis)
    save_fridge(items)
    return f"<div style='background:#ecfdf5;padding:14px;border-radius:12px'>✅ Added {emoji(last_analysis['food'])} {last_analysis['food']} to My Fridge.</div>", render_fridge_html()

def clear_all():
    save_fridge([])
    return render_fridge_html()

def remove_item(item_id):
    items = load_fridge()
    try:
        idx = int(item_id)-1
        if 0 <= idx < len(items):
            items.pop(idx)
            save_fridge(items)
    except Exception:
        pass
    return render_fridge_html()

LINE_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN","").strip()

def send_line_message(line_user_id, message):
    if not LINE_TOKEN:
        return "LINE token is missing. Add LINE_CHANNEL_ACCESS_TOKEN in Hugging Face Space Settings > Secrets."
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {LINE_TOKEN}"},
        json={"to":line_user_id.strip(),"messages":[{"type":"text","text":message}]}
    )
    return "LINE reminder sent successfully." if r.status_code == 200 else f"LINE reminder failed. Status: {r.status_code}. Response: {r.text}"

def send_latest_line(line_user_id, send_to_line, last_analysis):
    if not send_to_line:
        return "<b>LINE reminder was not sent.</b><br>Please check the box first."
    if not line_user_id or not line_user_id.strip():
        return "<b>Error:</b><br>Please enter your LINE User ID."
    data = last_analysis
    if data is None:
        items = load_fridge()
        data = items[-1] if items else None
    if data is None:
        return "<b>Error:</b><br>No reminder found. Please analyze food first."
    msg = f"""AI Fridge Reminder

Food: {data.get('food')}
Risk Level: {data.get('risk')}
Days Until Expiry: {data.get('days_until_expiry')}
Storage Type: {data.get('storage_type')}
Package Status: {data.get('opened_status')}

Reminder:
{data.get('reminder_message')}
"""
    return f"<b>LINE Status:</b><br>{send_line_message(line_user_id, msg)}"

css = """.gradio-container{font-family:'Segoe UI',system-ui,sans-serif}
#fridge-header{background:linear-gradient(135deg,#10b981,#059669);color:white;padding:24px 28px;border-radius:18px;margin-bottom:12px;box-shadow:0 8px 20px rgba(0,0,0,.12)}
#fridge-header h1{margin:0;font-size:28px;font-weight:800} #fridge-header p{margin:8px 0 0;opacity:.95;font-size:15px}
#info-box{background:#ecfdf5;border:1px solid #bbf7d0;padding:14px 18px;border-radius:14px;margin-bottom:12px}
#warning-box{background:#fff7ed;border:1px solid #fed7aa;padding:14px 18px;border-radius:14px;margin-bottom:12px}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=css, title="AI Smart Fridge") as demo:
    gr.HTML("""<div id="fridge-header"><h1>🧊 AI Smart Fridge</h1><p>Recognize food • read expiry dates • predict spoilage risk • create reminders • send LINE reminder</p></div>""")
    gr.HTML("""<div id="info-box"><b>Try the AI:</b> Upload a food photo and an expiry-date photo. The system will recognize the food, read the date, predict the risk, and create a reminder.</div>""")
    last_analysis = gr.State(None)
    with gr.Tabs():
        with gr.Tab("📷 Scan Food"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Upload Food Information")
                    food_in = gr.Image(type="pil", label="Food Photo", height=260)
                    expiry_in = gr.Image(type="pil", label="Expiry-Date Label Photo", height=220)
                    days_in = gr.Number(value=1, precision=0, label="Days Already Stored")
                    manual_in = gr.Number(value=5, precision=0, label="Days Until Expiry if OCR Fails")
                    opened_in = gr.Radio(["Unopened","Opened"], value="Unopened", label="Package Status")
                    storage_in = gr.Radio(["fridge","freezer","room temperature"], value="fridge", label="Storage Location")
                    analyze_btn = gr.Button("🔍 Analyze Food", variant="primary")
                with gr.Column():
                    gr.Markdown("### 2. AI Result")
                    result_out = gr.HTML()
                    add_btn = gr.Button("➕ Add this item to My Fridge")
                    add_msg = gr.HTML()
        with gr.Tab("💬 LINE Reminder"):
            gr.HTML("""<div id="warning-box"><b>LINE Reminder:</b> Add <code>LINE_CHANNEL_ACCESS_TOKEN</code> in Hugging Face Secrets, paste your LINE User ID, and check “Send reminder to LINE”.</div>""")
            line_user_id_in = gr.Textbox(label="LINE User ID", placeholder="Paste your LINE User ID here, starting with U...")
            send_line_checkbox = gr.Checkbox(label="Send reminder to LINE", value=False)
            line_btn = gr.Button("📩 Send Latest Reminder to LINE", variant="primary")
            line_status_out = gr.HTML()
        with gr.Tab("🧊 My Fridge"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Reminders")
                clear_btn = gr.Button("🗑️ Clear All Items")
            fridge_out = gr.HTML(render_fridge_html())
            remove_id = gr.Textbox(label="Remove Item by ID", placeholder="Example: 3")
            remove_btn = gr.Button("Remove Item")
        with gr.Tab("📱 Exhibition Demo"):
            gr.Markdown("""### How Visitors Can Try the AI
1. Upload a food photo.
2. Upload an expiry-date photo.
3. Click **Analyze Food**.
4. Open **LINE Reminder** if you want to send the reminder to LINE.

Recommended demo foods: yogurt, milk, eggs, bread, banana, tomato.""")
        with gr.Tab("ℹ️ How It Works"):
            gr.Markdown("""### How the System Works
1. **Upload Food Photo** — AI recognizes the food item.
2. **Upload Expiry-Date Photo** — OCR reads the expiry date. If OCR fails, manual input is used.
3. **Enter Storage Information** — days stored, package status, and storage location.
4. **AI Predicts Spoilage Risk** — Random Forest predicts Low, Medium, or High Risk.
5. **Reminder System** — the system creates a reminder and saves it into My Fridge.
6. **LINE Reminder** — the latest reminder can be sent to the user’s LINE account.""")

    analyze_btn.click(analyze, inputs=[food_in, expiry_in, days_in, opened_in, manual_in, storage_in], outputs=[result_out, last_analysis])
    add_btn.click(add_to_fridge, inputs=[last_analysis], outputs=[add_msg, fridge_out])
    refresh_btn.click(lambda: render_fridge_html(), outputs=fridge_out)
    clear_btn.click(clear_all, outputs=fridge_out)
    remove_btn.click(remove_item, inputs=remove_id, outputs=fridge_out)
    line_btn.click(send_latest_line, inputs=[line_user_id_in, send_line_checkbox, last_analysis], outputs=line_status_out)

demo.launch()
