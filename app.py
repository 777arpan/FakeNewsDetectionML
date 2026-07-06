import os, re, sys, joblib, numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response, render_template

app = Flask(__name__, static_folder='../frontend', static_url_path='')
@app.route("/")
def home():
    return render_template("index.html")

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fake_news_model.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model not found. Run: python model/train_model.py")
    sys.exit(1)



def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s!?.,]', '', text)
    return text.strip()

def extract_signals(text):
    t = text.lower()
    sensational = ['shocking','breaking','urgent','exposed','secret','hidden','bombshell',
                   'leaked','proof','hoax','lie','conspiracy','banned','censored','suppressed',
                   'share before deleted','they dont want','wake up','exclusive','miracle',
                   'cure','hate this','whistleblower']
    credible = ['according to','study','research','published','university','journal','percent',
                'report','official','announced','data','statistics','findings','peer-reviewed',
                'confirmed','analysis','survey','trial']
    emotional = ['shocking','outrage','furious','terrifying','disgusting','hate','fear',
                 'danger','evil','corrupt','betrayal','betrayed','exposed']
    words = t.split()
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamations = text.count('!')
    s_hits = [w for w in sensational if w in t]
    c_hits = [w for w in credible if w in t]
    e_hits = [w for w in emotional if w in t]
    emotional_score  = min(100, int(caps_ratio*200 + exclamations*15 + len(s_hits)*12 + len(e_hits)*10))
    factual_score    = min(100, max(5, int(len(c_hits)*18 + len(words)/3)))
    source_score     = min(100, max(5, int(len(c_hits)*20 - len(s_hits)*12 + 40)))
    logic_score      = min(100, max(5, 70 - len(s_hits)*8 + len(c_hits)*6))
    red_flags, green_flags = [], []
    if caps_ratio > 0.1:
        red_flags.append(f"Excessive capitalization ({int(caps_ratio*100)}%) — common in sensational content")
    if exclamations >= 2:
        red_flags.append(f"Multiple exclamation marks ({exclamations}) — indicates emotional manipulation")
    if s_hits:
        red_flags.append(f"Sensational language detected: {', '.join(s_hits[:3])}")
    if 'share' in t and ('delete' in t or 'banned' in t):
        red_flags.append("Urgency to share before deletion — classic misinformation tactic")
    if any(w in t for w in ['secret','hidden','suppressed']):
        red_flags.append("Claims of suppressed information without verifiable sourcing")
    if c_hits:
        green_flags.append(f"Credible language: {', '.join(c_hits[:3])}")
    if any(w in t for w in ['percent','%','statistics','data']):
        green_flags.append("Contains specific numerical data or statistics")
    if any(w in t for w in ['according to','announced','confirmed']):
        green_flags.append("References attributed sources or official announcements")
    if len(words) > 30:
        green_flags.append("Adequate article length with contextual detail")
    return {
        "signals": {"emotional_manipulation": emotional_score, "factual_specificity": factual_score,
                    "source_credibility": source_score, "logical_coherence": logic_score},
        "red_flags": red_flags, "green_flags": green_flags
    }

def extract_topics(text):
    topics_map = {
        "Health":      ['cancer','vaccine','drug','cure','disease','virus','medical','doctor','health','fda'],
        "Politics":    ['government','election','president','senate','congress','vote','policy','law'],
        "Science":     ['nasa','study','research','scientist','discovery','space','climate','physics'],
        "Economy":     ['economy','market','stock','bank','dollar','inflation','gdp','trade','tax'],
        "Technology":  ['5g','ai','tech','internet','software','data','microchip','surveillance'],
        "Conspiracy":  ['deep state','new world order','globalist','illuminati','chemtrail','mind control'],
        "Environment": ['climate','environment','carbon','pollution','green','energy','weather'],
    }
    t = text.lower()
    found = [topic for topic, kw in topics_map.items() if any(k in t for k in kw)]
    return found[:4] if found else ["General"]

def get_verdict(prob_real):
    prob_fake = 1 - prob_real
    if prob_fake >= 0.70:
        return {
    "verdict": "FAKE",
    "icon": "summary",
    "summary": "Likely Fake News",
    "confidence": int(prob_fake * 100),
    "credibility_score": int(prob_real * 100),
    "explanation_prefix": "This content shows strong indicators of misinformation."
}
    elif prob_real >= 0.70:
        return {
    "verdict": "REAL",
    "icon": "summary",
    "summary": "Appears Credible",
    "confidence": int(prob_real * 100),
    "credibility_score": int(prob_real * 100),
    "explanation_prefix": "This content shows characteristics of credible journalism."
}
    else:
        return {
    "verdict": "UNCERTAIN",
    "icon": "summary",
    "summary": "Uncertain | Verify",
    "confidence": int(max(prob_real, prob_fake) * 100),
    "credibility_score": int(prob_real * 100),
    "explanation_prefix": "This content shows mixed signals and requires independent verification."
}

def build_explanation(text, verdict_data, signals_data):
    prefix = verdict_data["explanation_prefix"]
    v = verdict_data["verdict"]
    s = signals_data["signals"]
    t = text.lower()
    if v == "FAKE":
        reasons = []
        if s["emotional_manipulation"] > 50: reasons.append("heavy use of emotional and sensational language")
        if s["source_credibility"] < 40: reasons.append("lack of verifiable sources or citations")
        if s["factual_specificity"] < 40: reasons.append("vague or unsubstantiated claims")
        reason_str = ", ".join(reasons) if reasons else "multiple misinformation indicators"
        return f"{prefix} The model detected {reason_str}. Always cross-reference with established news organizations before sharing."
    elif v == "REAL":
        reasons = []
        if s["factual_specificity"] > 50: reasons.append("specific factual details and data points")
        if s["source_credibility"] > 50: reasons.append("credible sourcing language")
        if s["logical_coherence"] > 60: reasons.append("logical and measured tone")
        reason_str = ", ".join(reasons) if reasons else "credible writing patterns"
        return f"{prefix} The model identified {reason_str} consistent with reliable journalism. However, always verify independently."
    else:
        return f"{prefix} The model found mixed credible and questionable signals. Consult multiple independent sources before accepting or sharing."

def get_recommendation(verdict):
    recs = {
        "FAKE": "Do not share this content — verify with at least 3 independent credible news sources before drawing conclusions.",
        "REAL": "Content appears credible, but always cross-reference with original sources before sharing.",
        "UNCERTAIN": "Exercise caution — check reputable fact-checking sites like Snopes, PolitiFact, or FactCheck.org.",
    }
    return recs.get(verdict, recs["UNCERTAIN"])


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return Response(status=200)
    data = request.get_json(force=True)
    headline = data.get('headline', '').strip()
    article  = data.get('article', '').strip()
    if not article:
        return jsonify({"error": "Article text is required."}), 400
    full_text = f"{headline}. {article}" if headline else article
    processed = preprocess(full_text)
    proba = model.predict_proba([processed])[0]
    classes = list(model.classes_)
    prob_real = proba[classes.index(1)] if 1 in classes else proba[1]
    verdict_data  = get_verdict(prob_real)
    signals_data  = extract_signals(full_text)
    topics        = extract_topics(full_text)
    explanation   = build_explanation(full_text, verdict_data, signals_data)
    recommendation = get_recommendation(verdict_data["verdict"])
    return jsonify({
        "verdict": verdict_data["verdict"], "icon": verdict_data["icon"],
        "summary": verdict_data["summary"], "confidence": verdict_data["confidence"],
        "credibility_score": verdict_data["credibility_score"],
        "prob_fake": round((1-prob_real)*100, 1), "prob_real": round(prob_real*100, 1),
        "signals": signals_data["signals"], "red_flags": signals_data["red_flags"],
        "green_flags": signals_data["green_flags"], "topics": topics,
        "explanation": explanation, "recommendation": recommendation,
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + Logistic Regression", "version": "1.0"})

@app.route('/api/samples', methods=['GET'])


#Sample Articles for testing
def get_samples():
    return jsonify({
        "fake1": {
            "headline": "BREAKING: Scientists PROVE that drinking bleach cures cancer in 24 hours!!",
            "article": "A whistleblower who CANNOT BE NAMED has come forward with SHOCKING evidence that Big Pharma has been suppressing a 100% cure for cancer for decades. Share this before it gets DELETED! The mainstream media refuses to cover this bombshell truth."
        },
        "fake2": {
            "headline": "Government hiding 5G mind-control frequencies in new towers nationwide",
            "article": "Secret documents LEAKED online prove that 5G towers installed across the country contain hidden frequencies that alter brain chemistry and make citizens obedient. Deep state operatives have silenced scientists who tried to warn us. WAKE UP and share the truth before they ban this post!"
        },
        "real1": {
            "headline": "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
            "article": "The Federal Reserve announced Wednesday it would raise its benchmark interest rate by a quarter of a percentage point, bringing the federal funds rate to a target range of 5.25% to 5.5%. Fed Chair Jerome Powell stated the decision reflects the committee ongoing assessment of incoming economic data and evolving inflation conditions."
        },
        "real2": {
            "headline": "NASA confirms water ice deposits at Moon south pole",
            "article": "Data from NASA Lunar Reconnaissance Orbiter has confirmed the presence of water ice in permanently shadowed craters near the Moon south pole, according to a peer-reviewed study published in Nature Geoscience. Researchers analyzed surface reflectance data collected over three years and identified ice deposits spanning approximately 12,000 square kilometers."
        }
    })

if __name__ == '__main__':
    print("\nNewsGuard ML API running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
