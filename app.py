from flask import Flask, render_template, request, redirect, flash, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os, json, zipfile
from sqlalchemy.exc import IntegrityError 
from io import BytesIO
import pandas as pd
from datetime import datetime
from utils.pipeline import MalayalamTranscriptionPipeline
from utils.translation_utils import translate_to_malayalam
from utils.nlp_utils import analyze_sentiment_batch, detect_intent, detect_keywords, split_into_sentences,calculate_lead_score
from utils.report_utils import generate_analysis_pdf
from models import db, CallAnalysis  # ✅ Import from models.p
from datetime import datetime
from utils.performance_utils import compute_agent_summary
from flask import request, send_file, flash
import pandas as pd
from datetime import datetime
from flask import send_file, request, url_for
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import matplotlib.pyplot as plt
# import datetime
import os
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'static/uploads'
EXPORT_FOLDER = 'static/exports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)



app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///call_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

AGENT_FILE = 'agents.json'
def load_agents():
    return json.load(open(AGENT_FILE)) if os.path.exists(AGENT_FILE) else []

def save_agents(agents):
    with open(AGENT_FILE, 'w') as f:
        json.dump(agents, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    agents = load_agents()
    result = None
    result_data = None
    sort_by = request.args.get('sort_by', 'date')

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        agent = request.form.get('agent')
        phone = request.form.get('phone')
        model_size = request.form.get('model_size', 'small')
        custom_filename = request.form.get('custom_filename', '').strip()

        if not (file and agent and phone):
            flash("Please fill all fields", "warning")
            return redirect(url_for('index'))

        safe_name = secure_filename(custom_filename or file.filename)
        name, ext = os.path.splitext(safe_name)
        ext = ".mp3"
        filename = f"{name}{ext}"
        base_name = name
        counter = 1

        while CallAnalysis.query.filter_by(filename=filename).first():
            filename = f"{base_name}_{counter}{ext}"
            counter += 1

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        pipeline = MalayalamTranscriptionPipeline(model_size=model_size)
        data = pipeline.transcribe_audio(filepath)
        if not data or not data.get('raw_transcription'):
            flash("❌ Transcription failed", "danger")
            return redirect(url_for('index'))

        data = translate_to_malayalam(data)
        en_transcript = data['raw_transcription']
        ml_transcript = data.get('translated_malayalam', '')

        sentiment = analyze_sentiment_batch([en_transcript])[0]['label']
        intent_info = detect_intent(en_transcript)
        intent = intent_info.get('intent', 'Neutral')
        intent_score = int(intent_info.get('sentiment_score', 0.5) * 100)

        trigger_words = detect_keywords(en_transcript)
        lead_score = calculate_lead_score(intent_score, sentiment, trigger_words)

        ml_analysis = [{'text': s, 'sentiment': analyze_sentiment_batch([s])[0]['label']} for s in split_into_sentences(ml_transcript)]
        en_analysis = []
        for s in split_into_sentences(en_transcript):
            lab = analyze_sentiment_batch([s])[0]['label']
            it = detect_intent(s).get('intent', '-')
            en_analysis.append({'text': s, 'sentiment': lab, 'intent': it})

        comparison_data = []
        max_len = max(len(en_analysis), len(ml_analysis))
        for i in range(max_len):
            en = en_analysis[i] if i < len(en_analysis) else {}
            ml = ml_analysis[i] if i < len(ml_analysis) else {}
            comparison_data.append({
                'english_sentence': en.get('text', ''),
                'malayalam_sentence': ml.get('text', ''),
                'intent_match': en.get('intent') == ml.get('intent', '-'),
                'en_intent': en.get('intent', '-'),
                'ml_intent': ml.get('intent', '-'),
                'en_sentiment': en.get('sentiment', ''),
                'ml_sentiment': ml.get('sentiment', ''),
                'sentiment_diff': 0
            })

        try:
            entry = CallAnalysis(
                filename=filename, agent=agent, phone=phone,
                transcript_en=en_transcript, transcript_ml=ml_transcript,
                sentiment=sentiment, intent=intent,
                intent_score=intent_score, lead_score=int(lead_score),
                trigger_words=", ".join(trigger_words), audio_path=filepath
            )
            db.session.add(entry)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("❌ A file with this name already exists. Try a different custom filename.", "danger")
            return redirect(url_for('index'))

        result = {
            'file_name': filename,
            'transcript_en': en_transcript,
            'transcript_ml': ml_transcript,
            'sentiment': sentiment,
            'intent': intent,
            'intent_score': intent_score,
            'lead_score': int(lead_score),
            'trigger_words': trigger_words,
            'ml_analysis': ml_analysis,
            'en_analysis': en_analysis,
            'comparison_data': comparison_data
        }

        result_data = {
            'ml_sentiments': [x['sentiment'] for x in ml_analysis],
            'en_sentiments': [x['sentiment'] for x in en_analysis]
        }

    if sort_by == 'lead':
        recent = CallAnalysis.query.order_by(CallAnalysis.lead_score.desc()).all()
    elif sort_by == 'intent':
        recent = CallAnalysis.query.order_by(CallAnalysis.intent_score.desc()).all()
    elif sort_by == 'agent':
        recent = CallAnalysis.query.order_by(CallAnalysis.agent.asc()).all()
    else:
        recent = CallAnalysis.query.order_by(CallAnalysis.created_at.desc()).all()

    agent_summary, high_interest_by_agent = compute_agent_summary()

    return render_template("index.html",
        agents=agents,
        result=result,
        result_data=result_data,
        recent_files=recent,
        agent_summary=agent_summary,
        high_interest_by_agent=high_interest_by_agent
    )


# ✅ Optional filtered route for performance filtering
@app.route('/agent_performance', methods=['GET'])
def agent_performance():
    agent = request.args.get("agent")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    agents = load_agents()
    result = None
    result_data = None
    recent = CallAnalysis.query.order_by(CallAnalysis.created_at.desc()).all()

    if not (agent and start_date and end_date):
        flash("Please select agent and date range", "warning")
        return redirect(url_for('index'))

    summary, high_interest = compute_agent_summary(agent=agent, start=start_date, end=end_date)
    return render_template("index.html",
        agents=agents,
        result=result,
        result_data=result_data,
        recent_files=recent,
        agent_summary=summary,
        high_interest_by_agent=high_interest,
        selected_agent=agent,
        start_date=start_date,
        end_date=end_date
    )

@app.route('/download_agent_performance', methods=['GET', 'POST'])
def download_agent_performance():
    try:
        # Get form or query parameters
        agent = request.form.get("agent") or request.args.get("agent")
        start_date = request.form.get("start_date") or request.args.get("start_date")
        end_date = request.form.get("end_date") or request.args.get("end_date")

        if not (agent and start_date and end_date):
            flash("Agent, Start date, and End date are required.", "warning")
            return redirect(url_for("index"))

        # Get summary
        summary, _ = compute_agent_summary(agent=agent, start=start_date, end=end_date)
        weekly_data = summary["weekly_leads"]

        # Calculate average lead scores
        for week_data in weekly_data:
            total = (
                week_data.get("High Interest", 0) * 90 +
                week_data.get("Moderate Interest", 0) * 60 +
                week_data.get("Low Interest", 0) * 30
            )
            count = (
                week_data.get("High Interest", 0) +
                week_data.get("Moderate Interest", 0) +
                week_data.get("Low Interest", 0)
            )
            week_data["avg_lead_score"] = round(total / count, 2) if count > 0 else 0

        # Prepare PDF
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        import matplotlib.pyplot as plt

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph(f"Agent Performance Report: {agent}", styles["Title"]))
        elements.append(Paragraph(f"Date Range: {start_date} to {end_date}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Chart: Weekly Avg Lead Score
        weeks = [w["week"] for w in weekly_data]
        scores = [w["avg_lead_score"] for w in weekly_data]

        plt.figure(figsize=(6, 3))
        plt.bar(weeks, scores, color="skyblue")
        plt.title("Weekly Avg Lead Score")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_buf = BytesIO()
        plt.savefig(chart_buf, format='png')
        plt.close()
        chart_buf.seek(0)

        elements.append(Image(chart_buf, width=400, height=200))
        elements.append(Spacer(1, 12))

        # Table-like breakdown
        for w in weekly_data:
            elements.append(Paragraph(
                f"<b>{w['week']}</b>: Avg Score: {w['avg_lead_score']}, "
                f"High: {w.get('High Interest', 0)}, "
                f"Moderate: {w.get('Moderate Interest', 0)}, "
                f"Low: {w.get('Low Interest', 0)}", styles["Normal"]
            ))
            elements.append(Spacer(1, 6))

        # Build and return PDF
        doc.build(elements)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{agent}_performance_report.pdf"
        )

    except Exception as e:
        flash(f"Failed to generate report: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/generate_leads", methods=["POST"])
def generate_leads():
    try:
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        categories = request.form.getlist("categories")

        if not (start_date and end_date and categories):
            flash("Please select date range and at least one category.", "warning")
            return redirect("/")

        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        # Fetch data from database
        all_entries = CallAnalysis.query.all()

        # Prepare data as DataFrame
        rows = [{
            'date': entry.created_at.date(),
            'agent': entry.agent,
            'lead_score': entry.lead_score,
            'filename': entry.audio_filename,
            'phone': entry.phone,
            'lead_category': (
                "High Interest" if entry.lead_score >= 70 else
                "Moderate Interest" if entry.lead_score >= 40 else
                "Low Interest"
            )
        } for entry in all_entries]

        df = pd.DataFrame(rows)
        df = df[
            (df['date'] >= start_date_obj.date()) &
            (df['date'] <= end_date_obj.date()) &
            (df['lead_category'].isin(categories))
        ]

        leads_data = df.to_dict(orient='records')
        leads_selected = f"{start_date} to {end_date} ({', '.join(categories)})"

        agents = load_agents()
        recent = CallAnalysis.query.order_by(CallAnalysis.created_at.desc()).all()
        result = None
        result_data = None

        return render_template(
            "index.html",
            agents=agents,
            recent_files=recent,
            result=result,
            result_data=result_data,
            leads_data=leads_data,
            leads_selected=leads_selected,
            agent_summary=None,
            high_interest_by_agent=None
        )

    except Exception as e:
        flash(f"Error generating leads report: {str(e)}", "danger")
        return redirect("/")



@app.route('/delete_file', methods=['POST'])
def delete_file():
    fn = request.form.get('filename')
    e = CallAnalysis.query.filter_by(filename=fn).first()
    if e:
        try:
            if os.path.exists(e.audio_path): os.remove(e.audio_path)
            db.session.delete(e)
            db.session.commit()
            flash(f"✅ Deleted {fn}", "success")
        except Exception as ex:
            flash(f"❌ Delete failed: {ex}", "danger")
    else:
        flash("❌ File not found", "warning")
    return redirect(url_for('index'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    filename = request.form.get('file_name')
    entry = CallAnalysis.query.filter_by(filename=filename).first()
    if not entry:
        flash("❌ Report not found.", "danger")
        return redirect(url_for('index'))

    base = filename.replace('.mp3','')
    tmp = os.path.join(EXPORT_FOLDER, base)
    os.makedirs(tmp, exist_ok=True)

    # audio & transcripts
    dest_audio = os.path.join(tmp, filename)
    if os.path.exists(entry.audio_path):
        open(dest_audio,'wb').write(open(entry.audio_path,'rb').read())
    with open(os.path.join(tmp,f"{base}_english.txt"),'w',encoding='utf-8') as f: f.write(entry.transcript_en)
    with open(os.path.join(tmp,f"{base}_malayalam.txt"),'w',encoding='utf-8') as f: f.write(entry.transcript_ml)

    # sentence breakdown
    en_sents = split_into_sentences(entry.transcript_en)
    ml_sents = split_into_sentences(entry.transcript_ml)
    en_data, ml_data = [], []
    for s in en_sents:
        lab = analyze_sentiment_batch([s])[0]['label']
        it  = detect_intent(s).get('intent','-')
        en_data.append({'text':s,'sentiment':lab,'intent':it})
    for s in ml_sents:
        lab = analyze_sentiment_batch([s])[0]['label']
        ml_data.append({'text':s,'sentiment':lab})

    # CSV exports
    pd.DataFrame([{'Sentence':d['text'],'Sentiment':d['sentiment'],'Intent':d['intent']} for d in en_data]).to_csv(os.path.join(tmp,f"{base}_english.csv"),index=False)
    pd.DataFrame([{'Sentence':d['text'],'Sentiment':d['sentiment']} for d in ml_data]).to_csv(os.path.join(tmp,f"{base}_malayalam.csv"),index=False)

    # combined CSV
    rows=[]
    for i in range(max(len(en_data),len(ml_data))):
        rows.append({
          'EN_Sentence': en_data[i]['text'] if i<len(en_data) else '',
          'EN_Sentiment': en_data[i]['sentiment'] if i<len(en_data) else '',
          'EN_Intent': en_data[i]['intent'] if i<len(en_data) else '',
          'ML_Sentence': ml_data[i]['text'] if i<len(ml_data) else '',
          'ML_Sentiment': ml_data[i]['sentiment'] if i<len(ml_data) else ''
        })
    pd.DataFrame(rows).to_csv(os.path.join(tmp,f"{base}_combined.csv"),index=False)

    # PDF
    comp = {'lead_score': entry.lead_score, 'intent_score': entry.intent_score}
    pdfp = generate_analysis_pdf(en_data, ml_data, comp, base)
    pdf_dest = os.path.join(tmp, os.path.basename(pdfp))
    if pdfp != pdf_dest:
        os.replace(pdfp, pdf_dest)

    # ✅ ADD HERE
    entry.report_path = pdf_dest
    db.session.commit()

    # bundle zip
    bio=BytesIO()
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        for r,_,fs in os.walk(tmp):
            for f in fs:
                z.write(os.path.join(r,f),os.path.relpath(os.path.join(r,f),tmp))
    bio.seek(0)
    return send_file(bio,mimetype='application/zip',as_attachment=True,
                     download_name=f"{base}_report_bundle.zip")

@app.route('/export_csv', methods=['POST'])
def export_csv():
    filename = request.form.get('file_name')
    etype    = request.form.get('export_type')
    entry    = CallAnalysis.query.filter_by(filename=filename).first()
    if not entry:
        flash("❌ Export failed", "danger")
        return redirect(url_for('index'))

    en_s = split_into_sentences(entry.transcript_en)
    ml_s = split_into_sentences(entry.transcript_ml)
    en_d=[]; ml_d=[]
    for s in en_s: en_d.append({'Sentence':s,'Sentiment':analyze_sentiment_batch([s])[0]['label'],'Intent':detect_intent(s).get('intent','-')})
    for s in ml_s: ml_d.append({'Sentence':s,'Sentiment':analyze_sentiment_batch([s])[0]['label']})

    if etype=='en': df=pd.DataFrame(en_d);    name=f"{filename.replace('.mp3','')}_english.csv"
    elif etype=='ml':df=pd.DataFrame(ml_d);    name=f"{filename.replace('.mp3','')}_malayalam.csv"
    else:
        rows=[]
        for i in range(max(len(en_d),len(ml_d))):
            rows.append({
                'EN_Sentence':en_d[i]['Sentence'] if i<len(en_d) else '',
                'EN_Sentiment':en_d[i]['Sentiment'] if i<len(en_d) else '',
                'EN_Intent':en_d[i]['Intent'] if i<len(en_d) else '',
                'ML_Sentence':ml_d[i]['Sentence'] if i<len(ml_d) else '',
                'ML_Sentiment':ml_d[i]['Sentiment'] if i<len(ml_d) else ''
            })
        df=pd.DataFrame(rows)
        name=f"{filename.replace('.mp3','')}_combined.csv"
    path=os.path.join(EXPORT_FOLDER,name)
    df.to_csv(path,index=False)
    return send_file(path,as_attachment=True)

@app.route('/view_pdf')
def view_pdf():
    p=request.args.get('path')
    if os.path.exists(p): return send_file(p,mimetype='application/pdf')
    flash("❌ PDF not found","danger"); return redirect(url_for('index'))

@app.route('/download_pdf')
def download_pdf():
    p=request.args.get('path')
    if os.path.exists(p): return send_file(p,as_attachment=True,download_name=os.path.basename(p))
    flash("❌ PDF not found","danger"); return redirect(url_for('index'))

@app.route('/add_agent', methods=['POST'])
def add_agent():
    na=request.form.get('new_agent');ags=load_agents()
    if na and na not in ags: ags.append(na); save_agents(ags); flash(f"Agent '{na}' added","success")
    else: flash("Agent exists or empty","warning")
    return redirect(url_for('index'))

@app.route('/delete_agent', methods=['POST'])
def delete_agent():
    ag=request.form.get('delete_agent');pw=request.form.get('delete_password');ags=load_agents()
    if pw=='123456' and ag in ags: ags.remove(ag); save_agents(ags); flash(f"Agent '{ag}' deleted","danger")
    else: flash("Wrong password","danger")
    return redirect(url_for('index'))

if __name__=='__main__':
    app.run(debug=True)
