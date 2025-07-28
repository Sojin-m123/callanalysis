# utils/report_utils.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf


def generate_analysis_pdf(en_analysis, ml_analysis, comparison_data, filename_prefix):
    pdf_path = f"static/exports/{filename_prefix}_summary_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 📝 Key Metrics
    en_scores = [s["sentiment_score"] for s in en_analysis if "sentiment_score" in s]
    ml_scores = [s["sentiment_score"] for s in ml_analysis if "sentiment_score" in s]
    avg_en = round(sum(en_scores) / len(en_scores), 2) if en_scores else 0
    avg_ml = round(sum(ml_scores) / len(ml_scores), 2) if ml_scores else 0
    combined_avg = round((avg_en + avg_ml) / 2, 2)
    lead_score = comparison_data["lead_score"]
    intent_score = comparison_data["intent_score"]

    interpretation = "High interest lead" if lead_score >= 60 else "Low interest lead"

    story.append(Paragraph("🔍 <b>Key Metrics</b>", styles["Heading2"]))
    story.append(Paragraph(f"<b>English Avg Sentiment:</b> {avg_en}", styles["Normal"]))
    story.append(Paragraph(f"<b>Malayalam Avg Sentiment:</b> {avg_ml}", styles["Normal"]))
    story.append(Paragraph(f"<b>Combined Avg Sentiment:</b> {combined_avg}", styles["Normal"]))
    story.append(Paragraph(f"<b>Lead Score:</b> {lead_score}/100", styles["Normal"]))
    story.append(Paragraph(f"<b>Intent Score:</b> {intent_score}/100", styles["Normal"]))
    story.append(Paragraph(f"<b>Interpretation:</b> {interpretation}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # 📊 English Sentiment Distribution
    en_df = pd.DataFrame(en_analysis)
    fig1 = plt.figure()
    en_df["sentiment"].value_counts().plot(kind='bar', color='skyblue', title="English Sentiment Distribution")
    plt.ylabel("Count")
    story.append(RLImage(save_plot(fig1), width=400, height=250))
    plt.close(fig1)

    # 📊 Malayalam Sentiment Distribution
    ml_df = pd.DataFrame(ml_analysis)
    fig2 = plt.figure()
    ml_df["sentiment"].value_counts().plot(kind='bar', color='lightgreen', title="Malayalam Sentiment Distribution")
    plt.ylabel("Count")
    story.append(RLImage(save_plot(fig2), width=400, height=250))
    plt.close(fig2)

    # 📈 Sentiment Trend
    if en_scores and ml_scores:
        fig3 = plt.figure()
        plt.plot(range(len(en_scores)), en_scores, label='English', color='blue')
        plt.plot(range(len(ml_scores)), ml_scores, label='Malayalam', color='green')
        plt.title("Sentiment Trend Over Conversation")
        plt.xlabel("Sentence Number")
        plt.ylabel("Sentiment Score")
        plt.legend()
        story.append(RLImage(save_plot(fig3), width=400, height=250))
        plt.close(fig3)

    # 📊 Intent Distribution
    fig4 = plt.figure()
    pd.Series([x['intent'] for x in en_analysis if 'intent' in x]).value_counts().plot(kind='bar', color='purple', title="Intent Distribution")
    plt.ylabel("Count")
    story.append(RLImage(save_plot(fig4), width=400, height=250))
    plt.close(fig4)

    # 📉 Sentiment Differences
    diffs = [
        abs(en["sentiment_score"] - ml["sentiment_score"])
        for en, ml in zip(en_analysis, ml_analysis)
        if "sentiment_score" in en and "sentiment_score" in ml
    ]
    if diffs:
        fig5 = plt.figure()
        pd.Series(diffs).plot.hist(bins=10, title="English-Malayalam Sentiment Differences", color='orange')
        plt.xlabel("Sentiment Score Difference")
        story.append(RLImage(save_plot(fig5), width=400, height=250))
        plt.close(fig5)

    doc.build(story)
    return pdf_path
