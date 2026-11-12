"""User-facing text shared by the Gradio (app.py) and Streamlit demos."""

from __future__ import annotations

TITLE = "News Credibility Analyzer"
SUBTITLE = "বাংলা সংবাদের লেখা বিশ্লেষণ · Bangla news text-style classifier"

NOTICE_HTML = (
    "This tool estimates whether a Bangla news text <b>resembles</b> the fake or authentic "
    "examples it was trained on (BanFakeNews-2.0). It does <b>not</b> check facts, sources or "
    "dates, and it cannot tell you whether a story is true. Always confirm important claims with "
    "reputable sources or a fact-checking organisation."
)

# Illustrative texts written for this demo. They are not real news reports.
EXAMPLES = [
    "ঢাকা : রাজধানীতে আজ সকাল থেকে থেমে থেমে বৃষ্টি হচ্ছে। এতে অফিসগামী মানুষ বিভিন্ন সড়কে যানজটে "
    "ভোগান্তিতে পড়েন। আবহাওয়া অধিদপ্তর জানিয়েছে, মৌসুমি বায়ুর প্রভাবে আগামী দুই দিন দেশের বিভিন্ন "
    "অঞ্চলে হালকা থেকে মাঝারি ধরনের বৃষ্টি হতে পারে। চট্টগ্রাম ও সিলেট বিভাগের কোথাও কোথাও ভারী "
    "বর্ষণের সম্ভাবনাও রয়েছে। সমুদ্রবন্দরগুলোকে তিন নম্বর সতর্কসংকেত দেখাতে বলা হয়েছে। এদিকে ঢাকা "
    "উত্তর সিটি করপোরেশন জানিয়েছে, জলাবদ্ধতা নিরসনে বিভিন্ন এলাকায় কর্মী মোতায়েন করা হয়েছে।",
    "অবিশ্বাস্য! মাত্র তিন দিনে এক গ্লাস গরম লেবু পানি খেয়ে ডায়াবেটিস চিরতরে দূর করার গোপন উপায় "
    "ফাঁস করেছেন এক গ্রামের কবিরাজ! ডাক্তাররা চান না আপনি এই খবর জানুন। হাজার হাজার মানুষ ইতোমধ্যে "
    "ওষুধ খাওয়া বন্ধ করে দিয়েছেন এবং সবাই সুস্থ হয়ে গেছেন বলে দাবি করা হচ্ছে। খবরটি দ্রুত শেয়ার করুন, "
    "কারণ যেকোনো সময় এটি মুছে ফেলা হতে পারে! বিশেষজ্ঞরা নাকি এই আবিষ্কার দেখে হতবাক হয়ে গেছেন।",
]

NOTICE_MARKDOWN = NOTICE_HTML.replace("<b>", "**").replace("</b>", "**")


def about_markdown(metadata: dict) -> str:
    """Model summary and limitations, built from the saved model metadata."""
    m = metadata.get("test_metrics", {})
    if not m:
        return "Model metadata unavailable."
    fake = m["per_class"]["fake"]
    return (
        f"**Model:** `{metadata.get('selected_candidate')}` (TF-IDF + linear classifier, scikit-learn)  \n"
        f"**Held-out test set:** {m['n']:,} de-duplicated articles, grouped so near-duplicates never "
        f"cross the train/test split.  \n"
        f"**Macro F1:** {m['macro_f1']:.3f} · **Accuracy:** {m['accuracy']:.3f} · "
        f"**ROC-AUC:** {m.get('roc_auc', float('nan')):.3f}  \n"
        f"**Fake class:** precision {fake['precision']:.3f}, recall {fake['recall']:.3f}, F1 {fake['f1']:.3f}\n\n"
        "**Limitations.** The training articles come from a fixed period and a limited set of Bangla "
        "news portals, and the fake examples include satirical and sensational writing. The model "
        "therefore partly learns *writing style and source*, not truth. New topics, short texts and "
        "well-written misinformation can easily be scored as authentic-like.\n\n"
        "Labels: `0 = fake`, `1 = authentic`. Dataset: BanFakeNews-2.0, H. M. Shibu, Mendeley Data, "
        "doi:10.17632/kjh887ct4j.1 (CC BY 4.0)."
    )
