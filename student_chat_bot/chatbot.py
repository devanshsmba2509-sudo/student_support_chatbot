import nltk
import numpy as np
import pandas as pd
import random
import string
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ---------------- General FAQ Knowledge Base ----------------
student_support_data = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "namaste"],
        "responses": ["Hello! I'm your Student Support Assistant. How can I help you today?"]
    },
    "admission": {
        "patterns": ["admission", "how to apply", "enroll", "admission process", "eligibility"],
        "responses": ["For admissions, visit the college admission portal, fill the online form, and pay the application fee. The admission window is usually May to July."]
    },
    "fees": {
        "patterns": ["fee structure", "tuition fee details", "installment options"],
        "responses": ["Fee payment can be done online via the student portal. Fees are usually paid in 2 installments per year."]
    },
    "library": {
        "patterns": ["library timing", "how many books can i borrow", "library card"],
        "responses": ["The library is open Monday-Saturday, 9 AM to 8 PM. You can borrow up to 3 books for 15 days."]
    },
    "hostel": {
        "patterns": ["hostel accommodation", "warden contact", "mess timings"],
        "responses": ["Hostel accommodation requests must be submitted online. Contact the hostel warden's office for details."]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "helpful"],
        "responses": ["You're welcome! Let me know if you need anything else."]
    },
    "bye": {
        "patterns": ["bye", "goodbye", "see you", "exit", "quit"],
        "responses": ["Goodbye! Have a great day."]
    }
}

# ---------------- Preprocessing ----------------
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# ---------------- Build General FAQ Vectorizer ----------------
general_corpus = []
general_tags = []
for tag, content in student_support_data.items():
    for pattern in content["patterns"]:
        general_corpus.append(preprocess(pattern))
        general_tags.append(tag)

general_vectorizer = TfidfVectorizer()
general_matrix = general_vectorizer.fit_transform(general_corpus)

# ---------------- Load CSV Data & Build Separate Vectorizer ----------------
df = pd.read_csv("university_query_train.csv")
unique_rows = df.drop_duplicates(subset="Student_Query").reset_index(drop=True)

csv_queries = unique_rows["Student_Query"].tolist()
csv_departments = unique_rows["Department"].tolist()
csv_priorities = unique_rows["Priority_Label"].tolist()
csv_clean = [preprocess(q) for q in csv_queries]

csv_vectorizer = TfidfVectorizer()
csv_matrix = csv_vectorizer.fit_transform(csv_clean)

# ---------------- Combined Response Logic ----------------
def get_response(user_input):
    processed = preprocess(user_input)

    csv_vec = csv_vectorizer.transform([processed])
    csv_sim = cosine_similarity(csv_vec, csv_matrix)
    csv_best_idx = np.argmax(csv_sim)
    csv_best_score = csv_sim[0][csv_best_idx]

    gen_vec = general_vectorizer.transform([processed])
    gen_sim = cosine_similarity(gen_vec, general_matrix)
    gen_best_idx = np.argmax(gen_sim)
    gen_best_score = gen_sim[0][gen_best_idx]

    if csv_best_score >= 0.4:
        department = csv_departments[csv_best_idx]
        priority = csv_priorities[csv_best_idx]  # internal use ke liye rakha hai

        # Student ko sirf department aur ek natural time-estimate dikhao, "Priority: High" jaisa raw label nahi
        if priority == "High":
            response = (f"This has been forwarded to **{department}** on priority — "
                        f"you should hear back within 24 hours.")
        elif priority == "Medium":
            response = (f"Your query has been forwarded to **{department}**. "
                        f"You can expect a response within 2-3 working days.")
        else:
            response = (f"Your query has been forwarded to **{department}**. "
                        f"Expect a response within a week.")

        # Agar chaho to yaha priority ko console/log file mein save kar sakte ho staff ke liye
        print(f"[INTERNAL LOG] Query: '{user_input}' | Department: {department} | Priority: {priority}")

        return response

    if gen_best_score >= 0.3:
        matched_tag = general_tags[gen_best_idx]
        return random.choice(student_support_data[matched_tag]["responses"])

    return ("I'm not fully sure I understood that. Could you rephrase? "
            "You can ask about admissions, fees, exams, library, hostel, or specific issues like password reset, admit card, etc.")

# ---------------- Gradio UI ----------------
def chatbot_interface(message, history):
    return get_response(message)

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"]
)

demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="🎓 Student Support Services Chatbot",
    description="Namaste! General FAQs (admissions, fees, library) ya specific issues (password, admit card, scholarship) — kuch bhi pooch sakte ho!",
   # theme=custom_theme,
    examples=[
        "How do I apply for admission?",
        "My admit card has incorrect details.",
        "I have not received my scholarship amount.",
        "Library timings kya hai?",
        "LMS is not allowing assignment upload."
    ]
)

if __name__ == "__main__":
    demo.launch()