import os
import re 
import PyPDF2
import docx
import spacy
import pandas as pd
from collections import Counter
from spacy.matcher import PhraseMatcher
from flask import Flask, request, render_template, send_from_directory
import html

# --- Imports for Sentence Transformer ---
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("\n--- 'sentence-transformers' not found ---")
    print("Please run: pip install sentence-transformers")
    exit()

# --- NEW: Imports for TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- 1. GLOBAL SETUP: App Config & NLP Models ---

app = Flask(__name__)

# Get the absolute path to the directory this file is in
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Create an absolute path for the upload folder
UPLOAD_FOLDER_PATH = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH

# --- *** EXPANDED DOMAIN ONTOLOGY *** ---
DOMAIN_ONTOLOGY = {
    # Programming Languages
    "PROGRAMMING_LANGUAGES": [
        "python", "py", "java", "c", "c++", "c plus plus", "c#", "r", "php",
        "go", "golang", "rust", "javascript", "js", "typescript"
    ],
    # Frontend
    "FRONTEND_DEVELOPMENT": [
        "react", "react.js", "reactjs", "angular", "vue.js", "html", "css", 
        "frontend", "tailwind css", "figma", "vanilla css", "flutter"
    ],
    # Backend
    "BACKEND_DEVELOPMENT": [
        "node.js", "nodejs", "express.js", "django", "flask", "spring", 
        "spring boot", "backend", "ruby on rails", "fastapi", "firebase",
        "rest apis", "microservices"
    ],
    # Databases
    "DATABASE_MANAGEMENT": [
        "sql", "mysql", "postgresql", "t-sql", "pl/sql", "database management", 
        "dbms", "database administration", "nosql", "mongodb", "oracle",
        "sql workbench", "tinydb"
    ],
    # Cloud
    "CLOUD_COMPUTING": [
        "aws", "azure", "gcp", "google cloud", "amazon web services", "docker", 
        "kubernetes", "cloud", "devops", "ci/cd", "ec2", "s3", "oci", "vertexai"
    ],
    # AI/ML/Data Science
    "DATA_SCIENCE_AI_ML": [
        "data science", "machine learning", "ml", "deep learning", "ai",
        "artificial intelligence", "scikit-learn", "sklearn", "pandas", "numpy", 
        "data analysis", "data analytics", "tensorflow", "pytorch", "keras",
        "matplotlib", "rstudio", "nltk", "spacy", "hugging face", "bart",
        "data processing", "ai/ml basics", "prompt engineering", "generative al"
    ],
    # ECE/EEE/Embedded
    "EMBEDDED_IOT_EEE": [
        "embedded c", "embedded systems", "iot", "automation", "raspberry pi",
        "arduino", "esp32", "microcontroller", "power electronics", 
        "control systems", "signals & systems", "power systems analysis",
        "keil µvision", "ltspice", "pdfplumber", "pymupdf", "owasp"
    ],
    # Simulation & Design
    "SIMULATION_DESIGN": [
        "matlab", "simulink", "cadence", "altium designer", "kicad", "labview",
        "unity", "game dev"
    ],
    # Dev Tools & Platforms
    "DEV_TOOLS": [
        "git", "github", "vs code", "visual studio", "jupyter notebook", "google colab", 
        "eclipse", "co-pilot", "vs", "linux", "cisco packet tracer"
    ],
    # General Concepts
    "CONCEPTS": [
        "data structures", "algorithms", "data structures and algorithms", 
        "dbms", "system design", "cryptography", "oops", "object-oriented programming",
        "computer networks", "operating systems", "software engineering", "machine vision"
    ],
    # Security
    "SECURITY": [
        "cybersecurity", "log analysis", "encryption", "soc tools", "cryptography"
    ],
    # Soft Skills (Used for semantic context, not usually for matching)
    "SOFT_SKILLS": [
        "communication", "teamwork", "problem solving", "adaptability", 
        "quick learner", "detail-oriented", "leadership", "analytical"
    ]
}


# --- Create a reverse map for skill -> group ---
SKILL_TO_GROUP_MAP = {
    skill: group 
    for group, skills in DOMAIN_ONTOLOGY.items() 
    for skill in skills
}

# --- Load NLP Models ---
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("\n--- Spacy model 'en_core_web_sm' not found ---")
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

print("Loading Sentence Transformer model... (This may take a moment)")
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded.")
except Exception as e:
    print(f"Error loading SBERT model: {e}")
    print("Please ensure you have an internet connection for the first download.")
    exit()


def load_skills_and_build_matcher(domain_ontology):
    skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for domain, skills in domain_ontology.items():
        patterns = [nlp(skill.strip()) for skill in skills] 
        skill_matcher.add(domain.upper(), patterns)
    print(f"PhraseMatcher built successfully with {len(domain_ontology)} domains.")
    return skill_matcher

skill_matcher = load_skills_and_build_matcher(DOMAIN_ONTOLOGY)

# --- 2. HELPER FUNCTIONS ---

# --- *** CORRECTED CGPA FUNCTION *** ---
def extract_cgpa(text):
    """ 
    Extracts CGPA using regex. 
    Returns a tuple: (float_value, matched_string) 
    """
    pattern = r"""
        \b(                          # Start capturing group 1 (the full string)
            (?:CGPA|GPA)            # Match "CGPA" or "GPA"
            (?:[\s\:\-]+|           # Match separators like ":", "-", or spaces
               (?:\s+must\s+be)?\s+ # Or match optional " must be "
               (?:greater\s+than|above|minimum\s+of|of|is)\s+ # Match comparison words
            )
            ([\d\.]+)               # Capture group 2 (the number)
            (?:[\/]?[\d\.]*)?       # Optionally match "/10" or "/10.0"
        )\b
    """
    
    match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
    
    if match:
        try:
            val = float(match.group(2)) # Get the number
            string = match.group(1)   # Get the full matched text
            return (val, string)
        except (ValueError, IndexError):
            pass

    pattern2 = r"""
        \b(                          # Start capturing group 1 (the full string)
            ([\d\.]+)               # Capture group 2 (the number)
            \/[\d\.]+\s* # Match "/10" or "/10.0"
            (?:CGPA|GPA)            # Match "CGPA" or "GPA"
        )\b
    """
    match2 = re.search(pattern2, text, re.IGNORECASE | re.VERBOSE)
    
    if match2:
        try:
            val = float(match2.group(2)) # Get the number
            string = match2.group(1)   # Get the full matched text
            return (val, string)
        except (ValueError, IndexError):
            pass

    return (0.0, None)

# --- *** CORRECTED NEGATIVE SKILL FUNCTION *** ---
NEGATIVE_TRIGGERS = {
    'not preferred', 'not desired', 'not required', 'not looking for',
    'avoid', 'no experience', 'not preferrable'
}

def find_negative_skills(doc):
    """
    Finds skills mentioned in a negative context by checking sentences.
    """
    negative_skills = set()
    
    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        if any(trigger in sent_text_lower for trigger in NEGATIVE_TRIGGERS):
            negative_doc = nlp(sent.text) 
            skills_in_sentence = find_skills_with_ontology(negative_doc)
            negative_skills.update(skills_in_sentence)
            
    return negative_skills

def find_skills_with_ontology(doc):
    """ This function finds ALL skills """
    matches = skill_matcher(doc)
    found_skills_text = set()
    for match_id, start, end in matches:
        skill_text = doc[start:end].text
        found_skills_text.add(skill_text.lower())
    return found_skills_text

def extract_email(doc_text):
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', doc_text)
    return email.group(0) if email else 'N/A'

# --- (Text extraction functions) ---
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        doc_text = [para.text for para in doc.paragraphs]
        return '\n'.join(doc_text)
    except Exception as e:
        return f"An error occurred: {e}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"An error occurred: {e}"

def extract_text(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0: return ""
    if file_path.endswith(".pdf"): return extract_text_from_pdf(file_path)
    if file_path.endswith(".docx"): return extract_text_from_docx(file_path)
    if file_path.endswith(".txt"): return extract_text_from_txt(file_path)
    return ""
# --- (End of text extraction) ---

def preprocess_text(text):
    """ 
    This is for the SBERT model. 
    """
    text = re.sub(r'\s+', ' ', text) # Consolidate whitespace
    return text.strip()

# --- NEW: Preprocessing function for TF-IDF ---
def preprocess_text_for_tfidf(text):
    """ Cleans and lemmatizes text for TF-IDF. """
    # This function needs the 'nlp' model
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_alpha and not token.is_stop and not token.is_punct:
            lemmas.append(token.lemma_.lower())
            # print(token.lemma_, token.pos_, token.dep_, token.ent_type)
    return " ".join(lemmas)

# --- UPGRADED HTML Highlighter ---
def generate_highlighted_html(raw_text, green_skills, red_skills, green_strings=[], red_strings=[]):
    """
    Takes raw text and highlights:
    - green_skills (using word boundaries)
    - red_skills (using word boundaries)
    - green_strings (exact string match)
    - red_strings (exact string match)
    """
    html_safe_text = html.escape(raw_text)
    html_safe_text = html_safe_text.replace('\n', '<br>\n')
    
    HIGHLIGHT_CSS = """
    body { font-family: sans-serif; }
    pre { 
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: monospace; 
      font-size: 14px;
      line-height: 1.4;
      padding: 15px;
    }
    .highlight-green { 
      background-color: #a8e6cf;
      font-weight: bold;
      border-radius: 3px;
      padding: 1px 2px;
    }
    .highlight-red { 
      background-color: #fdd8d8;
      font-weight: bold;
      border: 1px solid #d10000;
      border-radius: 3px;
      padding: 1px 2px;
    }
    """
    
    highlighted_text = html_safe_text

    # Highlight Skills (using regex with word boundaries)
    if red_skills:
        red_set = set(skill.lower() for skill in red_skills)
        sorted_red = sorted(list(red_set), key=len, reverse=True)
        pattern_red = r'\b(' + '|'.join(re.escape(skill) for skill in sorted_red) + r')\b'
        highlighted_text = re.sub(
            pattern_red, 
            r'<span class="highlight-red">\1</span>', 
            highlighted_text, 
            flags=re.IGNORECASE
        )

    if green_skills:
        green_set = set(skill.lower() for skill in green_skills)
        sorted_green = sorted(list(green_set), key=len, reverse=True)
        pattern_green = r'\b(' + '|'.join(re.escape(skill) for skill in sorted_green) + r')\b'
        highlighted_text = re.sub(
            pattern_green, 
            r'<span class="highlight-green">\1</span>', 
            highlighted_text, 
            flags=re.IGNORECASE
        )

    # Highlight Exact Strings (like CGPA, no word boundaries)
    for string in red_strings:
        if string: # Avoid None
            highlighted_text = re.sub(
                re.escape(string),
                f'<span class="highlight-red">{string}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )
        
    for string in green_strings:
        if string: # Avoid None
             highlighted_text = re.sub(
                re.escape(string),
                f'<span class="highlight-green">{string}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )

    final_html = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <title>Highlighted Document</title>
      <style>{HIGHLIGHT_CSS}</style>
    </head>
    <body>
      <pre>{highlighted_text}</pre>
    </body>
    </html>
    """
    
    return final_html


# --- 3. FLASK APP ROUTES ---

@app.route('/')
def matchresume():
    return render_template('index3.html') 

@app.route('/upload', methods=['POST'])
def matcher():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    jd_file = request.files.get('job_description_file')
    resume_files = request.files.getlist('resumes')

    if not jd_file or not jd_file.filename:
        return render_template('index3.html', message="Please upload a Job Description file.")
    if not resume_files or not resume_files[0].filename:
        return render_template('index3.html', message="Please upload at least one resume.")

    # --- === 1. PROCESS JOB DESCRIPTION === ---
    jd_filename_original = "jd_" + jd_file.filename
    jd_filepath = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename_original)
    jd_file.save(jd_filepath)
    
    job_description_text = extract_text(jd_filepath)
    
    jd_required_cgpa_val, jd_cgpa_str = extract_cgpa(job_description_text)
    
    jd_text_for_rules = re.sub(r'[,/()\-]', ' ', job_description_text)
    jd_doc = nlp(jd_text_for_rules) 
    
    all_jd_skills = find_skills_with_ontology(jd_doc)
    jd_negative_skills = find_negative_skills(jd_doc) 
    
    jd_required_skills = all_jd_skills - jd_negative_skills
    
    print(f"--- JD Rules ---")
    print(f"Required CGPA: {jd_required_cgpa_val} (from string: '{jd_cgpa_str}')")
    print(f"All Skills Found: {all_jd_skills}")
    print(f"Negative Skills Found: {jd_negative_skills}")
    print(f"Final Required Skills: {jd_required_skills}")
    
    if not jd_required_skills:
        return render_template('index3.html', message="No required skills found in the Job Description.")
    
    # --- SBERT Processing ---
    jd_processed_text = preprocess_text(job_description_text)
    jd_vector = sbert_model.encode(jd_processed_text, convert_to_tensor=True)
    
    # --- NEW: TF-IDF Processing ---
    jd_tfidf_text = preprocess_text_for_tfidf(job_description_text) # Use full text
    
    jd_cgpa_green_strings = [jd_cgpa_str] if jd_cgpa_str else []
    
    jd_html_content = generate_highlighted_html(
        job_description_text, 
        jd_required_skills,   # Green skills
        jd_negative_skills,   # Red skills
        green_strings=jd_cgpa_green_strings
    )
    jd_html_filename = "jd_highlighted_" + os.path.splitext(jd_file.filename)[0] + ".html"
    jd_html_filepath = os.path.join(app.config['UPLOAD_FOLDER'], jd_html_filename)
    
    try:
        with open(jd_html_filepath, 'w', encoding='utf-8') as f:
            f.write(jd_html_content)
    except Exception as e:
        print(f"Error writing highlighted JD HTML: {e}")
        jd_html_filename = None
    
    # --- === 2. PROCESS RESUMES === ---
    resumes_data = []
    resume_tfidf_texts = [] # <-- NEW: Store texts for TF-IDF
    
    for resume_file in resume_files:
        if resume_file and resume_file.filename:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            
            resume_raw_text = extract_text(filename)
            if not resume_raw_text: continue

            # --- NEW: Add preprocessed text to list for TF-IDF ---
            resume_tfidf_texts.append(preprocess_text_for_tfidf(resume_raw_text))

            # --- Hard Rule Checking ---
            resume_cgpa_val, resume_cgpa_str = extract_cgpa(resume_raw_text)
            cgpa_match_status = "N/A" 
            resume_cgpa_green = []
            resume_cgpa_red = []

            if jd_required_cgpa_val > 0:
                if resume_cgpa_val > 0:
                    if resume_cgpa_val >= jd_required_cgpa_val:
                        cgpa_match_status = "Pass"
                        if resume_cgpa_str:
                            resume_cgpa_green.append(resume_cgpa_str)
                    else:
                        cgpa_match_status = "Fail"
                        if resume_cgpa_str:
                            resume_cgpa_red.append(resume_cgpa_str)
                else:
                    cgpa_match_status = "Not Found"
            
            # Create the resume doc
            resume_text_for_rules = re.sub(r'[,/()\-]', ' ', resume_raw_text)
            resume_doc = nlp(resume_text_for_rules)
            all_resume_skills = find_skills_with_ontology(resume_doc)
            email = extract_email(resume_raw_text)
            
            # --- === 3. SCORING (SBERT, KEYWORD, PENALTY) === ---
            
            resume_processed_text = preprocess_text(resume_raw_text)
            resume_vector = sbert_model.encode(resume_processed_text, convert_to_tensor=True)
            semantic_score = util.cos_sim(jd_vector, resume_vector).item()

            matched_skills = jd_required_skills.intersection(all_resume_skills)
            if jd_required_skills:
                keyword_score_normalized = len(matched_skills) / len(jd_required_skills)
            else:
                keyword_score_normalized = 0
            
            found_negative_skills = jd_negative_skills.intersection(all_resume_skills)
            negative_penalty = 0.0
            if found_negative_skills:
                negative_penalty = 0.5 * len(found_negative_skills) 
            
            base_score = (0.3 * semantic_score) + (0.7 * keyword_score_normalized)
            final_score = base_score - negative_penalty
            final_score = max(0, final_score) 

            # --- Generate highlighted HTML for Resume ---
            resume_html_content = generate_highlighted_html(
                resume_raw_text,
                matched_skills,           # Green skills
                found_negative_skills,    # Red skills
                green_strings=resume_cgpa_green,
                red_strings=resume_cgpa_red
            )
            resume_html_filename = os.path.splitext(resume_file.filename)[0] + "_highlighted.html"
            resume_html_filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_html_filename)
        
            try:
                with open(resume_html_filepath, 'w', encoding='utf-8') as f:
                    f.write(resume_html_content)
            except Exception as e:
                print(f"Error writing highlighted resume HTML: {e}")
                resume_html_filename = None

            resumes_data.append({
                'filename': resume_file.filename,
                'highlighted_file': resume_html_filename,
                'email': email,
                
                'final_score': final_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score_normalized,
                'negative_penalty': negative_penalty,
                
                'matched_skills_list': sorted(list(matched_skills)),
                'found_negative_list': sorted(list(found_negative_skills)),
                
                'match_count': len(matched_skills),
                'total_jd_skills': len(jd_required_skills),
                
                'resume_cgpa_val': resume_cgpa_val,
                'cgpa_match_status': cgpa_match_status
                # 'tfidf_score' will be added next
            })
    
    if not resumes_data:
        return render_template('index3.html', message="Could not process any of the uploaded resumes.")

    # --- === 4. NEW: TF-IDF CALCULATION (AFTER LOOP) === ---
    all_tfidf_texts = [jd_tfidf_text] + resume_tfidf_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tfidf_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    tfidf_scores = similarities[0][1:] # Get scores for all resumes

    # --- === 5. ADD TF-IDF SCORES & SORT === ---
    for i, resume in enumerate(resumes_data):
        resume['tfidf_score'] = tfidf_scores[i] # <-- Add the score here

    sorted_resumes = sorted(resumes_data, 
                            key=lambda x: x['final_score'], 
                            reverse=True)
    
    top_5_resumes = sorted_resumes[:5]
    
    jd_skills_for_template = {skill: 1 for skill in jd_required_skills}
    
    return render_template('index3.html', 
                           results=top_5_resumes, 
                           jd_skills_dict=jd_skills_for_template,
                           jd_negative_skills=sorted(list(jd_negative_skills)),
                           original_jd_filename=jd_filename_original,
                           jd_highlighted_filename=jd_html_filename,
                           jd_required_cgpa=jd_required_cgpa_val
                           )

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)