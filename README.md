# Resume Shortlister
This Flask web app intelligently ranks resumes against a job description. It uses a hybrid model, combining SBERT for semantic meaning and a spaCy PhraseMatcher for keywords. It also uses Regex to extract CGPA and negative skills, applying penalties to a final weighted score, and displays results in a ranked, highlighted table.

<h2>Features :</h2>

<li><strong>Hybrid Scoring System</strong> 
  The final ranking is not based on one model, but a weighted average of semantic meaning and explicit keyword matches, plus penalties.</li>

<li><strong>Semantic Matching (SBERT):</strong> Uses the all-MiniLM-L6-v2 Sentence Transformer to understand the context and meaning of the JD and resumes. It can identify that "cloud expert" and "AWS engineer" are conceptually similar.</li>

<li><strong>Keyword Matching (spaCy):</strong> Uses a high-speed PhraseMatcher to find specific keywords from a large, custom DOMAIN_ONTOLOGY.</li>
<br></br>
<strong>Rule-Based Engine (Regex):</strong>
<br></br>
<li>CGPA Validation: Automatically extracts and compares CGPA from the JD (e.g., "must be greater than 8.6") and each resume (e.g., "CGPA: 8.8/10").</li>

<li>Negative Skill Penalties: Intelligently parses sentences to find negative keywords (e.g., "Python not preferred" or "avoid Flutter") and applies a heavy penalty if a resume contains those skills.</li>
<br></br>
<li><strong>Dual-Model Comparison: </strong>The results table displays the SBERT (Semantic) score side-by-side with the TF-IDF (Lexical) score, allowing for easy comparison and analysis.</li>

<li><strong>Visual Highlighting:</strong> Generates a separate HTML version of the JD and each resume, dynamically highlighting:

<li>Green: Matched skills and passing CGPA.</li>

<li>Red: Negative skills and failing CGPA.</li>
<br></br>

<h2>Technologies Used</h2>
<br></br>
<li>Backend: Flask</li>

<li>Semantic NLP: sentence-transformers (SBERT)</li>

<li>Keyword NLP: spacy (PhraseMatcher, Sentencizer)</li>

<li>Lexical NLP: scikit-learn (TfidfVectorizer, cosine_similarity)</li>

<li>File Parsing: PyPDF2, python-docx</li>

<li>Rule Engine: re (Python's built-in Regular Expressions)</li>

<li>Data Handling: pandas</li>

<li>Frontend: HTML5, CSS3 (as seen in index3.html)</li>

<h2>Example Result shown: </h2>

Example 1 :

JD uploaded:

<img width="1132" height="653" alt="image" src="https://github.com/user-attachments/assets/f0c4c878-ed46-4083-9ba2-bcbe672542df" />


Results obtained: 

<img width="1136" height="715" alt="image" src="https://github.com/user-attachments/assets/5cf9a7a7-5cae-4990-9f3d-693d8296f392" />

Example 2 : 

JD Uploaded:

<img width="552" height="283" alt="image" src="https://github.com/user-attachments/assets/0579fd87-3136-450a-9f15-104386ff7b09" />

Results obtained:

<img width="1136" height="741" alt="image" src="https://github.com/user-attachments/assets/70b18999-b8ee-41ef-aba4-226f08af6335" />

Negative skills highlighted in red colour: 

<img width="849" height="284" alt="image" src="https://github.com/user-attachments/assets/0a97f60f-a847-484e-b5f2-a0507251ef39" />



