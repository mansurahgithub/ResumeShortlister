# ResumeShortlister
This Flask web app intelligently ranks resumes against a job description. It uses a hybrid model, combining SBERT for semantic meaning and a spaCy PhraseMatcher for keywords. It also uses Regex to extract CGPA and negative skills, applying penalties to a final weighted score, and displays results in a ranked, highlighted table.
