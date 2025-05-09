# ECHO-MED: AI-Powered Clinical Documentation Assistant

ECHO-MED is an innovative Streamlit web app that leverages AI to:
- Transcribe medical consultations (via audio upload or direct recording)
- Extract key patient information
- Generate comprehensive clinical assessments
- Create precise, downloadable prescriptions

## Features
- **Audio Input:** Upload or record audio for doctor-patient consultations
- **PDF Upload:** Add past medical records for richer context
- **AI Transcription:** Uses OpenAI Whisper for accurate transcription
- **Clinical Assessment:** Extracts chief complaints, patient data, history, differential diagnosis, and summary
- **Prescription Generator:** Produces a structured prescription and downloadable PDF

## Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up OpenAI API Key:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage
Run the app with:
```bash
streamlit run app.py
```

- Use the sidebar to navigate between About, Clinical Assessment, and Prescription Generator.
- Upload or record audio as prompted.
- Optionally upload past medical records (PDF) for clinical assessment.
- Download generated prescriptions as PDF.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
MIT 