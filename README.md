pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create `.env` file and add your OpenRouter API key:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_dspy_demo.py
   ```

## Creating a GitHub Repository

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name your repository
   - Choose public/private
   - Don't initialize with README (we already have one)

2. Initialize the local repository and push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_REPOSITORY_URL
   git push -u origin main
   ```

## Project Structure
```
├── streamlit_dspy_demo.py  # Main application file
├── .gitignore             # Git ignore rules
└── README.md             # This file