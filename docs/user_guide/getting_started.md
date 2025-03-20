# Getting Started with Bible-AI

Welcome to Bible-AI! This guide will help you get up and running with our intelligent scripture study and theological assistant platform.

## Installation

### Prerequisites
Before installing Bible-AI, ensure you have:
- Python 3.12 or higher
- Node.js 18 or higher
- Minimum 8GB of RAM (16GB recommended)
- (Optional) GPU for faster inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/bible-ai.git
cd bible-ai
```

### Step 2: Set Up the Backend
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Up the Frontend
```bash
cd frontend
npm install
cd ..
```

### Step 4: Download Bible Translations
```bash
python cli.py download-bible --versions NIV,ESV,KJV
```

## Running Bible-AI

### Start the Backend Server
```bash
python app.py
```

### Start the Frontend Application
In a separate terminal:
```bash
cd frontend
npm start
```

Access the application at `http://localhost:3000` in your web browser.

## First Steps

Once you have Bible-AI running, here are some things you can do:

### 1. Explore Bible Passages
- Enter a verse reference (e.g., "John 3:16") to view it in multiple translations
- Click on words to see their original Hebrew/Greek meanings
- Use the context panel to understand historical background

### 2. Ask Theological Questions
- Type natural language questions about the Bible, theology, or specific passages
- For example: "What does the Bible say about forgiveness?"
- Review multiple perspectives when available

### 3. Study a Topic
- Use the topic study feature to explore biblical themes
- For example: "Study the concept of grace in the New Testament"
- Review cross-references and related passages

### 4. Customize Your Experience
- Set your preferred Bible translation
- Select your denominational preference for theologically sensitive topics
- Adjust the depth of theological responses (basic to advanced)

## Tips for Best Results

- Be specific in your questions
- Provide context when asking about specific verses
- Use the "perspective" toggle to see different theological viewpoints
- Always verify information through personal study and prayer

Remember that Bible-AI is designed to assist your study of Scripture, not replace personal reflection, prayer, or pastoral guidance.
