from flask import Flask, jsonify, request
import os
import json
import re

app = Flask(__name__)

DOWNLOAD_DIR = "downloads"
DATA_FILE = "kjv_processed.json"

# Load the Bible data
def load_bible_data():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Bible data: {e}")
        return {}

bible_data = load_bible_data()

@app.route('/')
def home():
    """
    Root endpoint to confirm the API is running.
    """
    return jsonify({
        "message": "Welcome to Bible AI backend service.",
        "endpoints": {
            "/api/v1/verse/{book}/{chapter}/{verse}": "Get a specific verse",
            "/api/v1/chapter/{book}/{chapter}": "Get an entire chapter",
            "/api/v1/search": "Search the Bible text",
            "/api/v1/random": "Get a random verse"
        }
    })

@app.route('/bibles', methods=['GET'])
def list_bibles():
    """
    Lists all downloaded Bible translations.
    """
    if not os.path.exists(DOWNLOAD_DIR):
        return jsonify({"error": "No Bibles downloaded yet."}), 404

    files = os.listdir(DOWNLOAD_DIR)
    bibles = [f.replace('.txt', '') for f in files if f.endswith('.txt')]
    return jsonify({"bibles": bibles})

@app.route('/bibles/<version>', methods=['GET'])
def get_bible(version):
    """
    Retrieves the content of a specific Bible translation.
    """
    file_path = os.path.join(DOWNLOAD_DIR, f"{version}.txt")
    if not os.path.exists(file_path):
        return jsonify({"error": f"Bible version '{version}' not found."}), 404

    with open(file_path, 'r') as f:
        content = f.read()
    return jsonify({"version": version, "content": content})

@app.route('/api/v1/verse/<book>/<int:chapter>/<int:verse>')
def get_verse(book, chapter, verse):
    """
    Retrieve a specific verse from the Bible.
    """
    try:
        if book in bible_data:
            chapter_data = bible_data[book].get(str(chapter))
            if chapter_data:
                verse_text = chapter_data.get(str(verse))
                if verse_text:
                    return jsonify({
                        "book": book,
                        "chapter": chapter,
                        "verse": verse,
                        "text": verse_text
                    })
        return jsonify({"error": "Verse not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/chapter/<book>/<int:chapter>')
def get_chapter(book, chapter):
    """
    Retrieve an entire chapter from the Bible.
    """
    try:
        if book in bible_data:
            chapter_data = bible_data[book].get(str(chapter))
            if chapter_data:
                return jsonify({
                    "book": book,
                    "chapter": chapter,
                    "verses": chapter_data
                })
        return jsonify({"error": "Chapter not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/search')
def search_bible():
    """
    Search the Bible text.
    Query parameters:
    - q: search query
    - book: optional book to limit search to
    """
    query = request.args.get('q', '').lower()
    book = request.args.get('book')
    
    if not query:
        return jsonify({"error": "Search query required"}), 400
    
    results = []
    try:
        books_to_search = [book] if book and book in bible_data else bible_data.keys()
        
        for book_name in books_to_search:
            book_data = bible_data[book_name]
            for chapter in book_data:
                for verse, text in book_data[chapter].items():
                    if query in text.lower():
                        results.append({
                            "book": book_name,
                            "chapter": int(chapter),
                            "verse": int(verse),
                            "text": text
                        })
        
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results[:50]  # Limit results to 50 matches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/books')
def list_books():
    """
    List all available books in the Bible.
    """
    try:
        books = list(bible_data.keys())
        return jsonify({
            "count": len(books),
            "books": books
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    app.run(debug=True)