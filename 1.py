import json
with open('kjv_structured_complete.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    books = len(data['books'])
    chapters = sum(len(book['chapters']) for book in data['books'])
    verses = sum(sum(len(chap['verses']) for chap in book['chapters']) for book in data['books'])
    print(f"Books: {books}, Chapters: {chapters}, Verses: {verses}")