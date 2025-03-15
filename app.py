from flask import Flask, jsonify, request
import os

app = Flask(__name__)

DOWNLOAD_DIR = "downloads"

@app.route('/')
def home():
    """
    Root endpoint to confirm the API is running.
    """
    return jsonify({"message": "Welcome to Bible AI backend service."})

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

if __name__ == '__main__':
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    app.run(debug=True)