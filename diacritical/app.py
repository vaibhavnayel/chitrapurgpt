"""
Main application module for the Diacritical application.
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_bootstrap import Bootstrap
from functools import wraps
import os
import tempfile
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime

# Import modules
from database import init_db, DiacriticMapping, Feedback
from diacritics import translate_text, load_mappings_from_file
from db_operations import (
    load_mappings_from_db, save_mapping_to_db, update_mapping_in_db,
    delete_mapping_from_db, batch_delete_mappings_from_db,
    process_uploaded_mappings_file,
    save_feedback_to_db, get_all_feedback, delete_feedback_from_db
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
bootstrap = Bootstrap(app)

# Database configuration
database_url = os.environ.get('DATABASE_URL')
# Fix for Render's postgres:// vs postgresql:// issue
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
init_db(app)

# Get configuration from environment variables
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
USERNAME = os.environ.get('APP_USERNAME')
PASSWORD = os.environ.get('APP_PASSWORD')

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('translator'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('translator'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('translator'))

@app.route('/translator', methods=['GET'])
def translator():
    return render_template('translator.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Get mappings from database
        mappings = load_mappings_from_db()
        # Translate text
        translated = translate_text(text, mappings)
        return jsonify({'result': translated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mappings', methods=['GET'])
@login_required
def mappings_page():
    return render_template('mappings.html')

# API endpoints for mappings
@app.route('/api/mappings', methods=['GET'])
@login_required
def get_mappings():
    mappings = DiacriticMapping.query.all()
    return jsonify([mapping.to_dict() for mapping in mappings])

@app.route('/api/mappings', methods=['POST'])
@login_required
def add_mapping():
    data = request.json
    plain_text = data.get('plain_text', '').strip().lower()
    diacritic_text = data.get('diacritic_text', '').strip().lower()
    
    if not plain_text or not diacritic_text:
        return jsonify({'error': 'Both plain text and diacritic text are required'}), 400
    
    # Check if mapping already exists
    existing = DiacriticMapping.query.filter_by(plain_text=plain_text).first()
    if existing:
        return jsonify({'error': f'Mapping for "{plain_text}" already exists'}), 400
    
    try:
        mapping = save_mapping_to_db(plain_text, diacritic_text)
        return jsonify(mapping.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mappings/<int:mapping_id>', methods=['PUT'])
@login_required
def update_mapping(mapping_id):
    data = request.json
    plain_text = data.get('plain_text', '').strip().lower()
    diacritic_text = data.get('diacritic_text', '').strip().lower()
    
    if not plain_text or not diacritic_text:
        return jsonify({'error': 'Both plain text and diacritic text are required'}), 400
    
    # Check if the new plain_text already exists for another mapping
    existing = DiacriticMapping.query.filter_by(plain_text=plain_text).first()
    if existing and existing.id != mapping_id:
        return jsonify({'error': f'Mapping for "{plain_text}" already exists'}), 400
    
    try:
        mapping = update_mapping_in_db(mapping_id, plain_text, diacritic_text)
        if mapping:
            return jsonify(mapping.to_dict())
        return jsonify({'error': 'Mapping not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mappings/<int:mapping_id>', methods=['DELETE'])
@login_required
def delete_mapping(mapping_id):
    try:
        success = delete_mapping_from_db(mapping_id)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Mapping not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mappings/batch-delete', methods=['POST'])
@login_required
def batch_delete_mappings():
    data = request.json
    ids = data.get('ids', [])
    
    if not ids:
        return jsonify({'error': 'No IDs provided'}), 400
    
    try:
        deleted_count = batch_delete_mappings_from_db(ids)
        return jsonify({'success': True, 'deleted_count': deleted_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mappings/upload', methods=['POST'])
@login_required
def upload_mappings():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.txt'):
        return jsonify({'error': 'Only .txt files are allowed'}), 400
    
    mode = request.form.get('mode', 'update')
    if mode not in ['update', 'overwrite']:
        return jsonify({'error': 'Invalid mode. Must be "update" or "overwrite"'}), 400
    
    try:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the file
        stats = process_uploaded_mappings_file(file_path, mode)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mappings/download', methods=['GET'])
@login_required
def download_mappings():
    try:
        # Get all mappings from the database
        mappings = DiacriticMapping.query.order_by(DiacriticMapping.plain_text).all()
        
        # Create a string with all mappings in the format plain_text,diacritic_text
        mappings_text = ""
        for mapping in mappings:
            mappings_text += f"{mapping.plain_text},{mapping.diacritic_text}\n"
        
        # Create a response with the mappings text
        response = app.response_class(
            response=mappings_text,
            status=200,
            mimetype='text/plain'
        )
        
        # Set the Content-Disposition header to make the browser download the file
        response.headers["Content-Disposition"] = "attachment; filename=mappings.txt"
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoints for feedback
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    message = data.get('message', '').strip()
    email = data.get('email', '').strip() or None
    
    if not message:
        return jsonify({'error': 'Feedback message is required'}), 400
    
    try:
        feedback = save_feedback_to_db(message, email)
        return jsonify(feedback.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['GET'])
@login_required
def get_feedback():
    try:
        feedback_entries = get_all_feedback()
        return jsonify([feedback.to_dict() for feedback in feedback_entries])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/<int:feedback_id>', methods=['DELETE'])
@login_required
def delete_feedback(feedback_id):
    try:
        success = delete_feedback_from_db(feedback_id)
        if success:
            return jsonify({'success': True})
        return jsonify({'error': 'Feedback not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/download', methods=['GET'])
@login_required
def download_feedback():
    try:
        # Get all feedback from the database
        feedback_entries = get_all_feedback()
        
        # Create a CSV string with all feedback
        csv_data = "Date,Message,Email\n"
        for feedback in feedback_entries:
            # Format the date
            date = datetime.fromisoformat(feedback.to_dict()['created_at'].replace('Z', '+00:00'))
            formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Escape commas and quotes in the message
            message = feedback.message.replace('"', '""')
            email = feedback.email or "Not provided"
            
            # Add the row to the CSV
            csv_data += f'"{formatted_date}","{message}","{email}"\n'
        
        # Create a response with the feedback CSV
        response = app.response_class(
            response=csv_data,
            status=200,
            mimetype='text/csv'
        )
        
        # Set the Content-Disposition header to make the browser download the file
        response.headers["Content-Disposition"] = "attachment; filename=feedback.csv"
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
