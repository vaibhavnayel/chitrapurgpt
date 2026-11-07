# Diacritical Translator

A web application for adding diacritics to input text based on a dictionary of mappings. The application uses Flask for the backend and PostgreSQL for storing the mappings.

## Features

- Translate plain text to text with diacritics
- Manage diacritic mappings through a web interface
- Add, edit, and delete mappings
- Sort and filter mappings
- Upload custom mapping files to update or overwrite the database
- Download all mappings as a text file

## Setup

### Local Development

1. Clone the repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the following variables:
   ```
   FLASK_SECRET_KEY=your_secret_key
   APP_USERNAME=your_username
   APP_PASSWORD=your_password
   DATABASE_URL=postgresql://username:password@localhost:5432/diacritical
   ```
5. Run the application:
   ```
   python app.py
   ```

### Database Migration (Developer Only)

To migrate mappings from the text file to the database:

1. Ensure your database is set up and the connection string is in your `.env` file
2. Run the migration script:
   ```
   python migrate_data.py
   ```

## Deployment on Render

### Database Setup

1. Create a new PostgreSQL database on Render
2. Note the Internal Database URL provided by Render

### Web Service Setup

1. Create a new Web Service on Render
2. Connect your repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add the following environment variables:
   - `FLASK_SECRET_KEY`: A secure random string
   - `APP_USERNAME`: Username for logging in
   - `APP_PASSWORD`: Password for logging in
   - `DATABASE_URL`: The Internal Database URL from your PostgreSQL database

### Initial Data Migration (Developer Only)

After deploying, you can migrate your mappings from the text file to the database by running:

```
python migrate_data.py
```

This is a developer-only operation and should be performed by administrators with direct access to the server.

## Usage

1. Log in with your username and password
2. On the translator page, enter text without diacritics
3. Click "Add Diacritics" to translate the text
4. Use the "Manage Mappings" page to add, edit, or delete mappings
5. Upload custom mapping files to update or overwrite the database
6. Download all mappings as a text file for backup or sharing

## Technical Details

- Backend: Flask 3.0.2
- Database: PostgreSQL with SQLAlchemy 2.0.27
- Frontend: Bootstrap with DataTables for the mappings table
- Authentication: Simple session-based authentication