"""
Database operations module for the Diacritical application.
Handles CRUD operations for diacritic mappings.
"""

import logging
from database import db, DiacriticMapping, Feedback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mappings_from_db() -> dict[str, str]:
    """Load mappings from the database"""
    mappings = {}
    try:
        db_mappings = DiacriticMapping.query.all()
        for mapping in db_mappings:
            mappings[mapping.plain_text] = mapping.diacritic_text
        logger.info(f"Loaded {len(mappings)} mappings from database")
    except Exception as e:
        logger.error(f"Error loading mappings from database: {e}")
    return mappings

def save_mapping_to_db(plain_text: str, diacritic_text: str) -> DiacriticMapping:
    """Save a new mapping to the database"""
    try:
        mapping = DiacriticMapping(plain_text=plain_text.lower(), diacritic_text=diacritic_text.lower())
        db.session.add(mapping)
        db.session.commit()
        logger.info(f"Saved mapping: {plain_text.lower()} -> {diacritic_text.lower()}")
        return mapping
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving mapping to database: {e}")
        raise

def update_mapping_in_db(mapping_id: int, plain_text: str, diacritic_text: str) -> DiacriticMapping:
    """Update an existing mapping in the database"""
    try:
        mapping = DiacriticMapping.query.get(mapping_id)
        if mapping:
            mapping.plain_text = plain_text.lower()
            mapping.diacritic_text = diacritic_text.lower()
            db.session.commit()
            logger.info(f"Updated mapping {mapping_id}: {plain_text.lower()} -> {diacritic_text.lower()}")
        return mapping
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating mapping in database: {e}")
        raise

def delete_mapping_from_db(mapping_id: int) -> bool:
    """Delete a mapping from the database"""
    try:
        mapping = DiacriticMapping.query.get(mapping_id)
        if mapping:
            db.session.delete(mapping)
            db.session.commit()
            logger.info(f"Deleted mapping {mapping_id}")
            return True
        return False
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting mapping from database: {e}")
        raise

def batch_delete_mappings_from_db(ids: list[int]) -> int:
    """Delete multiple mappings from the database
    
    Args:
        ids: List of mapping IDs to delete
        
    Returns:
        Number of mappings deleted
    """
    try:
        # Get the mappings to delete
        mappings = DiacriticMapping.query.filter(DiacriticMapping.id.in_(ids)).all()
        deleted_count = len(mappings)
        
        # Delete the mappings
        for mapping in mappings:
            db.session.delete(mapping)
        
        db.session.commit()
        logger.info(f"Batch deleted {deleted_count} mappings with IDs: {ids}")
        return deleted_count
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error batch deleting mappings: {e}")
        raise

def process_uploaded_mappings_file(file_path: str, mode: str = 'update') -> dict:
    """Process an uploaded mappings file
    
    Args:
        file_path: Path to the uploaded file
        mode: 'update' to add/update mappings, 'overwrite' to replace all mappings
        
    Returns:
        Dictionary with statistics about the operation
    """
    from diacritics import load_mappings_from_file
    
    try:
        # Load mappings from the uploaded file
        uploaded_mappings = load_mappings_from_file(file_path)
        logger.info(f"Loaded {len(uploaded_mappings)} mappings from uploaded file")
        
        stats = {
            'total': len(uploaded_mappings),
            'added': 0,
            'updated': 0,
            'unchanged': 0
        }
        
        # If overwrite mode, delete all existing mappings
        if mode == 'overwrite':
            try:
                # Delete all existing mappings
                DiacriticMapping.query.delete()
                db.session.commit()
                logger.info("Deleted all existing mappings for overwrite mode")
                
                # Add all mappings from the uploaded file
                mappings_to_add = []
                for plain_text, diacritic_text in uploaded_mappings.items():
                    mappings_to_add.append({
                        'plain_text': plain_text.lower(),
                        'diacritic_text': diacritic_text.lower()
                    })
                
                # Use bulk insert for better performance
                batch_size = 1000
                for i in range(0, len(mappings_to_add), batch_size):
                    batch = mappings_to_add[i:i+batch_size]
                    db.session.bulk_insert_mappings(DiacriticMapping, batch)
                    db.session.commit()
                
                stats['added'] = len(uploaded_mappings)
                logger.info(f"Added {stats['added']} mappings in overwrite mode")
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error in overwrite mode: {e}")
                raise
                
        # Update mode - add new mappings and update existing ones
        else:
            # Get existing mappings
            existing_mappings = {}
            for mapping in DiacriticMapping.query.all():
                existing_mappings[mapping.plain_text] = {
                    'id': mapping.id,
                    'diacritic_text': mapping.diacritic_text
                }
            
            # Process each mapping from the uploaded file
            for plain_text, diacritic_text in uploaded_mappings.items():
                plain_text = plain_text.lower()
                diacritic_text = diacritic_text.lower()
                if plain_text in existing_mappings:
                    # Check if the mapping has changed
                    if existing_mappings[plain_text]['diacritic_text'] != diacritic_text:
                        # Update the existing mapping
                        mapping = DiacriticMapping.query.get(existing_mappings[plain_text]['id'])
                        mapping.diacritic_text = diacritic_text
                        stats['updated'] += 1
                    else:
                        stats['unchanged'] += 1
                else:
                    # Add new mapping
                    mapping = DiacriticMapping(plain_text=plain_text, diacritic_text=diacritic_text)
                    db.session.add(mapping)
                    stats['added'] += 1
            
            db.session.commit()
            logger.info(f"Update mode: Added {stats['added']}, updated {stats['updated']}, unchanged {stats['unchanged']}")
        
        return stats
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error processing uploaded mappings file: {e}")
        raise

def save_feedback_to_db(message: str, email: str = None) -> Feedback:
    """Save a new feedback entry to the database"""
    try:
        feedback = Feedback(message=message, email=email)
        db.session.add(feedback)
        db.session.commit()
        logger.info(f"Saved feedback: {message[:30]}...")
        return feedback
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving feedback to database: {e}")
        raise

def get_all_feedback() -> list[Feedback]:
    """Get all feedback entries from the database, ordered by most recent first"""
    try:
        feedback_entries = Feedback.query.order_by(Feedback.created_at.desc()).all()
        logger.info(f"Retrieved {len(feedback_entries)} feedback entries")
        return feedback_entries
    except Exception as e:
        logger.error(f"Error retrieving feedback from database: {e}")
        return []

def delete_feedback_from_db(feedback_id: int) -> bool:
    """Delete a feedback entry from the database"""
    try:
        feedback = Feedback.query.get(feedback_id)
        if feedback:
            db.session.delete(feedback)
            db.session.commit()
            logger.info(f"Deleted feedback {feedback_id}")
            return True
        return False
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting feedback from database: {e}")
        return False 