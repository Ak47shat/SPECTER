from .config import settings
from .logger import logger
from deep_translator import GoogleTranslator

# Initialize translators
hindi_translator = GoogleTranslator(source='en', target='hi')
english_translator = GoogleTranslator(source='hi', target='en')

def hinglish_converter(text: str, target_language: str = 'hindi') -> str:
    """
    Convert text between English and Hinglish (Hindi written in English).
    
    Args:
        text (str): The text to convert
        target_language (str): Target language ('hindi' or 'english')
        
    Returns:
        str: Converted text
    """
    try:
        if target_language.lower() == 'hindi':
            # Convert English to Hindi
            return hindi_translator.translate(text)
        else:
            # Convert Hindi to English
            return english_translator.translate(text)
    except Exception as e:
        logger.error(f"Error in hinglish conversion: {str(e)}", exc_info=True)
        return text  # Return original text if translation fails

def format_response_for_gradio(response: str, max_length: int = 2000) -> str:
    """
    Format response text for Gradio display, ensuring it's readable and well-formatted.
    
    Args:
        response (str): The response text to format
        max_length (int): Maximum length for the response
        
    Returns:
        str: Formatted response text
    """
    try:
        # Truncate if too long
        if len(response) > max_length:
            truncated = response[:max_length]
            last_period = truncated.rfind('.')
            if last_period > 0:
                response = truncated[:last_period + 1]
            else:
                response = truncated + "..."
        
        # Ensure proper spacing
        response = response.replace('\n\n', '\n').strip()
        
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}", exc_info=True)
        return response 