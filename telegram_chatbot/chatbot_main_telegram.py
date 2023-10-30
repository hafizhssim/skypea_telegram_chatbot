import sys
import time
import atexit
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr  # Import the SpeechRecognition library
from pydub import AudioSegment  # Import pydub

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming the path and import are correct
sys.path.append('FOLDER_PATH\skypea_telegram_chatbot\skypea_telegram')
from chatbot_main import gpt

# Initialize the gpt instance
gpt_instance = gpt()

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor()

# Shutdown hook
def shutdown():
    executor.shutdown()
    logger.info("Executor shut down")

atexit.register(shutdown)

def handle_message(update, context, text_query=None):
    chat_id = update.message.chat_id
    query = str(text_query) if text_query else update.message.text
    context.bot.send_message(chat_id=chat_id, text=f'{text_query}') if text_query else update.message.text
    try:
        logger.info("Received message")
        
        # query = update.message.text
        
        current_user_query_embedding = gpt_instance.generate_current_query_embedding(query)

        gpt_instance.initialize_knn_model()

        memory, internet_query = gpt_instance.fetch_and_clean_from_db(current_user_query_embedding, query)
        raw_db_response = memory
        # context.bot.send_message(chat_id=chat_id, text="Thinking...")
        context.bot.send_message(chat_id=chat_id, text=f' i need to search the internet... {internet_query}')
        context.bot.send_message(chat_id=chat_id, text=f' thinking... {memory}')
            
        sorted_links = gpt_instance.internet_route(current_user_query_embedding, query, internet_query)
        print(sorted_links)
        raw_internet_response = sorted_links
        print(raw_internet_response)
        context.bot.send_message(chat_id=chat_id, text="Reading on internet...")

        db_response, internet_response, combined_response = gpt_instance.combined_pipelines(raw_db_response, raw_internet_response, query)

        final_response = gpt_instance.generate_final_response(db_response, internet_response, query)
        context.bot.send_message(chat_id=chat_id, text=final_response)

        summary = gpt_instance.generate_conversation_summary(final_response, query)
        
        # STORING DATA
        gpt_instance.store_data_in_db(query, current_user_query_embedding, db_response, internet_response, final_response, summary)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        context.bot.send_message(chat_id=chat_id, text=f"Sorry, an error occurred: {e}")

# Function to convert OGG to WAV
def convert_ogg_to_wav(ogg_file_path, wav_file_path):
    audio = AudioSegment.from_ogg(ogg_file_path)
    audio.export(wav_file_path, format="wav")

# Function to convert audio to text using Google's Speech Recognition
def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            logger.error("Google's speech recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google's speech recognition service; {e}")
            return None

def handle_audio(update, context):
    chat_id = update.message.chat_id
    voice = context.bot.getFile(update.message.voice.file_id)
    
    # Download and save the file
    ogg_file_path = 'voice.ogg'
    wav_file_path = 'voice.wav'
    voice.download(ogg_file_path)
    
    # Convert OGG to WAV
    convert_ogg_to_wav(ogg_file_path, wav_file_path)
    
    # Convert audio to text
    text_query = convert_audio_to_text(wav_file_path)
    if text_query:
        logger.info(f"Converted text: {text_query}")
        handle_message(update, context, text_query)
    else:
        context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't understand the audio.")

# Main entry point
if __name__ == '__main__':
    logger.info("Bot is starting")
    updater = Updater(token='YOUR TOKEN HERE', use_context=True)
    dispatcher = updater.dispatcher

    message_handler = MessageHandler(Filters.text & ~Filters.command, handle_message)
    dispatcher.add_handler(message_handler)
    
    audio_handler = MessageHandler(Filters.voice, handle_audio)
    dispatcher.add_handler(audio_handler)
    
    updater.start_polling()
    logger.info("Bot is polling")

    # Add this to stop the bot gracefully on a KeyboardInterrupt
    try:
        updater.idle()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")