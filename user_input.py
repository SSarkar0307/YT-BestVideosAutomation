import speech_recognition as sr
import sys

def get_user_query() -> str:
    """Prompt user to choose text or voice input and return the search query."""
    print("How would you like to enter your search query?")
    print("1. Type the query (Text)")
    print("2. Speak the query (Voice, English or Hindi)")
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if choice == '1':
        query = input("Enter your search query (e.g., 'cooking videos'): ").strip()
        if not query:
            print("Error: Query cannot be empty.")
            sys.exit(1)
        return query
    
    # Voice input
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Speak your query now (in English or Hindi)...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            # Try Hindi first, then fall back to English
            try:
                query = recognizer.recognize_google(audio, language='hi-IN')
                print(f"Recognized (Hindi): {query}")
            except sr.UnknownValueError:
                query = recognizer.recognize_google(audio, language='en-US')
                print(f"Recognized (English): {query}")
            except sr.RequestError as e:
                print(f"Error with speech recognition: {e}")
                sys.exit(1)
            if not query:
                print("Error: No query recognized.")
                sys.exit(1)
            return query
        except sr.WaitTimeoutError:
            print("Error: No speech detected within timeout.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # For testing
    query = get_user_query()
    print(f"Final query: {query}")