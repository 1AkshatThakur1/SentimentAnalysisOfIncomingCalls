import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

def analyze_sentiment(sentence):
    #compound sentiment score by nltk
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(sentence)
    compound_score = sentiment_scores['compound']

    #additional sentiment information by nrclex
    nrc_lex = NRCLex(sentence)
    emotion_scores = nrc_lex.raw_emotion_scores

    #sentiment labelling based on the compound score
    if compound_score > 0.5:
        sentiment_label = 'Highly Positive'
    elif compound_score < -0.5:
        sentiment_label = 'Highly Negative'
    elif compound_score > 0.25 and compound_score < 0.5 :
        sentiment_label = 'positive'
    elif compound_score < -0.25 and compound_score > -0.5 :
        sentiment_label = 'negative'
    else:
        sentiment_label = 'Neutral'
    
    return {
        'compound_score': compound_score,
        'sentiment_label': sentiment_label,
        'emotion_scores': emotion_scores
    }

def analyze_audio_from_microphone():
    #speech recognizer
    recognizer = sr.Recognizer()

    #default microphone as the audio source
    with sr.Microphone() as source:
        print("Speak something:")
        audio = recognizer.listen(source)

    try:
        #Google Web Speech API for transcribing
        text = recognizer.recognize_google(audio)
        print("You said: {}".format(text))

        # Analyze sentiment of the transcribed text
        result = analyze_sentiment(text)

        print(f"Sentence: {text}")
        print()
        print(f"Compound Score: {result['compound_score']}")
        print(f"Sentiment Label: {result['sentiment_label']}")
        print(f"Emotion Scores: {result['emotion_scores']}")
        print()

    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))

def analyze_audio_from_file(wave_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(wave_file_path) as source:
        print("Processing audio file:")
        print()
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said: {}".format(text))

        # Analyze sentiment of the transcribed text
        result = analyze_sentiment(text)

        print(f"Sentence: {text}")
        print()
        print(f"Compound Score: {result['compound_score']}")
        print(f"Sentiment Label: {result['sentiment_label']}")
        print(f"Emotion Scores: {result['emotion_scores']}")
        print()

    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))

def main():
    while True:
        print("1. Speak for audio input")
        print("2. Provide the path for a file for audio input")
        print("3. Exit")
        choice = input("Enter your choice: \n")

        if choice == '1':
            analyze_audio_from_microphone()
        elif choice == '2':
            wave_file_path = input("Enter the path of the audio file: ")
            analyze_audio_from_file(wave_file_path)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
