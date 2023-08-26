
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from vosk import Model, KaldiRecognizer  # offline recognition from Vosk
from googlesearch import search  # Google search
from pyowm import OWM  # Using OpenWeatherMap to retrieve weather data
from termcolor import colored  # To change terminal output colors
from dotenv import load_dotenv  # Loading information from an .env file
import speech_recognition  # Custom Speech Recognition (Speech-To-Text)
import googletrans  # Using the system Google Translate
import pyttsx3  # Speech synthesis (Text-To-Speech)
import wikipediaapi  # To find definitions in Wikipedia
import random  # Random Number Generator
import webbrowser  # Work using the default browser (opening tabs with a web page)
import traceback  # Traceback output without stopping the program when catching exceptions
import json  # Working with JSON files and JSON strings
import wave  # Create and read WAV audio files
import os  # Working with the file system
import json
from googletrans import Translator
from gtts import gTTS


class Translation:
    """
     Getting a string translation embedded in the app to create a multilingual assistant
    """
    with open("translations.json", "r", encoding="UTF-8") as file:
        translations = json.load(file)

    def get(self, text: str):
        """
        Getting a line feed from a file into the desired language (by its code)
        :param text: the text you want to translate
        :return: text translation embedded in the application
        """
        if text in self.translations:
            return self.translations[text][assistant.speech_language]
        else:
            # If there is no translation, a message about this is displayed in the logs and the source text is returned
            print(colored("Not translated phrase: {}".format(text), "red"))
            return text


class OwnerPerson:
    """
    Information about the owner, including name, city of residence, native language of speech, target language (for text translations)
    """
    name = ""
    home_city = "cairo"
    native_language = ""
    target_language = ""


class VoiceAssistant:
    """
    Voice assistant settings, including name, gender, speech language
    Note: for multilingual voice assistants, it is better to create a separate class,
    which will take the translation from the JSON file with the desired language
    """
    name = ""
    sex = ""
    speech_language = ""
    recognition_language = ""


def setup_assistant_voice():
    """
     Set the default voice (the index may vary depending on the operating system settings)
    """
    voices = ttsEngine.getProperty("voices")

    if assistant.speech_language == "en":
        assistant.recognition_language = "en"
        if assistant.sex == "female":
            # Microsoft Zira Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[1].id)
        else:
            # Microsoft David Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[2].id)
    '''else:
        assistant.recognition_language = "ru-RU"
        # Microsoft Irina Desktop - Russian
        ttsEngine.setProperty("voice", voices[0].id)'''


def record_and_recognize_audio(*args: tuple):
    """
	Audio Recording & Recognition
    """
    with microphone:
        recognized_data = ""

        # memorization of ambient noises for subsequent cleaning of sound from them
        recognizer.adjust_for_ambient_noise(microphone, duration=4)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            play_voice_assistant_speech(translator.get("Can you check if your microphone is on, please?"))
            traceback.print_exc()
            return

        # use of online recognition via Google (high quality recognition)
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language=assistant.recognition_language).lower()

        except speech_recognition.UnknownValueError:
            #pass
            play_voice_assistant_speech("What did you say again?")

        # in case of problems with Internet access, an attempt is made to use offline recognition via Vosk
        except speech_recognition.RequestError:
            print(colored("Trying to use offline recognition...", "cyan"))
            recognized_data = use_offline_recognition()

        return recognized_data


def use_offline_recognition():
    """
	Switch to Offline Speech Recognition
    :return: Recognized phrase
    """
    recognized_data = ""
    try:
        # Verify that the model in the desired language is present in the application directory
        if not os.path.exists("models/vosk-model-" + assistant.speech_language + "0.22"):
            print(colored("Please download the model from:\n"
                          "https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.",
                          "red"))
            exit(1)

        # analysis of audio recorded in the microphone (to avoid repetition of the phrase)
        wave_audio_file = wave.open("microphone-results.wav", "rb")
        model = Model("models/vosk-model-" + assistant.speech_language + "0.22")
        offline_recognizer = KaldiRecognizer(model, wave_audio_file.getframerate())

        data = wave_audio_file.readframes(wave_audio_file.getnframes())
        if len(data) > 0:
            if offline_recognizer.AcceptWaveform(data):
                recognized_data = offline_recognizer.Result()

                # retrieving recognized text data from a JSON string (so that you can respond to it)
                recognized_data = json.loads(recognized_data)
                recognized_data = recognized_data["text"]
    except:
        traceback.print_exc()
        print(colored("Sorry, speech service is unavailable. Try again later", "red"))

    return recognized_data


def play_voice_assistant_speech(text_to_speech):
    """
    Voice assistant speech playback (without saving audio)
    :param text_to_speech: text to be converted to speech
    """
    ttsEngine.say(str(text_to_speech))
    ttsEngine.runAndWait()

def play_failure_phrase(*args: tuple):
    """
    Playing a random phrase in case of unsuccessful recognition
    """
    failure_phrases = [
        translator.get("Can you repeat, please?"),
        translator.get("What did you say again?")
    ]
    play_voice_assistant_speech(failure_phrases[random.randint(0, len(failure_phrases) - 1)])
    
def play_greetings(*args: tuple):
    """
    Playing a random welcome speech
    """
    greetings = [
        translator.get("Hello, {}! How can I help you today?").format(person.name),
        translator.get("Good day to you {}! How can I help you today?").format(person.name)
    ]
    play_voice_assistant_speech(greetings[random.randint(0, len(greetings) - 1)])


def play_farewell_and_quit(*args: tuple):
    """
    Playing a farewell speech and exiting
    """
    farewells = [
        translator.get("Goodbye, {}! Have a nice day!").format(person.name),
        translator.get("See you soon, {}!").format(person.name)
    ]
    play_voice_assistant_speech(farewells[random.randint(0, len(farewells) - 1)])
    ttsEngine.stop()
    quit()


def search_for_term_on_google(*args: tuple):
    """
    Google search with automatic opening of links (to the list of results and to the results themselves, if possible)
    :param args: search query phrase
    """
    if not args[0]: return
    search_term = " ".join(args[0])

    # Opening a link to a search engine in a browser
    url = "https://google.com/search?q=" + search_term
    webbrowser.get().open(url)

    # Alternate search with automatic opening of links to results (may not be safe in some cases)
    search_results = []
    try:
        for _ in search(search_term,  # What to look for
                        tld="com",  # Top-level domain
                        lang=assistant.speech_language,  # The language spoken by the assistant is used
                        num=1,  # Number of results per page
                        start=0,  # The index of the first result to retrieve
                        stop=1,  # index of the last result retrieved (I want the first result to open)
                        pause=1.0,  # latency between HTTP requests
                        ):
            search_results.append(_)
            webbrowser.get().open(_)

    # Since it is difficult to predict all errors, it will be caught and then withdrawn without stopping the program
    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    print(search_results)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on google").format(search_term))


def search_for_video_on_youtube(*args: tuple):
    """
    Search for a YouTube video with an automatic link to the list of results
    :param args: search query phrase
    """
    if not args[0]: return
    search_term = " ".join(args[0])
    url = "https://www.youtube.com/results?search_query=" + search_term
    webbrowser.get().open(url)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on youtube").format(search_term))


def search_for_definition_on_wikipedia(*args: tuple):
    """
    Search Wikipedia for definitions, then voice the results and open links
    :param args: search query phrase
    """
    if not args[0]: return

    search_term = " ".join(args[0])

    # setting the language (in this case, the language spoken by the assistant is used)
    wiki = wikipediaapi.Wikipedia(assistant.speech_language)

    # Search for a page by query, read summary, open a link to a page for details
    wiki_page = wiki.page(search_term)
    try:
        if wiki_page.exists():
            play_voice_assistant_speech(translator.get("Here is what I found for {} on Wikipedia").format(search_term))
            webbrowser.get().open(wiki_page.fullurl)

            # the assistant reading the first two sentences of the summary from the Wikipedia page
            # (there may be problems with multilingualism)
            play_voice_assistant_speech(wiki_page.summary.split(".")[:2])
        else:
            # opening a link to a search engine in the browser if nothing could be found on Wikipedia on request
            play_voice_assistant_speech(translator.get(
                "Can't find {} on Wikipedia. But here is what I found on google").format(search_term))
            url = "https://google.com/search?q=" + search_term
            webbrowser.get().open(url)

    # Since it is difficult to predict all errors, it will be caught and then withdrawn without stopping the program
    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

"""def arabic(*args: tuple):

	translator = Translator()

	with open(data_json_path, 'r') as f:
    	data_json = json.load(f)

	english_as_list = [sample['text'] for sample in data_json]
	arabic = [translator.translate(sample, src='en', dest='ar').text for sample in english_as_list]

	for translation in arabic:
    	tts = gTTS(text=translation, lang='ar')
    	tts.save("translation.mp3")
    	os.system("mpg321 translation.mp3")  # Replace with the appropriate command to play the audio file"""


def get_translation(*args: tuple):
    """
    Obtaining a translation of a text from one language to another (in this case, from the target to the native language or vice versa)
    :param args: the phrase you want to translate
    """
    if not args[0]: return

    search_term = " ".join(args[0])
    google_translator = googletrans.Translator()
    translation_result = ""

    old_assistant_language = assistant.speech_language
    try:
        # If the language of speech of the assistant and the native language of the user are different, then the translation is performed in the native language
        if assistant.speech_language != person.native_language:
            translation_result = google_translator.translate(search_term,  # What to translate
                                                      src=person.target_language,  # From which language
                                                      dest=person.native_language)  # in what language

            play_voice_assistant_speech("The translation for {} in Russian is".format(search_term))

            # change the voice of the assistant to the user's native language (so that you can pronounce the translation)
            assistant.speech_language = person.native_language
            setup_assistant_voice()

        # If the language of speech of the assistant and the native language of the user are the same, then the translation is performed into the target language
        else:
            translation_result = google_translator.translate(search_term,  # What to translate
                                                      src=person.native_language,  # From which language
                                                      dest=person.target_language)  # in what language
            play_voice_assistant_speech("In English, {} will be like".format(search_term))

            # changing the voice of the assistant to the user's language being studied (so that you can pronounce the translation)
            assistant.speech_language = person.target_language
            setup_assistant_voice()

        # Translation pronunciation
        play_voice_assistant_speech(translation_result.text)

    # Since it is difficult to predict all errors, it will be caught and then withdrawn without stopping the program
    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()

    finally:
        # Revert to the previous assistant voice settings
        assistant.speech_language = old_assistant_language
        setup_assistant_voice()


def get_weather_forecast(*args: tuple):
    """
    Receiving and voicing the weather forecast
    :param args: the city in which the zap should be performed
    """
    # if there is an additional argument, the weather request is made by it,
    # otherwise - the city specified in the settings is used
    if args[0]:
        city_name = args[0][0]
    else:
        city_name = person.home_city

    try:
        # using the API key placed in the .env file following the example WEATHER_API_KEY = "01234abcd....."
        weather_api_key = ("419fc7e424d4f813f69948a5b0e68b68")
        open_weather_map = OWM(weather_api_key)

        # Request data on the current state of the weather
        weather_manager = open_weather_map.weather_manager()
        observation = weather_manager.weather_at_place(city_name)
        weather = observation.weather

    # Since it is difficult to predict all errors, it will be caught and then withdrawn without stopping the program
    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    # Splitting data into parts for easy working with them
    status = weather.detailed_status
    temperature = weather.temperature('celsius')["temp"]
    wind_speed = weather.wind()["speed"]
    pressure = int(weather.pressure["press"] / 1.333)  # converted from hPa to mmHg.

# Log output
    print(colored("Weather in " + city_name +
                  ":\n * Status: " + status +
                  "\n * Wind speed (m/sec): " + str(wind_speed) +
                  "\n * Temperature (Celsius): " + str(temperature) +
                  "\n * Pressure (mm Hg): " + str(pressure), "yellow"))

    # sounding the current state of the weather by an assistant (here additional work is required for multilingualism)
    play_voice_assistant_speech(translator.get("It is {0} in {1}").format(status, city_name))
    play_voice_assistant_speech(translator.get("The temperature is {} degrees Celsius").format(str(temperature)))
    play_voice_assistant_speech(translator.get("The wind speed is {} meters per second").format(str(wind_speed)))
    play_voice_assistant_speech(translator.get("The pressure is {} mm Hg").format(str(pressure)))


def change_language(*args: tuple):
    """
    Change the language of the voice assistant (speech recognition language)
    """
    assistant.speech_language = "en" if assistant.speech_language == "en" else "en"
    setup_assistant_voice()
    print(colored("Language switched to " + assistant.speech_language, "cyan"))


def run_person_through_social_nets_databases(*args: tuple):
    """
    Search for a person in the database of social networks VKontakte and Facebook
    :param args: first name, last name TODO city
    """
    if not args[0]: return

    google_search_term = " ".join(args[0])
    vk_search_term = "_".join(args[0])
    fb_search_term = "-".join(args[0])

    # Opening a link to a search engine in a browser
    url = "https://google.com/search?q=" + google_search_term + " site: vk.com"
    webbrowser.get().open(url)

    url = "https://google.com/search?q=" + google_search_term + " site: facebook.com"
    webbrowser.get().open(url)

    # Opening links to social media search engines in your browser
    vk_url = "https://vk.com/people/" + vk_search_term
    webbrowser.get().open(vk_url)

    fb_url = "https://www.facebook.com/public/" + fb_search_term
    webbrowser.get().open(fb_url)

    play_voice_assistant_speech(translator.get("Here is what I found for {} on social nets").format(google_search_term))


def toss_coin(*args: tuple):
    """
    "Toss a coin to choose from 2 options
    """
    flips_count, heads, tails = 3, 0, 0

    for flip in range(flips_count):
        if random.randint(0, 1) == 0:
            heads += 1

    tails = flips_count - heads
    winner = "Tails" if tails > heads else "Heads"
    play_voice_assistant_speech(translator.get(winner) + " " + translator.get("won"))


#def execute_command_with_name(command_name: str, *args: list):
#    """
#    Execute a user-defined command and arguments
#    :param command_name: team name
#    :param args: the arguments that will be passed to the
#    :return:
#    """
#    for key in commands.keys():
#        if comsmand_name in key:
#            commands[key](*args)
#        else:
#            pass  # print("Command not found")

def open_application(input):
 
    if "Edge" in input:
        assistant_speaks("Opening Microsot Edge")
        os.startfile('/opt/microsoft/msedge-beta/msedge.exe')
        return
 
   #elif "Edge" in input:
    #    assistant_speaks("Opening Microsot Edge")
     #   os.startfile('/opt/microsoft/msedge-beta/msedge.exe')
      #  return
 
    elif "Sudoku" in input:
        assistant_speaks("Opening Sudoku Game")
        os.startfile('/usr/games/gnome-sudoku.exe')
        return
 
    #elif "excel" in input:
     #   assistant_speaks("Opening Microsoft Excel")
      #  os.startfile('C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Microsoft Office 2013\\Excel 2013.lnk')
       # return'''
 
    else:
 
        assistant_speaks("Application not available")
        return
        
def search_web(input):

	driver = webdriver.Firefox()
	driver.implicitly_wait(1)
	driver.maximize_window()

	if 'youtube' in input.lower():

		assistant_speaks("Opening in youtube")
		indx = input.lower().split().index('youtube')
		query = input.split()[indx + 1:]
		driver.get("http://www.youtube.com/results?search_query =" + '+'.join(query))
		return

	elif 'wikipedia' in input.lower():

		assistant_speaks("Opening Wikipedia")
		indx = input.lower().split().index('wikipedia')
		query = input.split()[indx + 1:]
		driver.get("https://en.wikipedia.org/wiki/" + '_'.join(query))
		return

	else:

		if 'google' in input:

			indx = input.lower().split().index('google')
			query = input.split()[indx + 1:]
			driver.get("https://www.google.com/search?q =" + '+'.join(query))

		elif 'search' in input:

			indx = input.lower().split().index('google')
			query = input.split()[indx + 1:]
			driver.get("https://www.google.com/search?q =" + '+'.join(query))

		else:

			driver.get("https://www.google.com/search?q =" + '+'.join(input.split()))

		return
        
# list of commands to use (the tuple hashable type is used as dictionary keys)
# Alternatively, you can use a JSON object with intents and scripts
# (similar to those used for chatbots)
config = {
    "intents": {
        "greeting": {
            "examples": ["hello", "hi", "morning", "welmo",
                         "good morning"],
            "responses": play_greetings
        },
        "farewell": {
            "examples": ["bye", "goodbye", "quit", "exit", "stop", "sleep"],
            "responses": play_farewell_and_quit
        },
        "google_search": {
            "examples": ["search", "google", "find", "get from google", "search google for"],
            "responses": search_for_term_on_google
        },
        "youtube_search": {
            "examples": ["video", "youtube", "watch", "search youtube for"],
            "responses": search_for_video_on_youtube
        },
        "wikipedia_search": {
            "examples": ["wikipedia", "definition", "about", "define", "search wikipedia for", "tell about"],
            "responses": search_for_definition_on_wikipedia
        },
        "person_search": {
            "examples": ["find on facebook", " find person", "run person", "search for person"],
            "responses": run_person_through_social_nets_databases
        },
        "weather_forecast": {
            "examples": ["weather", "forecast", "today's temprature",
                         "weather forecast", "report weather"],
            "responses": get_weather_forecast
        },
        "translation": {
            "examples": ["language", "change languge",
                         "translate", "find translation"],
            "responses": get_translation
        },
        "language": {
            "examples": [
                         "change speech language", "language"],
            "responses": change_language
        },
        "toss_coin": {
            "examples": ["toss", "coin", "play coin",
                         "toss coin", "coin", "flip a coin"],
            "responses": toss_coin
        }
    },

    "failure_phrases": play_failure_phrase
}



def prepare_corpus():
    """
Prepare the model to guess the user's intent
    """
    corpus = []
    target_vector = []
    for intent_name, intent_data in config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            target_vector.append(intent_name)

    training_vector = vectorizer.fit_transform(corpus)
    classifier_probability.fit(training_vector, target_vector)
    classifier.fit(training_vector, target_vector)


def get_intent(request):
    """
    Get the most likely intent based on the user's request
    :param request: user request
    :return: the most likely intention
    """
    best_intent = classifier.predict(vectorizer.transform([request]))[0]

    index_of_best_intent = list(classifier_probability.classes_).index(best_intent)
    probabilities = classifier_probability.predict_proba(vectorizer.transform([request]))[0]

    best_intent_probability = probabilities[index_of_best_intent]

    # When adding new intents, it is worth reducing this indicator
    print(best_intent_probability)
    if best_intent_probability > 0.157:
        return best_intent


def make_preparations():
    """
        Prepare global variables to run the application
    """
    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, classifier

     # Initialize speech recognition and input tools
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    # Initialize the speech synthesis tool
    ttsEngine = pyttsx3.init()

    # Setting up user data
    person = OwnerPerson()
    person.name = "Nada"
    person.home_city = "cairo"
    person.native_language = "en"
    person.target_language = "en"

    # Setting up voice assistant data
    assistant = VoiceAssistant()
    assistant.name = "Welmo"
    assistant.sex = "male"
    assistant.speech_language = "en"
    
    # Set the default voice
    setup_assistant_voice()

    # adding the ability to translate phrases (from a prepared file)
    translator = Translation()

    # loading information from the .env file (there is an API key for OpenWeatherMap)
    load_dotenv()

    # Preparing the corpus to recognize user requests with some probability (search for similar ones)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()


if __name__ == "__main__":
    make_preparations()

    while True:
        # Start recording speech, followed by the output of recognized speech and delete the audio recorded in the microphone
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        # separation of commands from additional information (arguments)
        if voice_input:
            voice_input_parts = voice_input.split(" ")

            # If one word was said, execute the command immediately without additional arguments
            if len(voice_input_parts) == 1:
                intent = get_intent(voice_input)
                if intent:
                    config["intents"][intent]["responses"]()
                else:
                    config["failure_phrases"]()

            # in the case of a long phrase - searches for a key phrase and arguments through each word,
            # until a match is found
            if len(voice_input_parts) > 1:
                for guess in range(len(voice_input_parts)):
                    intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    print(intent)
                    if intent:
                        command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        print(command_options)
                        config["intents"][intent]["responses"](*command_options)
                        break
                    if not intent and guess == len(voice_input_parts)-1:
                        config["failure_phrases"]()

