# Ai_emotional_support_robo_pet

https://github.com/Inguzz15/Ai_emotional_support_robo_pet/assets/107085915/129ef35d-6732-41b1-95c5-cf3070d7a24a

"""
The assistant is able to:
* recognize and synthesize speech in offline mode (without access to the Internet);
* report on the weather forecast anywhere in the world;
* make a search query in the Google search engine
  (as well as open the list of results and the results of this query);
search for a video in the YouTube system and open a list of the results of this query;
* search for the definition in Wikipedia with further reading of the first two sentences;
* search for a person by name and surname in social networks VKontakte and Facebook;
* "flip a coin";
* play an accidental goodbye followed by the termination of the program;
* change the settings of the language of speech recognition and synthesis;

The voice assistant uses the built-in features of the Windows 10 operating system for speech synthesis
(i.e. votes vary by operating system). To do this, use the pyttsx3 library


For the correct operation of the speech recognition system in combination with the SpeechRecognition library
the PyAudio library is used to retrieve audio from the microphone.

To install PyAudio, you can find and download the wphl file you need depending on the architecture and version of Python here:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio


After downloading the file to the project folder, you can start the installation using the following command:
pip install PyAudio-0.2.11-cp38-cp38m-win_amd64.whl

To use SpeechRecognition in offline mode (without access to the Internet), you will need to additionally install
vosk, the .whl file for which can be found here depending on the required architecture and version of Python:
https://github.com/alphacep/vosk-api/releases/

After downloading the file to the project folder, you can start the installation using the following command:
pip install vosk-0.3.7-cp38-cp38-win_amd64.whl

To obtain weather forecast data, I used the OpenWeatherMap service, which requires an API key.
You can get an API key and read the documentation after registration (there is a free plan) here:
https://openweathermap.org/

Commands for installing libraries:
pip install google
pip install SpeechRecognition
pip install pyttsx3
pip install wikipedia-api
pip install googletrans
pip install python-dotenv
pip install pyowm


To quickly install all the required dependencies, you can use the command:
pip install requirements.txt

More information on installing and using the libraries can be found here:
https://pypi.org/
"""
