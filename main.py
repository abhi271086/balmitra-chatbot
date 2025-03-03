import os

from googletrans import Translator
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.utils.secret import SecretStr
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq
from langdetect import detect


def detect_language(text):
    """
    Detect the language of the input text.
    Currently supports English, Hindi, Marathi, and Urdu.
    """
    try:
        lang_code = detect(text)
        if lang_code == 'en':
            return 'english'
        elif lang_code == 'hi':
            return 'hindi'
        elif lang_code == 'mr':
            return 'marathi'
        elif lang_code == 'ur':
            return 'urdu'
        else:
            return 'english'  # Default to English if unsupported language
    except:
        return 'english'  # Default to English if detection fails


def translate_to_english(text, source_language):
    """
    Translate text to English for processing
    """
    if source_language == 'english':
        return text

    translator = Translator()
    try:
        translation = translator.translate(text,
                                           src=source_language[:2],
                                           dest='en')
        return translation.text
    except:
        return text  # Return original if translation fails


def translate_from_english(text, target_language):
    """
    Translate response from English to target language
    """
    if target_language == 'english':
        return text

    lang_code = {
        'hindi': 'hi',
        'marathi': 'mr',
        'urdu': 'ur'
    }.get(target_language, 'en')
    translator = Translator()
    try:
        translation = translator.translate(text, src='en', dest=lang_code)
        return translation.text
    except:
        return text  # Return English if translation fails


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the interface, 
    and handles the chat interaction with language detection and social support features.
    """

    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama3-8b-8192'
    # Initialize Groq Langchain chat object and conversation
    # Initialize ChatGroq with the correct parameters for the current version
    groq_chat = ChatGroq(api_key=SecretStr(groq_api_key), model=model)

    print(
        "Hello! I'm your friendly multilingual chatbot. I can help children with questions about social issues and provide support. I understand English, Hindi, Marathi, and Urdu!"
    )
    print(
        "नमस्ते! मैं आपका दोस्ताना बहुभाषी चैटबॉट हूँ। मैं बच्चों को सामाजिक मुद्दों के बारे में सवालों के साथ मदद कर सकता हूँ और समर्थन प्रदान कर सकता हूँ। मैं अंग्रेज़ी, हिंदी, मराठी और उर्दू समझता हूँ!"
    )
    print(
        "नमस्कार! मी आपला मैत्रीपूर्ण बहुभाषिक चॅटबॉट आहे. मी मुलांना सामाजिक समस्यांविषयी प्रश्नांना मदत करू शकतो आणि समर्थन देऊ शकतो. मला इंग्रजी, हिंदी, मराठी आणि उर्दू समजते!"
    )
    print(
        "السلام علیکم! میں آپ کا دوستانہ کثیر لسانی چیٹ بوٹ ہوں۔ میں بچوں کو سماجی مسائل کے بارے میں سوالات میں مدد کر سکتا ہوں اور حمایت فراہم کر سکتا ہوں۔ مجھے انگریزی، ہندی، مراٹھی اور اردو سمجھ آتی ہے!"
    )

    system_prompt = '''You are a friendly conversational chatbot specifically designed to help children from underprivileged backgrounds. 
    You provide supportive, age-appropriate responses about social issues including but not limited to:
    - Education access and resources
    - Health and nutrition
    - Community support
    - Bullying and interpersonal problems
    - Family challenges
    - Basic rights awareness
    
    Keep your answers simple, supportive, and encouraging. Provide practical guidance when possible.
    Always maintain a positive, hopeful tone. If the child appears to be in danger or needs immediate help,
    suggest they talk to a trusted adult, teacher, or community worker.
    '''

    conversational_memory_length = 5  # number of previous messages the chatbot will remember during the conversation

    memory = ConversationBufferWindowMemory(k=conversational_memory_length,
                                            memory_key="chat_history",
                                            return_messages=True)

    last_language = 'english'  # Default language

    while True:
        user_question = input("Ask a question: ")

        # If the user has asked a question,
        if user_question:
            # Detect language
            detected_language = detect_language(user_question)
            last_language = detected_language

            # Translate to English if needed
            english_question = translate_to_english(user_question,
                                                    detected_language)

            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(
                    content=system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.
                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ])

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=
                groq_chat,  # The Groq LangChain chat object initialized earlier.
                prompt=prompt,  # The constructed prompt template.
                verbose=
                False,  # TRUE Enables verbose output, which can be useful for debugging.
                memory=
                memory,  # The conversational memory object that stores and manages the conversation history.
            )

            # The chatbot's answer is generated by sending the full prompt to the Groq API.
            english_response = conversation.predict(
                human_input=english_question)

            # Translate the response back to the detected language
            final_response = translate_from_english(english_response,
                                                    detected_language)

            print("Chatbot:", final_response)


if __name__ == "__main__":
    main()
