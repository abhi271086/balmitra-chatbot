
# Multilingual Social Support Chatbot for Children

A supportive application that allows children from underprivileged backgrounds to interact with a conversational chatbot powered by LangChain and Groq. The application detects and responds in multiple languages (English, Hindi, Marathi, and Urdu) and is specifically designed to address social issues and challenges faced by children.

## Features

- **Multilingual Support**: Automatically detects and responds in English, Hindi, Marathi, and Urdu.

- **Child-Friendly Interface**: Simple conversational interface where children can ask questions about social issues.

- **Social Support Focus**: Provides guidance on education access, health, community support, bullying, family challenges, and basic rights awareness.

- **Contextual Responses**: Maintains a history of the conversation to provide context for the chatbot's responses.

- **LangChain + Groq Integration**: Powered by the LangChain framework and Groq's fast language models.

## How it Works

1. The user inputs a message in any of the supported languages
2. The system automatically detects the language
3. Translates the input to English for processing if needed
4. Generates a thoughtful, supportive response using the Groq language model
5. Translates the response back to the user's language
6. Continues the conversation with memory of previous exchanges

## Usage

You will need to store a valid Groq API Key as a secret to proceed with this example. You can generate one for free [here](https://console.groq.com/keys).

You can run this application on the command line with `python main.py`
