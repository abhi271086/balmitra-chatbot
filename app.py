
import os
import streamlit as st
import asyncio
from main import translate_to_english, translate_from_english

# Set page configuration
st.set_page_config(
    page_title="BalMitra - Children's Support Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "english"
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Import groq and other dependencies inside the app to avoid import errors
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq

# Header with title followed by logo
st.title("BalMitra - Support Chatbot for Children")
st.caption("A friendly companion to help with your questions")

# Display centered and larger logo below the title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=600, use_column_width=True)  # Increased width from 400 to 600

# Language selection
language_options = {
    "english": "English",
    "hindi": "Hindi / हिंदी",
    "marathi": "Marathi / मराठी",
    "urdu": "Urdu / اردو"
}

def get_greeting(language):
    """Return a greeting message in the selected language"""
    greetings = {
        'english': "Hello! I'm your friendly chatbot. I can help children with questions about social issues and provide support.",
        'hindi': "नमस्ते! मैं आपका दोस्ताना चैटबॉट हूँ। मैं बच्चों को सामाजिक मुद्दों के बारे में सवालों के साथ मदद कर सकता हूँ और समर्थन प्रदान कर सकता हूँ।",
        'marathi': "नमस्कार! मी आपला मैत्रीपूर्ण चॅटबॉट आहे. मी मुलांना सामाजिक समस्यांविषयी प्रश्नांना मदत करू शकतो आणि समर्थन देऊ शकतो.",
        'urdu': "السلام علیکم! میں آپ کا دوستانہ چیٹ بوٹ ہوں۔ میں بچوں کو سماجی مسائل کے بارے میں سوالات میں مدد کر سکتا ہوں اور حمایت فراہم کر سکتا ہوں۔"
    }
    return greetings.get(language, greetings['english'])

# Sidebar for language selection
with st.sidebar:
    st.header("Language Settings")
    selected_language = st.selectbox(
        "Choose your language:",
        options=list(language_options.keys()),
        format_func=lambda x: language_options[x],
        index=list(language_options.keys()).index(st.session_state.selected_language)
    )
    
    # Update the session state if language changed
    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.session_state.conversation_history = []  # Reset conversation when language changes
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot provides supportive conversations for children about:
    - Education access
    - Health and nutrition
    - Community support
    - Bullying
    - Family challenges
    - Basic rights awareness
    """)

# Initialize chatbot
@st.cache_resource
def init_chatbot():
    # Get Groq API key
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it up in the Secrets tab.")
        st.stop()
    
    model = 'llama3-8b-8192'
    groq_chat = ChatGroq(api_key=groq_api_key, model=model)
    
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
    
    memory = ConversationBufferWindowMemory(
        k=5,  # remember last 5 exchanges
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])
    
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    
    return conversation

# Display greeting
st.markdown(f"#### {get_greeting(st.session_state.selected_language)}")

# Initialize the chatbot
try:
    conversation = init_chatbot()
    st.session_state.initialized = True
except Exception as e:
    st.error(f"Error initializing chatbot: {str(e)}")
    st.session_state.initialized = False

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.initialized:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Process the message
            async def process_message():
                # Translate to English if needed
                english_question = await translate_to_english(user_input, st.session_state.selected_language)
                
                # Get response
                english_response = conversation.predict(human_input=english_question)
                
                # Translate back to selected language
                final_response = await translate_from_english(english_response, st.session_state.selected_language)
                return final_response
            
            # Run the async function
            final_response = asyncio.run(process_message())
            
            # Update message
            message_placeholder.markdown(final_response)
        
        # Add assistant message to chat history
        st.session_state.conversation_history.append({"role": "assistant", "content": final_response})
