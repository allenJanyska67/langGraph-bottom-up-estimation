import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from estimation_agent import graph, State

load_dotenv()
config = {"configurable": {"thread_id": "thread"}}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "canvas" not in st.session_state:
    st.session_state.canvas = ""

st.set_page_config(layout = "wide")

st.title("Chatbot")

st.html(
    """
    <style>
    .stColumn {
        height: 70vh;
        overflow-y: auto;
    }
    </style>
    """
)
left_column, right_column = st.columns([1,1], border=True)

# Display the canvas state.
# Handle user input.
with right_column:
    messages_display = st.container()
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with messages_display.chat_message("user"):
                messages_display.write(message.content)
        elif isinstance(message, AIMessage):
            with messages_display.chat_message("assistant"):
                messages_display.write(message.content)

    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with messages_display.chat_message("user"):
            messages_display.write(prompt)

        # Create a placeholder for the streaming response
        with messages_display.chat_message("assistant"):
            message_placeholder = messages_display.empty()
            full_response = ""
            
            try:
                state = State(messages=[user_message])
              

                # invoke the agent, streaming tokens from any llm calls directly
                for chunk, metadata in graph.stream(state, config=config, stream_mode="messages"):
                    if isinstance(chunk, AIMessage):
                        full_response = full_response + chunk.content
                        message_placeholder.markdown(full_response + "▌")

                    elif isinstance(chunk, ToolMessage):
                        full_response = full_response + f"🛠️ Used tool to get: {chunk.content}\n\n"
                        message_placeholder.markdown(full_response + "▌")

                # Once streaming is complete, display the final message without the cursor
                message_placeholder.markdown(full_response)

                # Add the complete message to session state
                st.session_state.messages.append(AIMessage(content=full_response))
                print(graph.get_state(config).values)
                with left_column:
                    st.write("Canvas")
                    st.session_state.canvas = graph.get_state(config).values["canvas"]
                    st.write(st.session_state.canvas)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}") 