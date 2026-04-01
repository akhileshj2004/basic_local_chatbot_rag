import ollama
import sys

def chat_engine():
    model_name = "llama3.2:1b"
    conversation_history = []

    print(f"System: Loaded {model_name}. Ctrl+C to exit.")

    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                sys.exit()

            conversation_history.append({'role': 'user', 'content': user_input})

            stream = ollama.chat(
                model=model_name,
                messages=conversation_history,
                stream=True
            )

            print("Bot: ", end="", flush=True)
            
            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end="", flush=True)
                full_response += content
            
            conversation_history.append({'role': 'assistant', 'content': full_response})

        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_engine()