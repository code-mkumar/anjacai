import streamlit as st
import random

# Page configuration
st.set_page_config(page_title="Fun Games", layout="centered")

# Sidebar menu for game selection
st.sidebar.title("Select a Game")
game_choice = st.sidebar.radio("Choose a game:", ["Rock-Paper-Scissors", "Word Scramble"])

# Rock-Paper-Scissors Game
if game_choice == "Rock-Paper-Scissors":
    st.title("✊ Rock-Paper-Scissors")
    st.write("Play against the computer!")

    choices = ["Rock", "Paper", "Scissors"]
    user_choice = st.radio("Make your choice:", choices)
    if st.button("Play"):
        computer_choice = random.choice(choices)
        st.write(f"🤖 Computer chose: {computer_choice}")

        if user_choice == computer_choice:
            st.info("It's a tie! 🤝")
        elif (
            (user_choice == "Rock" and computer_choice == "Scissors") or
            (user_choice == "Paper" and computer_choice == "Rock") or
            (user_choice == "Scissors" and computer_choice == "Paper")
        ):
            st.success("You win! 🎉")
        else:
            st.error("You lose! 😢")

# Word Scramble Game
elif game_choice == "Word Scramble":
    st.title("🔀 Word Scramble Game")
    st.write("Unscramble the word and type your answer!")

    # List of words
    words = ["streamlit", "python", "programming", "developer", "challenge"]
    
    # Initialize session state for the word
    if "scrambled_word" not in st.session_state:
        original_word = random.choice(words)
        scrambled_word = "".join(random.sample(original_word, len(original_word)))
        st.session_state.scrambled_word = scrambled_word
        st.session_state.original_word = original_word

    # Display scrambled word
    st.write(f"Scrambled word: **{st.session_state.scrambled_word}**")
    
    # Input for the user's guess
    user_guess = st.text_input("Your guess:")
    if st.button("Submit"):
        if user_guess.lower() == st.session_state.original_word:
            st.success("🎉 Correct! You unscrambled the word.")
            # Reset the game
            original_word = random.choice(words)
            scrambled_word = "".join(random.sample(original_word, len(original_word)))
            st.session_state.scrambled_word = scrambled_word
            st.session_state.original_word = original_word
        else:
            st.error("❌ Incorrect! Try again.")
