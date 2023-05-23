
import streamlit as st

# Define your Streamlit app
def main():
    st.title('Text Summarization Training')

    # Add Streamlit UI elements for user input if necessary
    document = st.text_area('Enter the document:', height=200)
    max_summary_length = st.slider('Max Summary Length', min_value=10, max_value=100, value=50)

    # Run the summarization process on button click
    if st.button('Summarize'):
        with st.spinner('Generating Summary...'):
            # Run your summarization code here
            summary = generate_summary(document, max_summary_length)

        # Display the generated summary
        st.subheader('Generated Summary:')
        st.text(summary)

    # Add additional UI elements as needed

# Run the Streamlit app
if __name__ == '__main__':
    main()
