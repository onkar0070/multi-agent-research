import streamlit as st
from pipeline import run_research_pipeline

# Page config
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Multi-Agent Research Assistant")
st.markdown(
    """
    This UI runs your complete multi-agent research pipeline:
    
    - 🔍 Search Agent
    - 📖 Reader Agent
    - ✍️ Writer Agent
    - 🧠 Critic Agent
    """
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Built using Streamlit + Multi-Agent Pipeline")

# Input
topic = st.text_input(
    "Enter a research topic",
    placeholder="Example: Future of AI in Healthcare"
)

# Button
if st.button("Run Research Pipeline", type="primary"):

    if not topic.strip():
        st.warning("Please enter a topic.")
    else:

        # Progress section
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:

            # Run pipeline
            status_text.info("🔍 Running Search Agent...")
            progress_bar.progress(20)

            result = run_research_pipeline(topic)

            progress_bar.progress(100)
            status_text.success("✅ Pipeline Completed Successfully!")

            # Tabs for results
            tab1, tab2, tab3, tab4 = st.tabs([
                "Search Results",
                "Scraped Content",
                "Final Report",
                "Critic Feedback"
            ])

            # Search results
            with tab1:
                st.subheader("🔍 Search Results")
                st.write(result.get("search_results", "No search results"))

            # Scraped content
            with tab2:
                st.subheader("📖 Scraped Content")
                st.write(result.get("scraped_content", "No scraped content"))

            # Final report
            with tab3:
                st.subheader("✍️ Final Report")
                st.write(result.get("report", "No report generated"))

            # Critic feedback
            with tab4:
                st.subheader("🧠 Critic Feedback")
                st.write(result.get("feedback", "No feedback generated"))

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")