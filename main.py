import os
import re
import time
import requests
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search

########################################
#  HARDCODED CONFIG & INITIAL SETUP    #
########################################

# Hardcoded Google API Key for Gemini. Replace if needed.
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

# Initialise session state variables to avoid KeyErrors.
# These will persist across script reruns in a single user session.
if "log_messages" not in st.session_state:
    st.session_state["log_messages"] = []
if "organized_text" not in st.session_state:
    st.session_state["organized_text"] = ""

########################################
#             LOGGING                  #
########################################

def log(message: str) -> None:
    """
    Appends a log message to session_state and
    ensures it's displayed in the final logs.
    """
    st.session_state["log_messages"].append(message)

########################################
#        SCRAPING HELPER               #
########################################

def fetch_url(url: str, headers: dict, timeout: int = 10, retries: int = 3) -> str:
    """
    Synchronous function to fetch a URL with retries, logging progress to st.session_state.
    """
    for attempt in range(retries):
        log(f"[Fetch] Attempt {attempt + 1}: Fetching {url}")
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            log(f"[Fetch] Success: {url} returned HTTP {response.status_code}")
            return response.text
        except requests.exceptions.RequestException as e:
            log(f"[Fetch] Error on attempt {attempt + 1} for {url}: {e}")
            time.sleep(1)
    log(f"[Fetch] Failed to fetch {url} after {retries} attempts.")
    return ""

def extract_questions(html: str) -> list[str]:
    """
    Simple extractor to find sentences with a question mark.
    Filters out short or unhelpful lines.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            # Filter out lines with common spam
            if any(
                keyword in sentence.lower()
                for keyword in ["add to the discussion", "vote", "comment", "submit", "loading", "http"]
            ):
                continue
            # Avoid duplicates
            if sentence not in questions:
                questions.append(sentence)
    return questions

########################################
#         GEMINI (SYNCHRONOUS)         #
########################################

def call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """
    Synchronous call to Google's Gemini model. 
    Logs progress before and after the call.
    """
    try:
        log("[Gemini] Sending request...")
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": max_tokens}
        )
        if response and response.text:
            log("[Gemini] Received response.")
            return response.text
        else:
            log("[Gemini] No valid text response from Gemini.")
    except Exception as e:
        log(f"[Gemini] Error: {e}")
    return ""

def refine_extracted_questions(questions: list[str], url: str) -> list[str]:
    """
    Takes a list of extracted questions and refines them via Gemini to more natural search queries.
    """
    if not questions:
        return []
    prompt = (
        f"Below are candidate questions extracted from a Reddit thread (source: {url}):\n"
        + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        + "\n\nRewrite these as professional, natural search queries in plain language "
          "that someone would type into google, avoiding clickbait or headline style. "
          "Output them as a numbered list."
    )
    log(f"[Gemini] Refining {len(questions)} extracted questions from {url}...")
    raw_text = call_gemini(prompt, max_tokens=300)
    refined = []
    for line in raw_text.splitlines():
        line = line.strip()
        if re.match(r'^\d+\.', line):
            parts = line.split(". ", 1)
            refined.append(parts[1].strip() if len(parts) > 1 else line)
    log(f"[Gemini] Refined output length: {len(refined)}.")
    return refined

def infer_extra_questions(html: str, url: str, desired_count: int, truncate_len: int) -> list[str]:
    """
    Asks Gemini to infer additional questions from the Reddit thread content.
    """
    if desired_count < 1:
        return []
    truncated_text = html[:truncate_len]
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {desired_count} additional relevant questions "
        "rephrased as natural search queries in plain, professional Australian English. "
        "These should be typed exactly as a user would into google, avoiding clickbait or headline styles. "
        "Output them as a numbered list. If you cannot generate exactly that many, list as many as possible.\n\n"
        f"Thread content (truncated to {truncate_len} chars):\n{truncated_text}"
    )
    log(f"[Gemini] Inferring {desired_count} additional questions for {url}...")
    raw_text = call_gemini(prompt, max_tokens=300)
    inferred = []
    for line in raw_text.splitlines():
        line = line.strip()
        if re.match(r'^\d+\.', line):
            parts = line.split(". ", 1)
            inferred.append(parts[1].strip() if len(parts) > 1 else line)
    log(f"[Gemini] Inferred {len(inferred)} extra questions.")
    return inferred

########################################
#    ORGANISE QUESTIONS IN BATCHES     #
########################################

def organise_questions(candidates: list[dict], batch_size: int = 50) -> str:
    """
    Groups candidate questions into categories via Gemini, rewriting them in a Markdown format.
    """
    if not candidates:
        return "No candidate questions available."
    log(f"[Organise] Preparing to batch {len(candidates)} questions in chunks of {batch_size}.")
    final_markdown = ""
    # Split into chunks
    batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    for i, batch in enumerate(batches):
        # Build a string with question + type + url
        batch_text = "\n".join(
            f"{item['question']} [{item['type'].capitalize()}] ({item['url']})"
            for item in batch
        )
        # The prompt changes slightly depending on if it's the first batch or subsequent
        if i == 0:
            prompt = (
                "Rewrite and group the following candidate questions into multiple, specific topical categories for blog research. "
                "Each candidate question must be a natural search query in plain language exactly as someone would type into google, "
                "avoiding any clickbait or headline style. Output the final result in the exact Markdown format below. "
                "Only contain questions.\n\n"
                "### [Category Title]\n"
                "- [Rewritten Question] ([reddit thread URL])\n\n"
                f"Candidate Questions:\n{batch_text}"
            )
        else:
            prompt = (
                f"Below is the previously organised output in Markdown:\n{final_markdown}\n\n"
                "Now add these new questions:\n"
                f"{batch_text}\n\n"
                "Rewrite them as natural search queries in plain language, grouping them into relevant categories. "
                "Output everything in the same Markdown format as before, containing only questions."
            )
        log(f"[Organise] Processing batch {i+1}/{len(batches)} with {len(batch)} questions.")
        raw_md = call_gemini(prompt, max_tokens=800)
        if raw_md:
            final_markdown = raw_md
    return final_markdown

########################################
#           MAIN SCRIPT (UI)           #
########################################

def main():
    st.title("Reddit Research with Gemini")
    st.write(
        "Enter a search topic and how many threads/questions you want to process. "
        "This script will scrape Reddit, refine questions with Gemini, and finally organise them in Markdown."
    )

    # Input fields
    query = st.text_input("Enter a search topic", "retirement")
    threads_count = st.number_input("Number of Reddit threads to check", min_value=1, value=2)
    questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=2)
    truncate_len = st.number_input("Truncation length for Gemini", min_value=100, value=10000)

    # Click the button to run the scraping & gemini logic synchronously
    if st.button("Search"):
        # Clear old logs and output
        st.session_state["log_messages"] = []
        st.session_state["organized_text"] = ""

        with st.spinner("Running search & Gemini calls..."):
            run_search(query, threads_count, questions_per_thread, truncate_len)

        # Show logs
        st.subheader("Process Log")
        st.text("\n".join(st.session_state["log_messages"]))

        # Show final Markdown
        st.subheader("Final Organised Output")
        st.markdown(st.session_state["organized_text"])

def run_search(query: str, threads_count: int, questions_count: int, truncate_len: int) -> None:
    """
    Synchronously scrapes Reddit, refines & infers questions with Gemini,
    then organises the final list of questions into a Markdown structure.
    Stores logs and output in st.session_state.
    """
    if not query.strip():
        log("[Error] Please enter a non-empty topic.")
        return

    candidate_extracted = []
    candidate_inferred = []

    # Search
    log(f"[Search] Searching for: '{query} site:reddit.com'")
    try:
        results = list(search(f"{query} site:reddit.com", num_results=threads_count))
        log(f"[Search] Found {len(results)} URLs.")
    except Exception as e:
        log(f"[Search] Error during Google search: {e}")
        return

    # Scrape & process each URL
    for url in results:
        # Check it's a valid Reddit URL
        if "reddit.com" not in url:
            log(f"[Search] Skipping non-Reddit URL: {url}")
            continue
        # Convert to old.reddit
        if "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")

        log(f"[Search] Fetching HTML from {url}...")
        html = fetch_url(url, {"User-Agent": "Mozilla/5.0"})
        if not html:
            log(f"[Search] Skipping {url} because fetch failed.")
            continue

        # Extract & refine
        raw_questions = extract_questions(html)[:questions_count]
        refined = refine_extracted_questions(raw_questions, url)

        # Infer additional
        inferred = infer_extra_questions(html, url, questions_count, truncate_len)

        # Accumulate final candidate questions
        for q in refined:
            candidate_extracted.append({"url": url, "question": q, "type": "extracted"})
        for q in inferred:
            candidate_inferred.append({"url": url, "question": q, "type": "inferred"})

    # Combine & organise
    all_candidates = candidate_extracted + candidate_inferred
    log(f"[Search] Total combined questions: {len(all_candidates)}. Organising them now...")
    final_markdown = organise_questions(all_candidates, batch_size=50)
    st.session_state["organized_text"] = final_markdown
    log("[Search] Finished organising questions with Gemini!")

########################################
#       STREAMLIT ENTRY POINT          #
########################################

if __name__ == "__main__":
    main()