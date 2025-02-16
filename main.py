import os
import re
import sys
import time
import streamlit as st
import google.generativeai as genai
import cloudscraper
from bs4 import BeautifulSoup
from googlesearch import search

########################################
#  HARDCODED CONFIG & INITIAL SETUP    #
########################################

# Hardcoded Google API Key for Gemini (replace if needed).
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

# We create a cloudscraper session at the top level so we don’t recreate it each time.
scraper = cloudscraper.create_scraper()

# Initialise session state for logs and final markdown output.
if "log_messages" not in st.session_state:
    st.session_state["log_messages"] = []
if "organized_text" not in st.session_state:
    st.session_state["organized_text"] = ""

########################################
#          LOGGING FUNCTION            #
########################################

def log(message: str) -> None:
    """
    Append a message to st.session_state logs,
    so they persist across script reruns.
    """
    st.session_state["log_messages"].append(message)

def log_startup_details():
    """
    Logs environment details for extra diagnostics.
    """
    log("[Init] Starting new run.")
    log(f"[Init] Python version: {sys.version}")
    log(f"[Init] Working directory: {os.getcwd()}")
    # If you need environment variables:
    # for k, v in os.environ.items():
    #     log(f"[Env] {k}={v}")

########################################
#      CLOUDSCRAPER FETCH FUNCTION     #
########################################

def fetch_url(url: str, timeout: int = 10, retries: int = 3) -> str:
    """
    Uses cloudscraper to retrieve HTML from a URL, with a few retries.
    Logs any errors or success messages.
    """
    content = ""
    for attempt in range(retries):
        log(f"[Fetch] Attempt {attempt+1}: Fetching {url}")
        try:
            response = scraper.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.text
            log(f"[Fetch] Success: {url} returned HTTP {response.status_code}")
            return content
        except Exception as e:
            log(f"[Fetch] Error on attempt {attempt+1} for {url}: {e}")
            time.sleep(2)
    log(f"[Fetch] Failed to fetch {url} after {retries} attempts.")
    return content

########################################
#      EXTRACT QUESTIONS (HTML)        #
########################################

def extract_questions(html: str) -> list[str]:
    """
    Splits HTML text into sentences and picks out those containing a question mark,
    ignoring very short or spammy lines.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            if any(
                keyword in sentence.lower()
                for keyword in ["add to the discussion", "vote", "comment", "submit", "loading", "http"]
            ):
                continue
            if sentence not in questions:
                questions.append(sentence)
    log(f"[Extract] Found {len(questions)} potential questions in HTML.")
    return questions

########################################
#        GEMINI CALLS (SYNC)           #
########################################

def call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """
    Calls Google's Gemini model in a blocking manner,
    logs the prompt length and any errors or successes.
    """
    log(f"[Gemini] Sending request (prompt length: {len(prompt)} chars)...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": max_tokens}
        )
        if response and response.text:
            log(f"[Gemini] Received response (length: {len(response.text)} chars).")
            return response.text
        else:
            log("[Gemini] No valid response from Gemini.")
    except Exception as e:
        log(f"[Gemini] Error: {e}")
    return ""

def refine_extracted_questions(questions: list[str], url: str) -> list[str]:
    """
    Uses Gemini to rewrite extracted questions into natural search queries.
    """
    if not questions:
        log(f"[Refine] No questions to refine for {url}.")
        return []
    prompt = (
        f"Below are candidate questions extracted from a Reddit thread (source: {url}):\n"
        + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        + "\n\nRewrite these as professional, natural search queries in plain language that someone would type into google, "
          "avoiding clickbait. Output them as a numbered list."
    )
    log(f"[Refine] Refining {len(questions)} extracted questions from {url}...")
    raw_text = call_gemini(prompt, max_tokens=300)
    refined = []
    for line in raw_text.splitlines():
        line = line.strip()
        if re.match(r'^\d+\.', line):
            parts = line.split(". ", 1)
            refined.append(parts[1].strip() if len(parts) > 1 else line)
    log(f"[Refine] Final refined question count: {len(refined)}.")
    return refined

def infer_extra_questions(html: str, url: str, desired_count: int, truncate_len: int) -> list[str]:
    """
    Asks Gemini to infer additional questions from the given HTML content.
    """
    if desired_count < 1:
        return []
    truncated_text = html[:truncate_len]
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {desired_count} additional relevant questions "
        "rephrased as natural search queries in plain, professional Australian English. Output them as a numbered list. "
        "If you cannot generate exactly the requested number, list as many as possible.\n\n"
        f"Thread content (truncated to {truncate_len} chars):\n{truncated_text}"
    )
    log(f"[Infer] Inferring {desired_count} extra questions for {url}...")
    raw_text = call_gemini(prompt, max_tokens=300)
    inferred = []
    for line in raw_text.splitlines():
        line = line.strip()
        if re.match(r'^\d+\.', line):
            parts = line.split(". ", 1)
            inferred.append(parts[1].strip() if len(parts) > 1 else line)
    log(f"[Infer] Inferred {len(inferred)} extra questions.")
    return inferred

########################################
#  ORGANISE QUESTIONS INTO MARKDOWN    #
########################################

def organise_questions(candidates: list[dict], batch_size: int = 50) -> str:
    """
    Groups candidate questions into categories using Gemini,
    outputting them in Markdown.
    """
    if not candidates:
        log("[Organise] No candidate questions available.")
        return "No candidate questions available."

    log(f"[Organise] Organising {len(candidates)} total questions in batches of {batch_size}.")
    final_md = ""
    batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    for i, batch in enumerate(batches):
        batch_text = "\n".join(
            f"{item['question']} [{item['type'].capitalize()}] ({item['url']})"
            for item in batch
        )
        if i == 0:
            prompt = (
                "Rewrite and group the following candidate questions into multiple, specific topical categories for blog research. "
                "Each question must be a natural search query in plain language exactly as someone would type into google, avoiding clickbait. "
                "Output the final result in the exact Markdown format below. Only questions.\n\n"
                "### [Category Title]\n"
                "- [Rewritten Question] ([reddit thread URL])\n\n"
                f"Candidate Questions:\n{batch_text}"
            )
        else:
            prompt = (
                f"Previously organised output:\n{final_md}\n\n"
                f"Now add these new questions:\n{batch_text}\n\n"
                "Rewrite them as natural search queries in plain language, grouping them into relevant categories. "
                "Output everything in the same Markdown format. Only questions."
            )
        log(f"[Organise] Processing batch {i+1}/{len(batches)}, size={len(batch)}.")
        raw_md = call_gemini(prompt, max_tokens=800)
        if raw_md:
            final_md = raw_md
    return final_md

########################################
#   MAIN STREAMLIT UI & LOGIC          #
########################################

def main():
    log_startup_details()
    st.title("Reddit Research with Gemini (cloudscraper Edition)")
    st.write(
        "Enter a search topic and parameters. We'll search Reddit using cloudscraper, "
        "refine/infer questions via Gemini, and organise them. Logs below help diagnose any issues."
    )

    query = st.text_input("Enter a search topic", "retirement")
    threads_count = st.number_input("Number of Reddit threads to check", min_value=1, value=2)
    questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=2)
    truncate_len = st.number_input("Truncation length for Gemini", min_value=100, value=10000)

    if st.button("Search"):
        # Clear old logs & output
        st.session_state["log_messages"] = []
        st.session_state["organized_text"] = ""

        log_startup_details()
        log(f"[UserInput] query='{query}', threads_count={threads_count}, questions_per_thread={questions_per_thread}, truncate_len={truncate_len}")
        
        with st.spinner("Scraping Reddit and calling Gemini..."):
            run_search(query, threads_count, questions_per_thread, truncate_len)

        st.subheader("Process Log")
        st.text("\n".join(st.session_state["log_messages"]))

        st.subheader("Final Organised Output")
        st.markdown(st.session_state["organized_text"])

def run_search(query: str, threads_count: int, questions_count: int, truncate_len: int) -> None:
    """Synchronous flow: search Reddit, fetch HTML with cloudscraper, refine/infer, organise questions."""
    if not query.strip():
        log("[Error] Please enter a valid search topic.")
        return

    extracted_candidates = []
    inferred_candidates = []

    log(f"[Search] Searching for: '{query} site:reddit.com'")
    try:
        results = list(search(f"{query} site:reddit.com", num_results=threads_count))
        log(f"[Search] Found {len(results)} URLs for query '{query}'.")
    except Exception as e:
        log(f"[Search] Error during google search: {e}")
        return

    for url in results:
        if "reddit.com" not in url:
            log(f"[Search] Skipping non-Reddit URL: {url}")
            continue

        # Optionally convert "www.reddit.com" to "old.reddit.com"
        if "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")

        log(f"[Search] Now fetching HTML from: {url}")
        html = fetch_url(url, timeout=10, retries=3)
        if not html:
            log(f"[Search] Skipping {url} due to fetch failure.")
            continue
        log(f"[Search] Fetched HTML from {url} (length: {len(html)}).")

        # Extract & refine
        raw_extracted = extract_questions(html)[:questions_count]
        refined = refine_extracted_questions(raw_extracted, url)

        # Infer extra
        extra = infer_extra_questions(html, url, questions_count, truncate_len)

        for q in refined:
            extracted_candidates.append({"url": url, "question": q, "type": "extracted"})
        for q in extra:
            inferred_candidates.append({"url": url, "question": q, "type": "inferred"})

    all_candidates = extracted_candidates + inferred_candidates
    log(f"[Search] Combined total of {len(all_candidates)} candidate questions. Organising them now...")
    final_markdown = organise_questions(all_candidates, batch_size=50)
    st.session_state["organized_text"] = final_markdown
    log("[Search] Finished. Check final output above.")

########################################
#             ENTRY POINT              #
########################################

if __name__ == "__main__":
    main()
