import os
import re
import sys
import time
import requests
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search

########################################
#    HARDCODED CONFIG & INITIAL SETUP  #
########################################

# Hardcoded Google API Key for Gemini (Replace if needed)
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

# A more realistic user agent to reduce 403 blocks from Reddit
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

# Initialise session state variables so we don’t get KeyError
if "log_messages" not in st.session_state:
    st.session_state["log_messages"] = []
if "organized_text" not in st.session_state:
    st.session_state["organized_text"] = ""

def log(message: str) -> None:
    """Append a message to session_state and show it in final logs."""
    st.session_state["log_messages"].append(message)

########################################
#   ADDITIONAL DEBUG/DIAGNOSTIC LOGGING
########################################

def log_startup_details():
    """
    Logs environment details at the start
    to help diagnose issues like 403 blocks.
    """
    # Basic environment info
    log("[Init] Starting new run.")
    log(f"[Init] Python version: {sys.version}")
    log(f"[Init] Working directory: {os.getcwd()}")
    log(f"[Init] User agent set to: {HEADERS['User-Agent']}")
    # If you want to see certain environment variables, uncomment below:
    # for k, v in os.environ.items():
    #     if "KEY" in k or "SECRET" in k:
    #         continue  # hide sensitive info
    #     log(f"[Env] {k}={v}")

########################################
#           SCRAPING HELPER            #
########################################

def fetch_url(
    url: str,
    headers: dict,
    timeout: int = 10,
    retries: int = 3
) -> str:
    """
    Synchronous function to fetch a URL with retries, logging progress
    in st.session_state. Useful for diagnosing 403 or other errors.
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
    Extract lines that look like questions.
    We log how many we found to see if we actually retrieved meaningful HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            # skip spammy/unhelpful lines
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
#         GEMINI (SYNCHRONOUS)         #
########################################

def call_gemini(prompt: str, max_tokens: int = 300) -> str:
    """
    Logs prompt length and result length to diagnose issues with Gemini calls.
    """
    log(f"[Gemini] Sending request with prompt length {len(prompt)} chars.")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": max_tokens}
        )
        if response and response.text:
            log(f"[Gemini] Received response of length {len(response.text)} chars.")
            return response.text
        else:
            log("[Gemini] No valid text response from Gemini.")
    except Exception as e:
        log(f"[Gemini] Error: {e}")
    return ""

########################################
#  EXTRACT & INFER QUESTIONS VIA GEMINI
########################################

def refine_extracted_questions(questions: list[str], url: str) -> list[str]:
    """Rewrite extracted questions into more natural queries via Gemini."""
    if not questions:
        log(f"[Gemini] No extracted questions to refine for {url}.")
        return []
    prompt = (
        f"Below are candidate questions extracted from a Reddit thread (source: {url}):\n"
        + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        + "\n\nRewrite these as professional, natural search queries in plain language that someone would type into google, "
          "avoiding clickbait or headline style. Output them as a numbered list."
    )
    log(f"[Refine] Attempting to refine {len(questions)} extracted questions.")
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
    """Asks Gemini to infer more questions from the thread content."""
    if desired_count < 1:
        return []
    truncated_text = html[:truncate_len]
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {desired_count} additional relevant questions "
        "rephrased as natural search queries in plain, professional Australian English. The questions should be typed exactly as a user would "
        "in google, avoiding clickbait or headline style. Output them as a numbered list. If you can’t generate that many, list as many as possible.\n\n"
        f"Thread content (truncated to {truncate_len} chars):\n{truncated_text}"
    )
    log(f"[Infer] Asking for {desired_count} extra questions, truncated to {truncate_len} chars.")
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
#  FINAL ORG OF QUESTIONS INTO MD      #
########################################

def organise_questions(candidates: list[dict], batch_size: int = 50) -> str:
    """
    Group candidate questions with Gemini in Markdown format.
    Log batch size, etc.
    """
    if not candidates:
        log("[Organise] No candidate questions available.")
        return "No candidate questions available."
    log(f"[Organise] Organising total {len(candidates)} questions in batches of {batch_size}.")

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
                "Each question is a natural search query in plain language exactly as someone would type into google, avoiding clickbait. "
                "Output only in Markdown with the format:\n\n"
                "### [Category Title]\n"
                "- [Rewritten Question] ([reddit thread URL])\n\n"
                f"Candidate Questions:\n{batch_text}"
            )
        else:
            prompt = (
                f"Previously organised output:\n{final_md}\n\n"
                f"New candidate questions:\n{batch_text}\n\n"
                "Add these questions to the output, grouped into relevant categories, in the same Markdown format. Only questions."
            )
        log(f"[Organise] Processing batch {i+1}/{len(batches)} of size {len(batch)}.")
        raw_md = call_gemini(prompt, max_tokens=800)
        if raw_md:
            final_md = raw_md
    return final_md

########################################
#  STREAMLIT UI & MAIN SCRIPT          #
########################################

def main():
    # Always log startup details at the beginning of each run
    log_startup_details()

    st.title("Reddit Research with Gemini (Detailed Logging)")
    st.write(
        "This script tries to scrape Reddit for a given search topic, refine/infer questions via Gemini, "
        "and organise them in Markdown. The logs below should help diagnose 403 or other errors."
    )

    # Input fields
    query = st.text_input("Enter a search topic", "retirement")
    threads_count = st.number_input("Number of Reddit threads to check", min_value=1, value=2)
    questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=2)
    truncate_len = st.number_input("Truncation length for Gemini", min_value=100, value=10000)

    if st.button("Search"):
        # Clear old logs & final output
        st.session_state["log_messages"] = []
        st.session_state["organized_text"] = ""

        # Re-log the startup details so we see them in the final log
        log_startup_details()
        log(f"[UserInput] query='{query}', threads_count={threads_count}, questions_per_thread={questions_per_thread}, truncate_len={truncate_len}")

        with st.spinner("Scraping Reddit and calling Gemini..."):
            run_search(query, threads_count, questions_per_thread, truncate_len)

        st.subheader("Process Log")
        st.text("\n".join(st.session_state["log_messages"]))

        st.subheader("Final Organised Output")
        st.markdown(st.session_state["organized_text"])

def run_search(query: str, threads_count: int, questions_count: int, truncate_len: int):
    """Synchronously scrape, refine, infer, and organise. Logs all steps in detail."""
    if not query.strip():
        log("[Error] Query is empty. Nothing to do.")
        return

    extracted_candidates = []
    inferred_candidates = []

    # Searching
    log(f"[Search] Searching for: '{query} site:reddit.com'")
    try:
        results = list(search(f"{query} site:reddit.com", num_results=threads_count))
        log(f"[Search] Found {len(results)} URLs for query '{query}'.")
    except Exception as e:
        log(f"[Search] Google search error: {e}")
        return

    # For each URL, fetch HTML & process
    for url in results:
        if "reddit.com" not in url:
            log(f"[Search] Skipping non-Reddit URL: {url}")
            continue
        # Convert to old.reddit format
        if "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")

        log(f"[Search] Now fetching HTML from: {url}")
        html = fetch_url(url, HEADERS)
        if not html:
            log(f"[Search] Skipping {url} due to fetch failure.")
            continue

        # Extract & refine
        raw_extracted = extract_questions(html)[:questions_count]
        refined = refine_extracted_questions(raw_extracted, url)

        # Infer extra
        inferred = infer_extra_questions(html, url, questions_count, truncate_len)

        # Collect final
        for q in refined:
            extracted_candidates.append({"url": url, "question": q, "type": "extracted"})
        for q in inferred:
            inferred_candidates.append({"url": url, "question": q, "type": "inferred"})

    all_candidates = extracted_candidates + inferred_candidates
    log(f"[Search] Combined total of {len(all_candidates)} questions. Now organising via Gemini.")
    final_markdown = organise_questions(all_candidates, batch_size=50)
    st.session_state["organized_text"] = final_markdown
    log("[Search] Finished. Check the final organised output above.")

if __name__ == "__main__":
    main()