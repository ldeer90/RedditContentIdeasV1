import os
import re
import time
import asyncio
import requests
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search

###################
#  CONFIGURATION  #
###################

# Hardcoded API key for Google Generative AI
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

##################
# SESSION STATE  #
##################

# Make sure session state keys exist.
if "log_messages" not in st.session_state:
    st.session_state["log_messages"] = []
if "organized_text" not in st.session_state:
    st.session_state["organized_text"] = ""

###################
#    PLACEHOLDERS #
###################

# We’ll update these after the run completes.
log_placeholder = st.empty()
output_placeholder = st.empty()

###################
#   LOG FUNCTION  #
###################

def log(message: str) -> None:
    """Append a log message to the session state and update placeholder."""
    st.session_state["log_messages"].append(message)
    log_placeholder.text("\n".join(st.session_state["log_messages"]))

###################
#  SCRAPING LOGIC #
###################

def fetch_url(url: str, headers: dict, timeout: int = 10, retries: int = 3) -> str:
    """Fetch a URL with retries, logging each attempt."""
    for attempt in range(retries):
        log(f"[Fetch] Attempt {attempt+1}: Fetching {url}")
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            log(f"[Fetch] {url} returned HTTP status {response.status_code}")
            return response.text
        except requests.exceptions.RequestException as e:
            log(f"[Fetch] Error on attempt {attempt+1} for {url}: {e}")
            time.sleep(2)
    log(f"[Fetch] Failed to fetch {url} after {retries} attempts.")
    return ""

def extract_questions(html: str) -> list[str]:
    """Extract sentences that appear to be questions from HTML text."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            # Filter out unhelpful or repeated lines
            if any(keyword in sentence.lower() for keyword in [
                "add to the discussion", "vote", "comment", "submit", "read more", "loading", "http"
            ]):
                continue
            if sentence not in questions:
                questions.append(sentence)
    return questions

###################
# GEMINI CALLS    #
###################

async def assess_questions_relevance(questions: list[str], url: str) -> list[str]:
    """Rewrite extracted questions into more natural queries via Gemini."""
    if not questions:
        return []
    prompt = (
        f"Below are candidate questions extracted from a Reddit thread (source: {url}):\n"
        + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        + "\n\nRewrite these as professional, natural search queries in plain language that someone would actually type into google. "
          "Avoid clickbait or headline-like language. Only include the questions that are highly relevant and actionable. "
          "Output your answer as a numbered list."
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 300}
        )
        rewritten = []
        for line in response.text.splitlines():
            line = line.strip()
            if line.lower() == "none":
                return []
            if re.match(r'^\d+\.', line):
                parts = line.split(". ", 1)
                rewritten.append(parts[1].strip() if len(parts) > 1 else line)
        log(f"[Relevance] Rewritten {len(rewritten)} questions from {url}.")
        return rewritten
    except Exception as e:
        log(f"[Relevance] Error rewriting questions for {url}: {e}")
        return questions

async def get_thread_questions(html: str, url: str, desired_count: int, trunc_length: int) -> tuple[list[str], list[str]]:
    """Extract questions from HTML, refine them, and infer additional questions."""
    log(f"[Extract] Extracting candidate questions from {url}")
    extracted = extract_questions(html)[:desired_count]
    rewritten_extracted = await assess_questions_relevance(extracted, url)
    
    inferred = []
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {desired_count} additional relevant questions "
        "rephrased as natural search queries in plain, professional Australian English. The questions should be written exactly as someone "
        f"would type them into google, avoiding clickbait. Output your response as a numbered list. If you cannot generate exactly "
        f"the requested number, list as many as possible.\n\n"
        f"Thread content (truncated to {trunc_length} characters):\n{html[:trunc_length]}"
    )
    log(f"[Infer] Prompting Gemini to infer {desired_count} additional questions for {url}.")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 300}
        )
        additional_text = response.text
        log(f"[Infer] Raw inferred text: {additional_text}")
        for line in additional_text.splitlines():
            line = line.strip()
            if re.match(r'^\d+\.', line):
                parts = line.split(". ", 1)
                inferred.append(parts[1].strip() if len(parts) > 1 else line)
        log(f"[Infer] Inferred {len(inferred)} additional questions from {url}.")
    except Exception as e:
        log(f"[Infer] Error inferring questions for {url}: {e}")
    return rewritten_extracted, inferred

async def call_gemini(prompt: str, retries: int = 10, max_tokens: int = 2000) -> str:
    """General Gemini call with a backoff mechanism."""
    model = genai.GenerativeModel(MODEL_NAME)
    backoff_seconds = 2
    for attempt in range(retries):
        try:
            log(f"[Gemini] Attempt {attempt+1}: Calling Gemini...")
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.5, "max_output_tokens": max_tokens}
            )
            if response and response.text:
                log("[Gemini] Gemini call successful.")
                return response.text
            else:
                log("[Gemini] No valid result from Gemini.")
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                log(f"[Gemini] 429 error: attempt {attempt+1}, backing off.")
            else:
                log(f"[Gemini] Error: {e} (Attempt {attempt+1})")
        await asyncio.sleep(backoff_seconds)
        backoff_seconds *= 2
    raise RuntimeError(f"Gemini call failed after {retries} attempts.")

async def organise_batches_iteratively(candidates: list[dict], batch_size: int = 50) -> str:
    """Group and rewrite candidate questions in multiple topical categories."""
    if not candidates:
        return "No candidate questions available."
    batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    current_output = ""
    for i, batch in enumerate(batches):
        batch_text = "\n".join(
            f"{item['question']} [{item['type'].capitalize()}] ({item['url']})"
            for item in batch
        )
        if i == 0:
            prompt = (
                "Rewrite and group these candidate questions into multiple, specific topical categories for blog research. "
                "Each question should be rewritten as a natural search query in plain language exactly as someone would type into google, "
                "avoiding any clickbait or headline-like language. Group the questions into topics that make sense, and output the final "
                "result in the exact Markdown format shown below. The output should only contain questions.\n\n"
                "### [Category Title]\n"
                "- [Rewritten Question] ([reddit thread URL])\n\n"
                f"Candidate Questions:\n{batch_text}"
            )
        else:
            prompt = (
                f"Below is the previously organised output in Markdown:\n{current_output}\n\n"
                "Now here are additional candidate questions:\n"
                f"{batch_text}\n\n"
                "Update the organised output to incorporate these new questions. Rewrite them as natural search queries in plain language. "
                "Group them into appropriate topics and output the result in the exact Markdown format as before, containing only questions."
            )
        log(f"[Batch] Processing batch {i+1} of {len(batches)}")
        batch_output = await call_gemini(prompt, max_tokens=2000)
        current_output = batch_output
    return current_output

###################
#  MAIN APP FLOW  #
###################

def main():
    st.title("Reddit Research with Gemini")
    st.write(
        "Enter your search topic below. The app will search Reddit, extract questions, "
        "refine them with Gemini, and organise the final output in Markdown."
    )
    
    # Create input fields
    query = st.text_input("Enter topic", key="query")
    threads_count = st.number_input("Number of threads to check", min_value=1, value=10)
    questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=10)
    truncate_len = st.number_input("Truncation length for Gemini inference", min_value=100, value=10000)
    
    if st.button("Search"):
        st.session_state["log_messages"] = []
        st.session_state["organized_text"] = ""

        # Run everything synchronously with a spinner.
        with st.spinner("Scraping Reddit and calling Gemini..."):
            asyncio.run(run_search(query, threads_count, questions_per_thread, truncate_len))

        # After the run is complete, show final logs and output.
        st.subheader("Process Log")
        st.text("\n".join(st.session_state["log_messages"]))
        
        st.subheader("Final Organised Output")
        st.markdown(st.session_state["organized_text"])

async def run_search(query: str, threads_count: int, questions_count: int, truncate_len: int):
    """Synchronous-ish search + question extraction + Gemini calls, all in the main thread context."""
    global final_output
    if not query.strip():
        log("[Error] Please enter a valid topic.")
        return

    candidate_questions_extracted = []
    candidate_questions_inferred = []

    log(f"[Search] Starting search for: '{query} site:reddit.com'")
    try:
        results = list(search(f"{query} site:reddit.com", num_results=threads_count))
        log(f"[Search] Found {len(results)} URLs for query: {query}")
    except Exception as e:
        log(f"[Search] Error while searching: {e}")
        return

    for url in results:
        if "reddit.com" not in url:
            log(f"[Search] Skipping non-Reddit URL: {url}")
            continue
        log(f"[Search] Processing URL: {url}")

        # Convert "www.reddit.com" to "old.reddit.com" for more consistent scraping
        if "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")

        html = await asyncio.to_thread(fetch_url, url, {"User-Agent": "Mozilla/5.0"})
        if not html:
            log(f"[Search] Skipping {url} due to fetch failure.")
            continue

        log(f"[Search] Fetched HTML from {url} (length: {len(html)})")

        # Get extracted and inferred Qs
        extracted, inferred = await get_thread_questions(html, url, questions_count, truncate_len)
        for q in extracted:
            candidate_questions_extracted.append({"url": url, "question": q, "type": "extracted"})
        for q in inferred:
            candidate_questions_inferred.append({"url": url, "question": q, "type": "inferred"})

    all_candidates = candidate_questions_extracted + candidate_questions_inferred
    log(f"[Search] Found a total of {len(all_candidates)} candidate questions. Organising with Gemini...")

    final_output = await organise_batches_iteratively(all_candidates, batch_size=50)
    st.session_state["organized_text"] = final_output
    log("[Search] Finished organising questions with Gemini!")

###################
#   ENTRY POINT   #
###################

if __name__ == "__main__":
    main()
