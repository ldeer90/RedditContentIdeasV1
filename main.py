import os
import re
import time
import asyncio
import requests
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"  # Replace with YOUR API key
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"  # Using the flash-exp model
import os
import re
import time
import asyncio
import requests
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from googlesearch import search

# Configure Google Generative AI.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "REPLACE_WITH_YOUR_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

# Global variables for candidate questions.
candidate_questions_extracted = []
candidate_questions_inferred = []
final_output = ""

# A simple logger that appends messages to a session-state list.
def log(message):
    st.session_state["log_messages"].append(message)

def fetch_url(url, headers, timeout=10, retries=3):
    for attempt in range(retries):
        log(f"[Fetch] Attempt {attempt + 1}: Fetching {url}")
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            log(f"[Fetch] {url} returned HTTP status {response.status_code}")
            return response.text
        except requests.exceptions.RequestException as e:
            log(f"[Fetch] Error on attempt {attempt + 1} for {url}: {e}")
            time.sleep(2)
    log(f"[Fetch] Failed to fetch {url} after {retries} attempts.")
    return None

def extract_questions(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            lower = sentence.lower()
            if any(keyword in lower for keyword in [
                "add to the discussion", "vote", "comment", "submit", "read more", "loading", "http"
            ]):
                continue
            if sentence not in questions:
                questions.append(sentence)
    return questions

async def assess_questions_relevance(questions, url):
    if not questions:
        return []
    prompt = (
        f"Below are candidate questions extracted from a Reddit thread (source: {url}):\n" +
        "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]) +
        "\n\nRewrite these as professional, natural search queries in plain language that someone would actually type into google. Avoid clickbait or headline-like language. Only include the questions that are highly relevant and actionable. Output your answer as a numbered list."
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
                if len(parts) > 1:
                    rewritten.append(parts[1].strip())
                else:
                    rewritten.append(line)
        log(f"[Relevance] Rewritten {len(rewritten)} extracted questions from {url}.")
        return rewritten
    except Exception as e:
        log(f"[Relevance] Error rewriting questions for {url}: {e}")
        return questions

async def get_thread_questions(html, url, desired_count, trunc_length):
    log(f"[Extract] Extracting candidate questions from {url}")
    extracted = extract_questions(html)[:desired_count]
    rewritten_extracted = await assess_questions_relevance(extracted, url)
    inferred = []
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {desired_count} additional relevant questions rephrased as natural search queries in plain, professional Australian English. The questions should be written exactly as someone would type them into google, avoiding any clickbait or headline-like formatting. Please output your response as a numbered list. If you cannot generate exactly the requested number, list as many as possible.\n\n"
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
            if line and re.match(r'^\d+\.', line):
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    inferred.append(parts[1].strip())
                else:
                    inferred.append(line)
        log(f"[Infer] Inferred {len(inferred)} additional questions from {url}.")
    except Exception as e:
        log(f"[Infer] Error inferring questions for {url}: {e}")
    return rewritten_extracted, inferred

async def call_gemini(prompt, retries=10, max_tokens=2000):
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
                log("[Gemini] Gemini call did not return a valid result.")
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                log(f"[Gemini] 429 error detected on attempt {attempt+1}, delaying further attempts.")
            else:
                log(f"[Gemini] Error during Gemini call: {e} (Attempt {attempt+1})")
        await asyncio.sleep(backoff_seconds)
        backoff_seconds *= 2
    raise RuntimeError(f"Gemini call failed after {retries} attempts.")

async def organise_batches_iteratively(candidates, batch_size=50):
    if not candidates:
        return "No candidate questions available for organisation."
    batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    current_output = ""
    for i, batch in enumerate(batches):
        batch_text = "\n".join([f"{item['question']} [{item['type'].capitalize()}] ({item['url']})" for item in batch])
        if i == 0:
            prompt = (
                "Rewrite and group the following candidate questions into multiple, specific topical categories for blog research. "
                "Each candidate question should be rewritten as a natural search query in plain language exactly as someone would type into google, avoiding any clickbait or headline-like language. "
                "Group the questions into topics that make sense for someone researching the subject, and output the final result in the exact Markdown format shown below. The output should only contain questions.\n\n"
                "### [Category Title]\n"
                "- [Rewritten Question] ([reddit thread URL])\n\n"
                "Candidate Questions:\n" + batch_text
            )
        else:
            prompt = (
                "Below is the previously organised output in Markdown:\n" + current_output + "\n\n"
                "Now, here are additional candidate questions:\n" + batch_text + "\n\n"
                "Update the organised output to incorporate the new candidate questions. Rewrite all candidate questions as natural search queries in plain language exactly as someone would type into google. Avoid any clickbait or headline-like language. Group the questions into appropriate topics and output the result in the exact Markdown format as before, containing only questions."
            )
        log(f"[Batch] Processing batch {i+1} of {len(batches)} with {len(batch)} candidate questions.")
        batch_output = await call_gemini(prompt, max_tokens=2000)
        current_output = batch_output
        st.session_state["organized_text"] = current_output
        log(f"[Batch] Batch {i+1} processed.")
        await asyncio.sleep(1)
    return current_output

def main():
    st.title("Reddit Research with Gemini")
    # Initialize session state values if not present.
    if "log_messages" not in st.session_state:
        st.session_state["log_messages"] = []
    if "organized_text" not in st.session_state:
        st.session_state["organized_text"] = ""
    
    st.write("Enter your search topic below. The app will search Reddit, extract questions, refine them with Gemini, and organise the final output in Markdown.")

    # Using keys to persist input values.
    query = st.text_input("Enter topic", key="query")
    threads_count = st.number_input("Number of threads to check", min_value=1, value=10, key="threads_count")
    questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=10, key="questions_per_thread")
    truncate_len = st.number_input("Truncation length for Gemini inference", min_value=100, value=10000, key="truncate_len")
    start_button = st.button("Search")

    tabs = st.tabs(["Process Log", "Final Organised Output"])
    with tabs[0]:
        st.write("Below is the real-time process log:")
        for msg in st.session_state["log_messages"]:
            st.write(msg)
    with tabs[1]:
        st.write("Organised Questions in Markdown:")
        st.markdown(st.session_state["organized_text"])

    if start_button:
        # Debug: Log the received query value.
        log(f"[Debug] Query received: '{st.session_state.get('query', '')}'")
        st.session_state["log_messages"].clear()
        st.session_state["organized_text"] = ""
        asyncio.run(run_search(st.session_state.get("query", ""), threads_count, questions_per_thread, truncate_len))

async def run_search(query, threads_count, questions_count, truncate_len):
    global candidate_questions_extracted, candidate_questions_inferred, final_output
    candidate_questions_extracted = []
    candidate_questions_inferred = []
    if not query.strip():
        log("[Error] Please enter a valid topic.")
        return

    log(f"[Search] Starting search for: '{query} site:reddit.com'")
    try:
        search_results = list(search(f"{query} site:reddit.com", num_results=threads_count))
        log(f"[Search] Found {len(search_results)} URLs for query: {query}")
    except Exception as e:
        log(f"[Search] Error during Google search: {e}")
        return

    for url in search_results:
        if "reddit.com" not in url:
            log(f"[Search] Skipping non-Reddit URL: {url}")
            continue
        log(f"[Search] Processing URL: {url}")
        if "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")

        html = await asyncio.to_thread(fetch_url, url, {"User-Agent": "Mozilla/5.0"})
        if not html:
            log(f"[Search] Skipping {url} due to fetch failure.")
            continue

        log(f"[Search] Fetched HTML from {url} (length: {len(html)})")
        _, inferred = await get_thread_questions(html, url, questions_count, truncate_len)
        # For this example, we only accumulate inferred questions.
        for q in inferred:
            candidate_questions_inferred.append({"url": url, "question": q, "type": "inferred"})
        await asyncio.sleep(1)

    all_candidates = candidate_questions_extracted + candidate_questions_inferred
    log("[Search] Combining all candidate questions and organising them in batches via Gemini...")
    final_output = await organise_batches_iteratively(all_candidates, batch_size=50)
    st.session_state["organized_text"] = final_output
    log("[Search] Process complete.")

if __name__ == "__main__":
    main()
