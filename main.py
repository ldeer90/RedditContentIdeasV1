import asyncio
import re
import time
import datetime
import streamlit as st
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import google.generativeai as genai

# Configure your Google API key for Gemini
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

st.set_page_config(layout="centered")
st.title("Question Forum Finder")

if "log_text" not in st.session_state:
    st.session_state.log_text = ""


def log(message):
    st.session_state.log_text += message + "\n"


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


def extract_questions(html, source="generic"):
    soup = BeautifulSoup(html, "html.parser")

    # Remove the head section entirely
    if soup.head:
        soup.head.decompose()

    # Remove common non-content tags
    for tag in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Attempt to restrict parsing to the main content container
    text = ""
    if source == "quora":
        # For Quora, the user-generated content is usually inside a div with classes like "q-box" and "quora-rich-text--renderer"
        main_content = soup.find("div", class_=lambda x: x and "q-box" in x and "quora-rich-text" in x)
        if main_content:
            text = main_content.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
    elif source == "reddit":
        # For Reddit, most user text is within a div with class "usertext-body"
        main_content = soup.find("div", class_=lambda x: x and "usertext-body" in x)
        if main_content:
            text = main_content.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
    else:
        text = soup.get_text(separator=" ")

    # Now split text into sentences and filter for questions
    sentences = re.split(r'(?<=[.?!])\s+', text)
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if "?" in sentence and len(sentence) > 25:
            lower = sentence.lower()
            if any(keyword in lower for keyword in
                   ["add to the discussion", "vote", "comment", "submit", "read more", "loading", "http", "www."]):
                continue
            if sentence not in questions:
                questions.append(sentence)
    return questions


async def assess_questions_relevance(questions, url):
    if not questions:
        return []
    prompt = (
            f"Below are candidate questions extracted from a thread (source: {url}):\n" +
            "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)]) +
            "\n\nRewrite these as professional, natural search queries in plain language that someone would type into Google. Avoid clickbait or headline-like language. Only include relevant and actionable ones. Output your answer as a numbered list."
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 300}
        )
        if not response or not response.text:
            return []
        lines = response.text.splitlines()
        rewritten = []
        for line in lines:
            line = line.strip()
            if line.lower() == "none":
                return []
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                rewritten.append(match.group(1).strip())
        log(f"[Relevance] Rewritten {len(rewritten)} extracted questions from {url}.")
        return rewritten
    except Exception as e:
        log(f"[Relevance] Error rewriting questions for {url}: {e}")
        return questions


async def get_thread_questions(html, url, desired_count, trunc_length):
    log(f"[Extract] Extracting candidate questions from {url}")
    # Use specific extraction based on source
    if "quora.com" in url:
        extracted = extract_questions(html, source="quora")
    else:
        extracted = extract_questions(html, source="generic")
    extracted = extracted[:desired_count]
    rewritten_extracted = await assess_questions_relevance(extracted, url)
    missing = desired_count
    prompt = (
            f"Based on the following thread content (source: {url}), list exactly {missing} additional relevant questions rephrased as natural search queries in plain, professional Australian English. " +
            f"Output them as a numbered list, avoiding clickbait or headline-like language. If you cannot generate all {missing}, list as many as possible.\n\n" +
            f"Thread content (truncated to {trunc_length} characters):\n{html[:trunc_length]}"
    )
    log(f"[Infer] Prompting Gemini to infer {missing} additional questions for {url}.")
    inferred = []
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 300}
        )
        if response and response.text:
            additional_text = response.text
            log(f"[Infer] Raw inferred text: {additional_text}")
            for line in additional_text.splitlines():
                line = line.strip()
                match = re.match(r'^\d+\.\s*(.*)', line)
                if match:
                    inferred.append(match.group(1).strip())
            log(f"[Infer] Inferred {len(inferred)} additional questions from {url}.")
    except Exception as e:
        log(f"[Infer] Error inferring questions for {url}: {e}")
    return rewritten_extracted, inferred


async def call_gemini(prompt, retries=10, max_tokens=2000):
    model = genai.GenerativeModel(MODEL_NAME)
    backoff_seconds = 2
    for attempt in range(retries):
        try:
            log(f"[Gemini] Attempt {attempt + 1}: Calling Gemini...")
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
                log(f"[Gemini] 429 error detected on attempt {attempt + 1}, delaying further attempts.")
            else:
                log(f"[Gemini] Error during Gemini call: {e} (Attempt {attempt + 1})")
        await asyncio.sleep(backoff_seconds)
        backoff_seconds *= 2
    raise RuntimeError(f"Gemini call failed after {retries} attempts.")


def display_questions_in_expanders(markdown_text):
    categories = [cat.strip() for cat in markdown_text.split("###") if cat.strip()]
    for cat in categories:
        lines = cat.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        content_lines = lines[1:]
        with st.expander(header):
            for line in content_lines:
                if line.strip().startswith("- "):
                    st.markdown(line)


async def organise_batches_iteratively(candidates, batch_size=50):
    if not candidates:
        return "No candidate questions available for organisation."
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    current_output = ""
    for i, batch in enumerate(batches):
        batch_text = "\n".join(
            [f"{item['question']} [Source: {item['source'].capitalize()}] ({item['url']})" for item in batch])
        if i == 0:
            prompt = (
                    "Rewrite and group the following candidate questions into categories for blog research. Within each category, group questions by their source domain. " +
                    "Each candidate question should be a natural query that someone would type into Google, avoiding clickbait or headline style. " +
                    "Output it all in Markdown with headings and bullet points for questions. The format is:\n\n" +
                    "### [Category Title]\n" +
                    "- [Rewritten Question] (Source: [Domain]) ([Source URL])\n\n" +
                    "Candidate Questions:\n" + batch_text
            )
        else:
            prompt = (
                    "Below is the previously organised output in Markdown:\n" + current_output + "\n\n" +
                    "Here are additional candidate questions:\n" + batch_text + "\n\n" +
                    "Update the organised output so that the questions are grouped by topic and also by their source within each category. " +
                    "Use the same Markdown format and avoid any clickbait or headline language."
            )
        log(f"[Batch] Processing batch {i + 1} of {len(batches)} with {len(batch)} candidate questions.")
        batch_output = await call_gemini(prompt, max_tokens=2000)
        current_output = batch_output
        log(f"[Batch] Batch {i + 1} processed.")
        await asyncio.sleep(1)
    return current_output


async def perform_search(query, threads_count, questions_per_thread, trunc_length, reddit_on, quora_on, whirlpool_on,
                         top_container, start_date, end_date):
    candidate_questions = []
    domains_to_use = []
    if reddit_on:
        domains_to_use.append("site:reddit.com")
    if quora_on:
        domains_to_use.append("site:quora.com")
    if whirlpool_on:
        domains_to_use.append("site:whirlpool.net.au")

    if not domains_to_use:
        st.write("No sites selected. Please enable at least one checkbox.")
        return

    date_modifiers = ""
    if start_date and start_date != datetime.date(1970, 1, 1):
        date_modifiers += f" after:{start_date.strftime('%Y-%m-%d')}"
    if end_date and end_date != datetime.date(1970, 1, 1):
        date_modifiers += f" before:{end_date.strftime('%Y-%m-%d')}"

    query_string = f"{query} {' OR '.join(domains_to_use)}{date_modifiers}"
    log(f"[Search] Starting search for: '{query_string}'")

    try:
        search_results = list(search(query_string, num_results=threads_count))
        log(f"[Search] Found {len(search_results)} URLs for query: {query_string}")
    except Exception as e:
        log(f"[Search] Error during Google search: {e}")
        return

    for url in search_results:
        if not any(domain[5:] in url for domain in domains_to_use):
            log(f"[Search] Skipping non-target URL: {url}")
            continue
        if "reddit.com" in url and "old.reddit.com" not in url:
            url = url.replace("www.reddit.com", "old.reddit.com")
        html = fetch_url(url, {"User-Agent": "Mozilla/5.0"})
        if not html:
            log(f"[Search] Skipping {url} due to fetch failure.")
            continue
        log(f"[Search] Fetched HTML from {url} (length: {len(html)})")

        truncated_html = html[:trunc_length] + "\n...[truncated]"
        st.text_area(f"[Raw] HTML content from {url} (truncated to {trunc_length} chars)", truncated_html, height=200)

        extracted, inferred = await get_thread_questions(html, url, questions_per_thread, trunc_length)
        log(f"[Search] From {url}: Extracted {len(extracted)} questions; Inferred {len(inferred)} questions.")

        source_label = "reddit"
        if "quora.com" in url:
            source_label = "quora"
        elif "whirlpool.net.au" in url:
            source_label = "whirlpool"

        for q in extracted:
            candidate_questions.append({"url": url, "question": q, "type": "extracted", "source": source_label})
        for q in inferred:
            candidate_questions.append({"url": url, "question": q, "type": "inferred", "source": source_label})

        st.write(f"[Raw] Questions from {url}:")
        if extracted:
            st.write("Extracted Questions:")
            for question_text in extracted:
                st.write(f"{question_text} ({url})")
        else:
            st.write("No extracted questions found.")
        if inferred:
            st.write("Inferred Questions:")
            for question_text in inferred:
                st.write(f"{question_text} ({url})")
        await asyncio.sleep(2)

    log("[Search] Combining all candidate questions and organising them...")
    final_output = await organise_batches_iteratively(candidate_questions, batch_size=50)
    with top_container:
        st.subheader("Organised Questions")
        display_questions_in_expanders(final_output)
        st.download_button("Download Final Organised Output", final_output, "results.md")
    log("[Search] Process complete.")


top_output_container = st.container()

reddit_on = st.checkbox("Search Reddit", value=True)
quora_on = st.checkbox("Search Quora", value=True)
whirlpool_on = st.checkbox("Search Whirlpool", value=True)

query = st.text_input("Enter topic", "")
threads_count = st.number_input("Number of threads to check", min_value=1, value=30)
questions_per_thread = st.number_input("Number of questions per thread", min_value=1, value=40)
trunc_length = st.number_input("Truncation length for Gemini inference", min_value=1000, value=10000)

start_date = st.date_input("Start date (optional)", value=None, min_value=datetime.date(2020, 1, 1))
end_date = st.date_input("End date (optional)", value=None, min_value=datetime.date(2020, 1, 2))

if st.button("Search"):
    asyncio.run(perform_search(
        query,
        threads_count,
        questions_per_thread,
        trunc_length,
        reddit_on,
        quora_on,
        whirlpool_on,
        top_output_container,
        start_date,
        end_date
    ))

st.subheader("Process Log")
st.text_area("", st.session_state.log_text, height=200)
