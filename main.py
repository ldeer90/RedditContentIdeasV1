import asyncio
import re
import time
from nicegui import ui
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyCdoGJ77AtAzw9C7gf7mfk-cKDmUUgkf-4"  # Replace with YOUR API key
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"  # Using the flash-exp model

# --- Global Variables ---
process_container = None
results_container = None
organized_container = None
candidate_questions_extracted = []  # List of dicts: {"url": ..., "question": ..., "type": "extracted"}
candidate_questions_inferred = []   # List of dicts: {"url": ..., "question": ..., "type": "inferred"}
final_output = ""  # Final organised output in Markdown

def log(message):
    if process_container:
        with process_container:
            ui.label(message)

def fetch_url(url, headers, timeout=10, retries=3):
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
    missing = desired_count
    prompt = (
        f"Based on the following Reddit thread content (source: {url}), list exactly {missing} additional relevant questions rephrased as natural search queries in plain, professional Australian English. The questions should be written exactly as someone would type them into google, avoiding any clickbait or headline-like formatting. Please output your response as a numbered list. If you cannot generate exactly the requested number, list as many as possible.\n\n"
        f"Thread content (truncated to {trunc_length} characters):\n{html[:trunc_length]}"
    )
    log(f"[Infer] Prompting Gemini to infer {missing} additional questions for {url}.")
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
        render_organised_output(current_output)
        log(f"[Batch] Batch {i+1} processed.")
        await asyncio.sleep(1)
    return current_output

def render_organised_output(markdown_text):
    organized_container.clear()
    categories = [cat.strip() for cat in markdown_text.split("###") if cat.strip()]
    for cat in categories:
        lines = cat.splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        headings = []
        for line in lines[1:]:
            if line.startswith("-"):
                headings.append(line.lstrip("- ").strip())
        with organized_container:
            with ui.card().classes("mb-4 p-4 shadow-lg"):
                ui.label(f"📂 {title}").classes("text-2xl font-bold mb-2")
                for heading in headings:
                    match = re.search(r"\(\[.*\]\((.*?)\)\)", heading)
                    if match:
                        source_link = match.group(1)
                        heading_text = heading.split("(")[0].strip()
                        with ui.row().classes("ml-4 items-center"):
                            ui.icon("chevron_right").classes("text-blue-700")
                            ui.label(heading_text).classes("ml-1 text-base text-black")
                            ui.link(f"({source_link})", source_link, new_tab=True).classes("ml-1 text-blue-700 hover:bg-blue-700")
                    else:
                        with ui.row().classes("ml-4 items-center"):
                            ui.icon("chevron_right").classes("text-blue-700")
                            ui.label(heading).classes("ml-1 text-base text-black")

async def perform_search():
    global candidate_questions_extracted, candidate_questions_inferred, final_output
    candidate_questions_extracted = []
    candidate_questions_inferred = []
    query = search_input.value.strip()
    if query:
        process_container.clear()
        results_container.clear()
        organized_container.clear()
        log(f"[Search] Starting search for: '{query} site:reddit.com'")
        try:
            search_results = list(search(f"{query} site:reddit.com", num_results=int(threads_input.value)))
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
            trunc_length = int(truncate_input.value)
            truncated = html[:trunc_length] + "\n...[truncated]"
            with results_container:
                ui.label(f"[Raw] HTML content (truncated, {trunc_length} chars) from {url}:").classes("text-sm")
                ui.label(truncated).classes("text-xs")
            extracted, inferred = await get_thread_questions(html, url, int(questions_input.value), trunc_length)
            log(f"[Search] From {url}: Extracted {len(extracted)} questions; Inferred {len(inferred)} questions.")
            for q in extracted:
                candidate_questions_extracted.append({"url": url, "question": q, "type": "extracted"})
            for q in inferred:
                candidate_questions_inferred.append({"url": url, "question": q, "type": "inferred"})
            with results_container:
                ui.label(f"[Raw] Questions from {url}:").classes("font-bold")
                if extracted:
                    ui.label("Extracted Questions:").classes("underline")
                    for q in extracted:
                        ui.label(f"{q} ({url})").classes("text-base")
                else:
                    ui.label("No extracted questions found.").classes("italic")
                if inferred:
                    ui.label("Inferred Questions:").classes("underline")
                    for q in inferred:
                        ui.label(f"{q} ({url})").classes("text-base")
            await asyncio.sleep(2)
        log("[Search] Combining all candidate questions and organising them in batches via Gemini...")
        all_candidates = candidate_questions_extracted + candidate_questions_inferred
        final_output = await organise_batches_iteratively(all_candidates, batch_size=50)
        log("[Search] Organised output received. Rendering final organised layout.")
        render_organised_output(final_output)
        log("[Search] Process complete.")

def export_results():
    ui.run_javascript(f"navigator.clipboard.writeText({repr(final_output)})")

with ui.row().classes("p-4 bg-blue-50"):
    search_input = ui.input().classes("w-1/2 p-2 border rounded")
    ui.label("Enter topic").classes("ml-2")
    ui.button("Search", on_click=lambda: asyncio.create_task(perform_search())).classes("ml-4 p-2 bg-blue-600 text-white rounded hover:bg-blue-700")

with ui.row().classes("p-4 bg-blue-50"):
    ui.label("Number of threads to check:").classes("mr-2")
    threads_input = ui.input(value="30").classes("w-1/5 p-2 border rounded").props("type=number")
    ui.label("Number of questions per thread:").classes("ml-8 mr-2")
    questions_input = ui.input(value="40").classes("w-1/5 p-2 border rounded").props("type=number")
    ui.label("Truncation length for Gemini inference:").classes("ml-8 mr-2")
    truncate_input = ui.input(value="10000").classes("w-1/5 p-2 border rounded").props("type=number")

with ui.row().classes("p-4"):
    with ui.column().classes("w-1/3 mr-4 bg-white p-4 rounded shadow"):
        ui.label("Final Organised Output:").classes("text-xl font-bold text-blue-700")
        organized_container = ui.column().classes("overflow-auto")
        ui.button("Copy All Results", on_click=export_results).classes("mt-4 p-2 bg-green-600 text-white rounded hover:bg-green-700")
    with ui.column().classes("w-2/3 bg-white p-4 rounded shadow"):
        with ui.tabs() as tabs:
            with ui.tab("Process Log"):
                process_container = ui.column().classes("overflow-auto p-2")
            with ui.tab("Raw Results"):
                results_container = ui.column().classes("overflow-auto p-2")

log("[Init] Script started.")

ui.run(port=8080)
