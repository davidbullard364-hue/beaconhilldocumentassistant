import hmac
import json
import logging
import os
from pathlib import Path
from time import monotonic

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

import rag
import repository as repo

TITLE = "Beacon Hill Document Assistant"
PAGES = ("Welcome", "Document Library", "Document Assistant", "Executive Summaries", "Owner Tools")
STARTERS = (
    "How do I file a complaint?",
    "Summarize the bylaws",
    "What records does the association retain?",
    "Explain the conflict-of-interest policy",
)
LOG = logging.getLogger(__name__)

st.set_page_config(page_title=TITLE, layout="wide")


def setting(name, default=""):
    if name in os.environ:
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except StreamlitSecretNotFoundError:
        return default


ROOT = Path(setting("RAG_STORAGE_DIR", str(Path(__file__).with_name("data")))).expanduser()


def hero():
    st.html("""
    <style>
    .stMainBlockContainer { max-width: none; padding: 2rem clamp(1rem, 3vw, 3rem); }
    .beacon-hero { padding: 1.25rem 1.4rem; border: 1px solid #ddd5c4;
      border-left: 5px solid #a48159; border-radius: 20px;
      background: linear-gradient(110deg, #fcfaf5, #efe7d4); color: #252821; }
    .beacon-hero h1 { font: 700 clamp(2rem, 4vw, 3.5rem)/1.1 Georgia, serif;
      color: inherit; padding: 0; margin: 0 0 .55rem; }
    .beacon-hero p { font-size: clamp(1.1rem, 1.6vw, 1.35rem); margin: .35rem 0; }
    .st-key-beacon-nav { padding-bottom: .6rem; border-bottom: 1px solid #ddd5c4; }
    </style>
    <header class="beacon-hero">
      <h1>Beacon Hill Document Assistant</h1>
      <p>Ask questions about community documents and get executive summaries with sources.</p>
      <p><strong>Community resource maintained by a resident.</strong>
      Check cited documents when making decisions.</p>
    </header>
    """)


def welcome():
    with st.container(border=True):
        st.subheader("Welcome to the Beacon Hill Document Assistant")
        st.markdown(
            "**Document Library** · Browse an alphabetical list of uploaded and indexed documents.\n\n"
            "**Document Assistant** · Ask questions about the collection, choose a starter, "
            "or ask to see the wording behind an answer.\n\n"
            "**Executive Summaries** · Generate a plain-language summary with citations "
            "and ask follow-up questions about the selected document."
        )


def sorted_documents(documents):
    return sorted(documents, key=lambda item: item["title"].casefold())


def download(item, key):
    path = repo.pdf_path(ROOT, item)
    if path is None:
        return
    st.download_button(
        "Download PDF", data=path.read_bytes, file_name=f"{item['title']}.pdf",
        mime="application/pdf", key=key, on_click="ignore",
    )


def library(documents):
    st.subheader("Document Library")
    st.caption(f"{len(documents)} indexed documents")
    for number, item in enumerate(sorted_documents(documents)):
        with st.expander(item["title"]):
            st.text(f"{len(item['pages'])} indexed pages")
            if repo.pdf_path(ROOT, item):
                download(item, f"library-pdf-{number}")
            else:
                st.caption("Indexed text is available in the assistant.")


def ask(question, history, document, documents):
    question = question.strip()
    if not 0 < len(question) <= 1500:
        st.warning("Enter a question of up to 1,500 characters.")
        return None
    with st.spinner("Preparing your answer…"):
        try:
            return rag.answer(
                question, history, document, documents,
                setting("OPENAI_API_KEY").strip(),
                setting("OPENAI_MODEL").strip() or "gpt-4.1-mini",
            )
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            LOG.warning("Answer failed: %s (status=%s, code=%s)", type(error).__name__,
                        getattr(error, "status_code", None), getattr(error, "code", None))
            st.error("The assistant could not answer right now. Please try again.")
    return None


def show_results(history, documents, key, label):
    st.markdown(f"**{label}**")
    active = {item["title"]: item for item in documents}
    with st.container(height=440, border=True, key=f"{key}-{len(history)}"):
        if not history:
            st.caption("Your questions and answers will appear here.")
        for turn, (question, answer, sources) in reversed(list(enumerate(history))):
            st.text(question)
            st.markdown(answer)
            for source in sources:
                title = source["document"]
                st.text(f"[{source['source']}] {title} · PDF page {source['pdf_page']}")
                item = active.get(title)
                if item and item.get("pdf_id") == source.get("pdf_id"):
                    download(item, f"{key}-{turn}-{source['source']}")
            if turn:
                st.divider()


def assistant(documents):
    history = st.session_state.setdefault("chat", [])
    sidebar, conversation = st.columns([1, 4])
    prompt = None
    with sidebar:
        with st.container(border=True):
            st.markdown("**Try a question**")
            for starter in STARTERS:
                if st.button(starter, width="stretch"):
                    prompt = starter
            st.markdown("**Answer prompts**")
            if st.button("Explain more simply", disabled=not history, width="stretch"):
                prompt = "Explain your previous answer more simply."
            if st.button("Show supporting wording", disabled=not history, width="stretch"):
                prompt = "Show the exact wording that supports your previous answer."
            last = st.session_state.get("last_question")
            retry = st.button("Retry last question", disabled=not last, width="stretch")
            if retry:
                prompt = last
            if st.button("New conversation", width="stretch"):
                history.clear()
                st.session_state.pop("last_question", None)
    with conversation:
        with st.form("ask-question", clear_on_submit=True, border=False):
            question = st.text_area("Your question", placeholder="What would you like to understand?",
                                    max_chars=1500, height=80)
            if st.form_submit_button("Ask", type="primary"):
                prompt = question
        if prompt is not None:
            if prompt.strip():
                st.session_state.last_question = prompt.strip()
            replace_last = retry and bool(history) and history[-1][0] == prompt
            turn = ask(prompt, history[:-1] if replace_last else history, rag.ALL_DOCUMENTS, documents)
            if turn:
                if replace_last:
                    history.pop()
                history.append(turn)
        show_results(history, documents, "assistant-results", "Document conversation")


def summaries(documents):
    st.subheader("Executive Summaries")
    titles = [item["title"] for item in sorted_documents(documents)]
    selected = st.selectbox("Document", titles, index=None, placeholder="Choose a document")
    if st.button("Generate executive summary", type="primary", disabled=not selected):
        turn = ask(f"Write a plain-language executive summary of {selected}.", [], selected, documents)
        if turn:
            st.session_state.summary_title = selected
            st.session_state.summary = [turn]
    ready = bool(selected) and selected == st.session_state.get("summary_title")
    history = st.session_state.get("summary", []) if ready else []
    results = st.container()
    with st.form("follow-up", clear_on_submit=True, border=False):
        prompt = st.text_input("Follow-up question", disabled=not ready, max_chars=1500)
        submitted = st.form_submit_button("Ask follow-up", disabled=not ready)
    if submitted and ready:
        turn = ask(prompt, history, selected, documents)
        if turn:
            history.append(turn)
    with results:
        show_results(history, documents, "summary-results", "Executive summary")


def owner():
    st.subheader("Owner Tools")
    password = setting("RAG_ADMIN_PASSWORD")
    if not password:
        st.info("The owner needs to configure RAG_ADMIN_PASSWORD.")
        return

    password_token = hmac.digest(password.encode(), b"beacon-owner-auth", "sha256")
    grant = st.session_state.get("owner_grant")
    if not grant or grant[0] <= monotonic() or not hmac.compare_digest(grant[1], password_token):
        st.session_state.pop("owner_grant", None)
        with st.form("owner-login", clear_on_submit=True):
            candidate = st.text_input("Owner password", type="password")
            if st.form_submit_button("Unlock Owner Tools"):
                if hmac.compare_digest(candidate.encode(), password.encode()):
                    st.session_state.owner_grant = (monotonic() + 900, password_token)
                    st.rerun()
                st.error("Incorrect owner password.")
        return
    if st.button("Lock Owner Tools"):
        st.session_state.pop("owner_grant", None)
        st.rerun()

    replace = st.checkbox("Replace matching titles")
    pdfs = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index PDFs", type="primary", disabled=not pdfs):
        if len(pdfs) > 50:
            st.warning("Choose up to 50 PDFs at a time.")
        else:
            seen = set()
            progress = st.empty()
            for number, pdf in enumerate(pdfs, 1):
                progress.text(f"Indexing {number}/{len(pdfs)}: {pdf.name}")
                try:
                    item, data = repo.index_pdf(pdf)
                    title = item["title"].casefold()
                    if title in seen:
                        st.warning(f"Skipped duplicate title: {item['title']}")
                        continue
                    st.text(repo.save(ROOT, [(item, data)], replace)[0])
                    seen.add(title)
                except ValueError as error:
                    st.warning(f"{pdf.name}: {error}")
                except Exception as error:
                    LOG.warning("PDF indexing failed: %s", type(error).__name__)
                    st.error(f"{pdf.name}: indexing failed.")
            progress.empty()

    index = st.file_uploader("Import a Beacon Hill JSON index", type=["json"])
    if st.button("Import index", disabled=index is None):
        try:
            if index.size > repo.LIMIT:
                raise ValueError("Index exceeds 50 MiB.")
            documents = repo.validate(json.loads(index.getvalue().decode("utf-8-sig")))
            for message in repo.save(ROOT, [(item, None) for item in documents], replace):
                st.text(message)
            if not documents:
                st.info("The JSON index had no entries.")
        except ValueError as error:
            st.warning(str(error))
        except OSError as error:
            LOG.warning("Index import failed: %s", type(error).__name__)
            st.error("Could not save the imported index. Check storage configuration.")
    with st.form("clear-collection"):
        confirmation = st.text_input("Type CLEAR to confirm")
        if st.form_submit_button("Clear collection"):
            if confirmation != "CLEAR":
                st.warning("Type CLEAR to confirm.")
            else:
                try:
                    repo.clear(ROOT)
                    st.success("The active collection was cleared.")
                except OSError as error:
                    LOG.warning("Collection clear failed: %s", type(error).__name__)
                    st.error("Could not clear the collection. Check storage configuration.")


def main():
    hero()
    with st.container(key="beacon-nav"):
        page = st.radio("Navigate", PAGES, horizontal=True, label_visibility="collapsed")
    if page == "Welcome":
        welcome()
    elif page == "Owner Tools":
        owner()
    else:
        try:
            documents = repo.read(ROOT)
        except (OSError, ValueError) as error:
            LOG.warning("Index load failed: %s", type(error).__name__)
            st.error("The document index could not be loaded.")
            return
        if page == "Document Library":
            library(documents)
        elif page == "Document Assistant":
            assistant(documents)
        else:
            summaries(documents)
    st.caption("Answers are generated from the current collection. Interpretations may require further review.")


if __name__ == "__main__":
    main()
