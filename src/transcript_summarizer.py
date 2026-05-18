import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def _set_mistral_api_key(mistral_api_key: str) -> None:
    os.environ["MISTRAL_API_KEY"] = mistral_api_key


def build_chain(system_prompt: str, mistral_api_key: str):
    _set_mistral_api_key(mistral_api_key)
    llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str, mistral_api_key: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract all action items from the transcript. "
        "For each provide:\n- Task description\n- Owner (who is responsible)\n- Deadline (if mentioned)\n"
        "Format as a numbered list. If none found, say 'No action items found.'",
        mistral_api_key
    )

    result = chain.invoke({"text": transcript})

    return result


def extract_key_decisions(transcript: str, mistral_api_key: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract all key decisions made. "
        "Format as a numbered list. If none found, say 'No key decisions found.'",
        mistral_api_key
    )
    result = chain.invoke({"text": transcript})

    return result


def extract_questions(transcript: str, mistral_api_key: str) -> str:
    chain = build_chain(
        "Extract all unresolved questions or topics needing follow-up from the transcript. "
        "Format as a numbered list. If none found, say 'No open questions found.'"
        "DONT make up questions - only extract those that are explicitly mentioned as open or needing follow-up.",
        mistral_api_key
    )
    result = chain.invoke({"text": transcript})

    return result


# main summarization function that uses the above helper functions
def summarize_transcript(transcript: str, mistral_api_key: str) -> dict:
    _set_mistral_api_key(mistral_api_key)
    llm = ChatMistralAI(model_name="mistral-small-latest", temperature=0.2)

    # 1. Summarize individual chunks
    map_prompt = ChatPromptTemplate.from_template(
        "Summarize this part of a video transcript concisely:\n\n{text}"
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    print(f"[INFO] Summarizing {len(chunks)} chunks...")

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    # 2. Combine partial summaries into one text
    combined_partial_summaries = "\n\n".join(chunk_summaries)

    # 3. Combine into final bullet points
    final_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional video summarizer. 
        Combine these partial summaries into one final professional video summary using bullet points and give suitable title.
        
        Summaries:
        {text}
        """
    )
    
    final_chain = final_prompt | llm | StrOutputParser()

    # 4. Full summary
    full_summary = final_chain.invoke(
        {"text": combined_partial_summaries})

    with open("full_summary.txt", "w", encoding="utf-8") as f:
        f.write(full_summary)

    # 5. Extract actionable items, key decisions, and open questions from the combined partial summaries
    actionable_items = extract_action_items(combined_partial_summaries, mistral_api_key)
    key_decisions = extract_key_decisions(combined_partial_summaries, mistral_api_key)
    open_questions = extract_questions(combined_partial_summaries, mistral_api_key)

    return {
        "summary": full_summary,
        "action_items": actionable_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions
    }
