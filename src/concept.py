import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def extract_concepts(text):

    # Normalize text
    text = text.lower()

    # Extract words
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    # Remove stopwords
    concepts = {
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    }

    return concepts


def tool_to_text(tool):

    return (
        tool["function"]["name"]
        + " "
        + tool["function"]["description"]
    )


def reduce_concept(instruction, tools, top_k):

	# concepts from the user query
    request_concepts = (extract_concepts(instruction))

    tool_scores = []

    for tool in tools:

    	# concepts from the tool name + description
        tool_concepts = (extract_concepts(tool_to_text(tool)))

        # the score is the size of the intersection set
        score = len(request_concepts & tool_concepts)

        tool_scores.append((score, tool))

    # Sort
    tool_scores.sort(key=lambda pair: pair[0], reverse=True)

    # Select top_k tools
    reduced_tools = [tool for score, tool in tool_scores[:top_k]]

    return reduced_tools
