from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def tool_to_text(tool):

    text = tool["function"]["name"]
    text += " "
    text += tool["function"]["description"]

    return text


def reduce_semantic(instruction, tools, top_k=5):

    # Create one document per tool
    tool_texts = [tool_to_text(tool) for tool in tools]

    # Embed instruction and tools
    instruction_embedding = model.encode([instruction])

    tool_embeddings = model.encode(tool_texts)

    # Compute cosine similarity
    similarities = cosine_similarity(instruction_embedding, tool_embeddings)[0]

    # Rank tools
    ranked_indices = similarities.argsort()[::-1]

    # Return top-k tools
    selected_tools = [tools[i] for i in ranked_indices[:top_k]]

    return selected_tools