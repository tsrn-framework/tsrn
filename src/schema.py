from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def tool_to_schema_text(tool):

    text_parts = []

    parameters = (
        tool["function"]
        .get("parameters", {})
        .get("properties", {})
    )

    # key: parameter name, value: parameter type and string
    for parameter_name, parameter_info in parameters.items():

        text_parts.append(parameter_name)

        description = parameter_info.get("description","")

        text_parts.append(description)

        parameter_type = parameter_info.get("type","")

        text_parts.append(parameter_type)

    return " ".join(text_parts)


def reduce_schema(instruction, tools, top_k):

    # user request embedding
    instruction_embedding = model.encode([instruction])


    # extract tool arguments names and descriptions
    schema_texts = [tool_to_schema_text(tool) for tool in tools]

    # tool argument embeddings
    schema_embeddings = model.encode(schema_texts)


    # calculate similarities
    similarities = cosine_similarity(instruction_embedding, schema_embeddings)[0]

    # sort
    ranked_indices = (similarities.argsort()[::-1])

    # select top_k tools
    reduced_tools = [tools[i] for i in ranked_indices[:top_k]]

    return reduced_tools
