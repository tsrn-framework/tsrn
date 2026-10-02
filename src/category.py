from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def tool_to_text(tool):
    return tool["function"]["name"] + " " + tool["function"]["description"]


def reduce_category(instruction, tools, n_clusters):

    # Convert tools into text representations
    tool_texts = [tool_to_text(tool) for tool in tools]

    # Embed tools
    tool_embeddings = model.encode(tool_texts)

    # Cluster tools
    clustering = KMeans(
        n_clusters=min(n_clusters, len(tools)),
        random_state=42,
        n_init=10,
    )

    cluster_labels = clustering.fit_predict(tool_embeddings)

    centroids = clustering.cluster_centers_

    # Embed instruction
    instruction_embedding = model.encode([instruction])

    # Find closest cluster
    similarities = cosine_similarity(instruction_embedding, centroids)[0]

    best_cluster = similarities.argmax()

    # Keep tools from the selected cluster
    reduced_tools = [
        tool
        for tool, cluster_label in zip(
            tools,
            cluster_labels,
        )
        if cluster_label == best_cluster
    ]

    return reduced_tools