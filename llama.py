from dotenv import load_dotenv
from llama_cloud import LlamaCloud

load_dotenv()


def get_markdown(pdf_path: str):
    client = LlamaCloud()

    file = client.files.create(
        file=pdf_path,
        purpose="parse"
    )

    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    markdown = "\n\n".join(
        page.markdown
        for page in result.markdown.pages
    )

    return markdown