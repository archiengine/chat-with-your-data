This is a chatbot application to use to ask questions about the NYS Department of Buildings code.

Requirements to run application are:
VS Code; Remote Development VS Code Extension; Docker Desktop; If on Windows, WSL 2; Pinecone; Openai

How to run:
run docker, open VS Code in root directory; once container has been created, open a terminal window and run `pip install -r requirements.txt`; copy the .env.template file and rename to .env; create pinecone index named `test-index`; run program with command `python.main.py`