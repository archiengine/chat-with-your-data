# Chat with the building code

## Overview

This is a chatbot application to use to ask questions about the department of buildings code.

Building architects spend a lot of time combing through building code. As a residential architect,
and someone who consumes the building code, I created this chatbot to make querying the building code easy!

## Requirements

Requirements to run application are:

- [VS code](https://code.visualstudio.com/)
- [Remote Development VS Code extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [If on Windows, WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Pinecone key](https://www.pinecone.io/)
- [Openai key](https://openai.com/)

## How to run

- Make sure docker is running
- Open VS Code in root directory
- VS Code will prompt for you to open the folder in a dev container. Say yes.
- Once container has been created, open a terminal window and run:

  `pip install -r requirements.txt`

- Copy .env.template file and rename to .env
- Add your keys / values to the .env file
- create pinecone index named `test-index`
- run program with this command:

  `python.main.py`