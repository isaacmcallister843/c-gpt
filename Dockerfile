FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

WORKDIR /app 

RUN apt-get update 
RUN apt-get install -y stockfish pip && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -U pip
RUN pip install -e ".[cloud]"