:: call conda create -n llmtime python=3.9
call conda activate llmtime
call pip install ipykernel
call pip install numpy
call pip install python-dotenv
call pip install -U jax[cpu]
call pip install torch --index-url https://download.pytorch.org/whl/cu118
call pip install openai==0.28.1
call pip install tiktoken
call pip install tqdm
call pip install matplotlib
call pip install "pandas<2.0.0"
call pip install darts
call pip install gpytorch
call pip install transformers
call pip install datasets
call pip install multiprocess
call pip install SentencePiece
call pip install accelerate
call pip install gdown
call pip install mistralai
