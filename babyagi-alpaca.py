import os
import asyncio
import pinecone
import time
import sys
import openai
import threading
from pydantic import BaseModel, Field
from collections import deque
from uuid import UUID, uuid4
from typing import Dict, List, Generator, Optional, Mapping, Any, Union
from langchain.llms.base import LLM, Generation, LLMResult, BaseLLM
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import subprocess

model_path = "/home/drusepth/dalai/alpaca/models/7B/ggml-model-q4_0.bin"
sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

PRINT_MODEL_THOUGHTS_LIVE = True

class ChatParameters(BaseModel):
    model: str = Field(default=model_path)
    temperature: float = Field(default=0.2)

    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)

    max_length: int = Field(default=256)

    repeat_last_n: int = Field(default=64)
    repeat_penalty: float = Field(default=1.3)

class Question(BaseModel):
    question: str
    answer: str

class Chat(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created: datetime = Field(default_factory=datetime.now)
    questions: Optional[List[Question]]
    parameters: ChatParameters

class Llama(BaseLLM, BaseModel):
    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        response = ""
        async for token in generate(prompt=prompts[0]):
            response += token
            self.callback_manager.on_llm_new_token(token, verbose=True)

        generations = [[Generation(text=response)]]
        return LLMResult(generations=generations)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        result = None

        def run_coroutine_in_new_loop():
            nonlocal result
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(self._agenerate(prompts, stop))
            finally:
                new_loop.close()

        result_thread = threading.Thread(target=run_coroutine_in_new_loop)
        result_thread.start()
        result_thread.join()

        print("result from within llama:")
        print(result)
        print("==================")

        return result

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self._generate([prompt], stop)
        return result.generations[0][0].text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "llama"

# Set Variables
load_dotenv()

# Set API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
assert PINECONE_ENVIRONMENT, "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"

# Print OBJECTIVE
print("\033[96m\033[1m"+"\n*****OBJECTIVE*****\n"+"\033[0m\033[0m")
print(OBJECTIVE)

# Configure Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Configure OpenAI for embeddings (TODO find local solution for this too)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

START_PINECONE_FROM_SCRATCH = False
if START_PINECONE_FROM_SCRATCH:
    pinecone.delete_index(table_name)
    pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

# Alpaca integration
async def generate(
    prompt: str,
    #model: str = "ggml-alpaca-13b-q4.bin",
    #model: str = model_path,
    model: str = "/home/drusepth/dalai/alpaca/models/7B/ggml-model-q4_0.bin",
    n_predict: int = 300,
    temp: float = 0.8,
    top_k: int = 10000,
    top_p: float = 0.40,
    repeat_last_n: int = 100,
    repeat_penalty: float = 1.2,
    chunk_size: int = 4,  # Define a chunk size (in bytes) for streaming the output bit by bit
):
    args = (
        #"./llama",
        "/home/drusepth/dalai/alpaca/main",
        "--model",
        "/home/drusepth/dalai/alpaca/models/7B/ggml-model-q4_0.bin", #"" + model,
        "--prompt",
        prompt,
        "--n_predict",
        str(n_predict),
        "--temp",
        str(temp),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--repeat_last_n",
        str(repeat_last_n),
        "--repeat_penalty",
        str(repeat_penalty),
        "--threads",
        "8",
    )
    #print(args)
    procLlama = await asyncio.create_subprocess_exec(
        *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    answer = ""
    buffer = b""

    while True:
        chunk = await procLlama.stdout.read(chunk_size)
        if not chunk:
            return_code = await procLlama.wait()

            if return_code != 0:
                error_output = await procLlama.stderr.read()
                raise ValueError(error_output.decode("utf-8"))
            else:
                return

        chunk = chunk.decode("utf-8")

        if PRINT_MODEL_THOUGHTS_LIVE:
            sys.stdout.write(chunk)
            sys.stdout.flush()

        answer += chunk

        yield remove_matching_end(prompt, chunk)

def remove_matching_end(a: str, b: str):
    index = b.find(a)
    if index != -1:
        return b[index + len(a):]
    return b

async def llama_call(prompt: str, stop: Optional[List[str]] = None) -> str:
    return llama._call(prompt, stop)

async def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = f"You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create a list of new tasks to be completed by the AI system that do not overlap with incomplete tasks. Print print each task on a new line.\n\nTasks:\n"
    response = await llama_call(prompt)

    new_tasks = response.split("\n")
    print("split to find " + str(len(new_tasks)) + " new tasks")
    return [{"task_name": task_name} for task_name in new_tasks]

async def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = f"You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as an ordered list with each task on its own line, in order of priority.\n\nPrioritized tasks:\n"
    response = await llama_call(prompt)
    new_tasks = response.split("\n")
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

    print("Tasks reprioritized:")
    print(task_list)

async def execution_agent(objective: str, task: str) -> str:
    context = context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"

    # TODO: this is where we should insert langchain and its agents to actually execute the task, rather than talk about how to do it

    return await llama_call(prompt)

def context_agent(query: str, index: str, n: int):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n, include_metadata=True)
    # print("***** RESULTS *****")
    # print(results)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata['task'])) for item in sorted_results]

def get_ada_embedding(text):
    #return get_embedding(text)
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def get_embedding(text: str):
    text = text.replace("\n", " ")
    embedding = sbert_model.encode([text])[0]
    return embedding.tolist()

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)
# Main loop
task_id_counter = 1

llama = Llama()

async def main_loop():
    task_id_counter = 1
    while True:
        if task_list:
            # Print the task list
            print("\033[95m\033[1m" + "\n*****QUEUED TASK LIST*****\n" + "\033[0m\033[0m")
            for t in task_list:
                print(str(t['task_id']) + ": " + t['task_name'])

            # Step 1: Pull the first task
            task = task_list.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
            print(str(task['task_id']) + ": " + task['task_name'])

            # Send to execution function to complete the task based on the context
            print("\033[92m\033[1m" + "\n*****ATTEMPTING TO COMPLETE TASK*****\n" + "\033[0m\033[0m")
            result = await execution_agent(OBJECTIVE, task["task_name"])
            this_task_id = int(task["task_id"])
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
            print(result)

            # Step 2: Enrich result and store in Pinecone
            enriched_result = {'data': result}  # This is where you should enrich the result if needed

            result_id = f"result_{task['task_id']}"
            vector = enriched_result['data']  # extract the actual result from the dictionary
            index.upsert([(result_id, get_ada_embedding(vector), {"task": task['task_name'], "result": result})])

        # Step 3: Create new tasks and reprioritize task list
        print("\033[92m\033[1m" + "\n*****CREATING NEW TASKS*****\n" + "\033[0m\033[0m")
        new_tasks = await task_creation_agent(OBJECTIVE, enriched_result, task["task_name"], [t["task_name"] for t in task_list])

        print(str(len(new_tasks)) + " new tasks created: " + str(new_tasks))
        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)

        # Step 4: Reprioritize task list
        print("\033[92m\033[1m" + "\n*****REPRIORITIZING TASK LIST*****\n" + "\033[0m\033[0m")
        await prioritization_agent(this_task_id)

        print("Sleeping for a second before re-looping")
        time.sleep(1)  # Sleep before checking the task list again

# Run the main loop asynchronously
asyncio.run(main_loop())


# async def test_agent():
#     test_response = await llama_call("Your name is Bob. Please repeat your name back:")
#     print("response = " + str(test_response) + " len=" + str(len(test_response)))
#     return test_response

# test_scope = asyncio.run(test_agent())
# print("test_scope = " + str(test_scope) + " len=" + str(len(test_scope)))