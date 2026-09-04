# Agenda

- **LangChain Introduction**
  - Key Features
  - Why Learn
  - Alternatives

- **Course Objective**
  - Fundamentals, RAG, Agents

- **LangChain Components**
  - Models
  - Prompts: Dynamic, Role-Based, Few-Shot, etc.
  - Chains: Sequential, Parallel, Conditional
  - Indexes: Doc Loaders, Text Splitters, Vector Stores, Retrievers
  - Memory
  - Agents

- **Models**
  - Language: LLMs (Base), Chat Models (Conversational) — [see comparison table](https://github.com/SagarChhabriya/LangChain/tree/main/03-models#2chat-models) 
  - Embedding Models
  - Open Source vs Closed Source: When to Use Which

- **Prompts**
  - Static vs Dynamic
  - Prompt Templates
  - Messages: System, Human, AI
  - Chat Prompt Templates
    - `.invoke()`
      - Single message: Single Turn
      - List of messages: Multi Turn
    - Message Placeholder

- **Structured Output**
  - Practice: returning responses in well-defined formats (e.g., JSON) vs free-form text
  - Methods: With Structured Output LLMs, Without Structured Output LLMs
  - Formats: TypedDict (Single, Automated, Literal, complex), Pydantic, JSON Schema
  - Issue: TypedDict limitations; Solution: Pydantic
  - JS frontend + Python backend incompatibility with TypedDict/Pydantic → use JSON Schema

- **Output Parsers**
  - Convert raw LLM responses to structured formats (JSON, CSV, Pydantic, etc.)
  - Paid models (OpenAI, Anthropic, Gemini) mostly support structured output; most open-source don’t by default
  - Common Parsers: StrOutputParser, JSONOutputParser, StructuredOutputParser, PydanticOutputParser

- **Chains**
  - Sequence of steps processing inputs through components
  - `chain = prompt | llm | output_parser`
  - Types: LLMChain, SimpleSequentialChain, SequentialChain, RouterChain, MultiPromptChain, MultiRouteChain, ConditionalChain, TransformChain, APIChain, etc.

- **Runnables**
  - Standard interface for processing input/output
  - Part of LangChain Expression Language (LCEL) framework for chain composition
  - Components with `.invoke()` (or `.stream()`, `.batch()`) are Runnables
  - Types: RunnableLambda, RunnableMap, RunnableSequence (Sequential chain), RunnableParallel (Parallel chain), RunnableBranch (Conditional chain), RunnablePassthrough, RunnableRetry, RunnableWithFallbacks, RunnableEach, RunnableAssign, RunnablePick, RunnableConfigurationFields
  - LCEL features: Component Integration, Stream & Sync support, Parallelization, Flexibility, Composability, Unified Interface (`invoke()`, `ainvoke()`)

- **Document Loaders**
  - RAG: Document Loaders, Text Splitters, Vector DB, Retrievers
  - Loaders: TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader, etc.
  - `load()` vs `lazyload()`

- **Text Splitters**
  - Length-Based, Text Structure-Based, Document Structure-Based, Semantic Meaning-Based

- **Vector Stores**
  - Store and retrieve data as numerical vectors
  - Features: Storage, Similarity Search, Indexing, CRUD operations


---


# **Introduction to LangChain**  

## **Foundation Models**  
- **User Perspective**  
- **Builder Perspective**  

In this guide, we will focus on the **User Perspective**, covering the following key areas:  

### **1. Building Basic LLM Applications**  
   - Open-source vs. Closed-source LLMs  
   - Utilizing LLM APIs  
   - LangChain Framework  
   - Hugging Face Integration  
   - Ollama for Local LLMs  

### **2. Enhancing LLM Responses**  
   - Prompt Engineering Techniques  
   - Retrieval-Augmented Generation (RAG)  
   - Fine-Tuning Models  

### **3. Latest Advancements**  
   - AI Agents  
   - LLMOps (Large Language Model Operations)  

### **4. Miscellaneous Topics**  

---

## **What is LangChain?**  

LangChain is an **open-source framework** designed to facilitate the development of applications powered by large language models (LLMs). It offers **modular components** and **end-to-end tools**, enabling developers to build sophisticated AI solutions, including:  
- Chatbots  
- Question-Answering Systems  
- Retrieval-Augmented Generation (RAG) Pipelines  
- Autonomous Agents  
- And many more  

### **Key Features of LangChain**  
1. **Broad LLM Compatibility** – Supports all major LLMs.  
2. **Simplified Development** – Streamlines the creation of LLM-based applications.  
3. **Extensive Integrations** – Works seamlessly with major AI tools and platforms.  
4. **Open-Source & Actively Maintained** – Free to use with continuous improvements.  
5. **Comprehensive Use Case Coverage** – Supports all major Generative AI applications.  

### **Why Learn LangChain First?**  
Learning LangChain provides exposure to a wide range of LLM functionalities, making it an ideal starting point for mastering AI application development.  

---

## **LangChain Course Structure**  

### **1. Fundamentals**  
   - Introduction to LangChain  
   - Core Components  
   - Models (LLM Integrations)  
   - Prompts & Prompt Templates  
   - Output Parsing  
   - Runnables & LCEL (LangChain Expression Language)  
   - Chains (Sequential Workflows)  
   - Memory (State Management)  

### **2. Retrieval-Augmented Generation (RAG)**  
   - Document Loaders  
   - Text Splitting Techniques  
   - Embeddings & Vector Representations  
   - Vector Databases  
   - Retrievers  
   - Building a RAG Application  

### **3. Agents**  
   - Tools & Toolkits  
   - Tool Calling  
   - Developing an AI Agent  

We will be learning the **latest version of LangChain (v0.3)** to ensure up-to-date knowledge.  


# 01 **What is LangChain**  

LangChain is an open-source framework for building applications that leverage large language models (LLMs). It helps developers connect LLMs to tools, APIs, and data sources, enabling intelligent, multi-step applications.

**Why Do We Need LangChain**  
LLMs alone can't access real-time information, retain conversation history, or interact with external systems. LangChain adds these capabilities by abstracting complex workflows, making it easier to build functional, intelligent applications with minimal custom code.

**Beginner-Friendly Example**  
Suppose you run an online bookstore and want to create a chatbot that:
- Recommends books
- Checks stock availability
- Places orders

LangChain enables this by combining an LLM with your inventory database and order system, coordinating everything through structured workflows.

**Core Concepts**

*Chains*  
Chains are sequences of steps where each component’s output becomes the next input. LangChain supports:
- Sequential Chains: Steps occur in order (e.g., Question → Search → Summarize → Answer).
- Parallel Chains: Tasks run simultaneously (e.g., Summarize and extract keywords at the same time).
- Conditional Chains: Flow is determined by input (e.g., route to different tools based on query type).

*Model-Agnostic Design*  
LangChain lets you easily switch LLM providers (OpenAI, Anthropic, etc.) with minimal code changes:
```python
llm = OpenAI()  # Switch to: llm = Anthropic()
```

*Memory and State Handling*  
LangChain supports memory modules to retain context across conversations, allowing your app to remember past interactions and user preferences.

*Ecosystem Highlights*  
LangChain includes:
- Prompt templates
- Integration with vector stores (e.g., Pinecone, Weaviate)
- Tool and API connectors
- Memory modules
- Agents for dynamic, multi-step decision-making

**What You Can Build**
| Application                | Example Use Case                            |
|----------------------------|---------------------------------------------|
| Conversational Chatbots    | Virtual assistants, support bots            |
| Knowledge Assistants       | Internal document Q&A tools                 |
| AI Agents                  | Multi-step task executors                   |
| Workflow Automation        | Data pipelines, report generation           |
| Summarization Tools        | Research digesters, meeting note creators   |

**Alternatives to LangChain**
- **LlamaIndex**: Great for retrieval-augmented generation (RAG) and document Q&A.
- **Haystack**: Suitable for building NLP pipelines with traditional and neural search.

LangChain is more flexible for complex, LLM-centered workflows, while alternatives may offer simpler solutions for specialized tasks.



# **02-LangChain Components Overview**

LangChain provides a modular framework for building applications with large language models (LLMs). It includes six foundational components that help structure, manage, and scale AI-powered workflows.

---

## **1. Models**

Models are the core interface to LLMs and embeddings.

### **Key Challenges**
- Natural language understanding (NLU)
- Context-aware generation
- Local deployment limitations due to model size (often >100GB)

### **LangChain’s Approach**
- Standardized **Model Component Interface**
- Easy switching between providers like OpenAI, Anthropic, or Ollama via APIs

### **Supported Model Types**
| Type               | Input → Output         | Use Case                         |
|--------------------|------------------------|----------------------------------|
| **Language Models** | Text → Text            | General text generation          |
| **Embedding Models** | Text → Vector         | Semantic search, similarity matching |

---

## **2. Prompts**

Prompts control how LLMs behave by shaping the input structure.

### **Common Prompting Techniques**

- **Dynamic and Reusable Templates**
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Summarize {topic} in a {tone} tone")
print(prompt.format(topic="cricket", tone="fun"))
```

- **Role-Based Prompts**
```python
from langchain.prompts import ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced {profession}"),
    ("user", "Tell me about {topic}")
])
formatted_messages = chat_prompt.format_messages(profession="Doctor", topic="headache")
```

- **Few-Shot Prompting**
```python
# Step 1: 
examples = [
    {"input": "I was charged twice for my subscription this month",  "output": "Billing"},
    {"input": "The App crashes everytime I try to log in.", "output": "Technical Support"},
    {"input":"Can you explain how to upgrade my plan?","output":"General Inquiry"},
    {"input":"I need a refund for a payment I didn't authorize","output":"Billing Issue"},
]

# Step 2: Create an example template
example_template = """
ticket: {input}
category: {output}

"""

# Step 3: build the few_shot_prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = PromptTemplate(input_variables=["input","output"], template=example_template),
    prefix = "Classify the following customer support tickets into one of the categories: 'Billing', 'Issue', 'Technical Problem', or 'General Inquiry'" 
    suffix = "Ticket: {user_input} Category: ",
    input_varibles = ["user_input"]
)

```

---

## **3. Chains**

Chains enable the creation of multi-step pipelines using LLMs and tools.

### **Chain Types**
| Type            | Flow Description                       | Example                                  |
|------------------|----------------------------------------|------------------------------------------|
| **Sequential**    | Step-by-step                           | Translate → Summarize → Output           |
| **Parallel**      | Tasks in parallel                      | Analyze sentiment + Extract keywords     |
| **Conditional**   | Input-based routing                    | Route medical questions to Doctor agent  |

Chains reduce complexity by modularizing logic and allow easy reuse and scaling of workflows.

![](../assets/2.0-seq.png)
![](../assets/2.1-para.png)
![](../assets/2.2-cond.png)

---

## **4. Indexes**

Indexes connect your application to external, unstructured data sources.

### **Core Components**
1. **Document Loaders** – Import content from PDFs, websites, databases
2. **Text Splitters** – Chunk large documents for LLM consumption
3. **Vector Stores** – Store vector embeddings for similarity search
4. **Retrievers** – Retrieve relevant chunks based on a query

*Example Use Case*: Upload manuals, split them into chunks, embed into a vector DB, and use semantic search to answer questions from users.

![](../assets/2.3-indexes.png)
---

## **5. Memory**

LangChain adds memory to track interaction history—solving the stateless limitation of LLMs.

### **Why Memory Matters**
Without memory:
```
User: Zardari is Pakistan’s president, age 80  
Follow-up: How old is Zardari?  
LLM: I don’t know.
```
With memory, the LLM recalls earlier context.

### **Memory Types**
| Type                         | Description                             |
|------------------------------|-----------------------------------------|
| `ConversationBufferMemory`   | Stores full conversation history        |
| `ConversationBufferWindowMemory` | Retains last *k* messages             |
| `ConversationSummaryMemory` | Summarizes conversation for efficiency  |
| **Custom Memory**            | Build your own context tracking logic   |

---

## **6. Agents**

Agents are autonomous systems that reason, decide, and use tools dynamically.

### **Key Capabilities**
- **Tool usage** (e.g., search, calculator, database query)
- **Multi-step reasoning** (e.g., decompose and solve)
- **Dynamic workflows** (adjust steps based on the situation)

### **Example: Support Agent**
1. Detects issue type  
2. Searches a knowledge base  
3. Initiates refund or escalates  

Agents allow LLMs to behave like decision-making systems, not just passive responders.

---

## **Summary Table**

| Component   | Purpose                                  | Example Use Case                            |
|-------------|-------------------------------------------|----------------------------------------------|
| Models      | Generate or embed text                    | Answering queries, semantic similarity       |
| Prompts     | Format instructions for the model         | Summarization, classification                |
| Chains      | Multi-step workflows                      | Translate + summarize                        |
| Indexes     | Access to external knowledge              | Searching PDFs, websites                     |
| Memory      | Maintain context                          | Remember user preferences                    |
| Agents      | Enable reasoning + tool usage             | Dynamic customer support                     |

# 03- Model Component

The Model Component in LangChain provides a standardized interface to interact with various AI models, making it easier to work across providers and model types. It abstracts away implementation details and enables developers to:

- Use different LLMs and embedding models through a unified API
- Build tools for text generation, semantic search, and dialogue systems
- Seamlessly switch between providers like OpenAI, Anthropic, and more


![](../assets/3.0-models-hierarchy.png)

![](../assets/3.1-source-hierarchy.png)


## Language Models Types
Language Models are AI Systems designed to process, generate, and understand natural language text.

### 1. Base Language Models (LLMs)
These are general-purpose models that take raw text as input and return text as output. They are stateless and designed for one-off tasks.
    - Stateless, single-turn interaction
    - Suitable for content transformation and batch tasks
    - No role or conversational context awareness

### 2.**Chat Models**:
Language models that are specialized for conversational tasks. They take a sequence of messages as input and returns chat messages as output (as opposed to using plain text). These are traditionally newer models and used more in comparision to the LLMs.


| Feature             | LLMs (Base Models)                                      | Chat Models (Instruction-Tuned)                                            |
|---------------------|---------------------------------------------------------|----------------------------------------------------------------------------|
| Purpose             | Free-form text generation                               | Optimized for multi-turn conversations                                     |
| Training Data       | General text corpora (books, articles)                  | Fine-tuned on chat datasets (dialogues, user-assistant conversations)      |
| Memory & Context    | No built-in memory                                      | Supports structured conversation history                                   |
| Role Awareness      | No understanding of roles like "user" or "assistant"    | Understands roles like "system", "user", and "assistant"                   |
| Example Models      | GPT-3, LLaMA-2-7B, OPT-1.3B                              | GPT-4, GPT-3.5 Turbo, LLaMA-2-Chat, Mistral-Instruct, Claude               |
| Use Cases           | Text generation, summarization, translation, code gen   | Conversational AI, chatbots, virtual assistants, customer support, AI tutors |



### Setting up Env: Using `pip` (Traditional)

1. **Create a new project folder**

2. **Open the folder in VS Code**

3. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   On PowerShell:

   ```bash
   .\venv\Scripts\Activate
   ```

5. **(Optional) If you encounter an execution policy error**, run:

   ```bash
   Set-ExecutionPolicy RemoteSigned -Scope Process
   ```

6. **Create a `requirements.txt` file** and add the necessary dependencies:

```py
    # LangChain Core
    langchain
    langchain-core

    # OpenAI Integration
    langchain-openai
    openai

    # Anthropic Integration
    langchain-anthropic

    # Google Gemini (PaLM) Integration
    langchain-google-genai
    google-generativeai

    # Hugging Face Integration
    langchain-huggingface
    transformers
    huggingface-hub

    # Environment Variable Management
    python-dotenv

    # Machine Learning Utilities
    numpy
    scikit-learn
```

7. **Install the packages**

   ```bash
   pip install -r requirements.txt
   ```

8. **Verify the installation**

   ```python
   import langchain
   print(langchain.__version__)
   ```


### With uv package manager

```py
pip install uv
uv venv venv 
```

### For later use | init project
```py
uv init porject_name
cd project_name
uv run main.py
uv add pandas
uv build
```

---

# Open Source Models
Open-source language models are freely available AI models that can be downloaded, modified, fine-tuned, and deployed without restrictions from a central provider. Unlike closed-source models such as OpenAI's GPT-4, Anthropic's Claude, or Google's Gemini, open-source models allow full control and customization.

|Feature | Open-Source Models | Closed-Source Models |
|--------|--------------------|----------------------|
|Cost    | Free-to-use (no API cost)| Paid API usage (e.g., OpenAI charges per token)|
| Control| Can modify, fine-tune, and deploy anywhere | Locked to provider's infrastructure|
| Data Privacy | Runs locally (no data sent to external servers) | Sends queries to provider's servers |
| Customization | Can fine-tune on specific datasets | No access to fine-tuning in most cases |
| Deployment | Can be deployed on *on-premise* servers or cloud | Must use vendor's API|


## Some Famous Open Source Models

| Model | Developer | Parameters | Best Use Case |
|-------|-----------|------------|---------------|
| LLaMa-2-7B/13B/70B | MetaAI | 7B - 70B | General Purpose text-generation |
| Mixtral-8x7B | Mistral AI | 8x7B (MoE) | Efficient & fast responses |
| Mistral-7B | Mistral AI | 7B | Best small-scale model (outperforms LLaMa-2-13B)|
| Falcon-7B/40B | Tll UAE | 7B - 40B | High Speed inference | 
| BLOOM-176B | BigScience | 176B | Multilingual text generation|
| GPT-J-6B | EleutherAI | 6B | Lightweight and Efficient |
| GPT-NeoX-20B | EleutherAI | 20B | Large-scale applications |
| StableLM | StableAI | 3B-7B | Compact models for chatbots |   


### Where to Find them?
HuggingFace - The largest repository of open-source LLMs.


### Ways to use open-source models?

1. Using HuggingFace inference API
2. Running Locally


### Disadvantages of Open-Source Models 

| Disadvantages | Details | 
|---------------|---------|
|High Hardware Requirements  | Running large models (e.g., LLaMA-2-70B) requires GPUs.|
| Setup Complexity | Requires installation of dependencies like `PyTorch`, `CUDA`, `trasnformers`|
| Lack of RLHF | Most open-source models don't have `fine-tuning with human feedback`, making then weaker in instruction following|
| Limited Multimodal Abilities | Open source models don't support `images`, `audio`, or `video` like GPT-4V.|

# 04 Prompts
Prompts are the input instructions or queries given to a model to guide its output.


- **Static vs Dynamic Prompts**

| Feature     | Static Prompts                         | Dynamic Prompts                                         |
| ----------- | -------------------------------------- | ------------------------------------------------------- |
| Definition  | Fixed text used directly in the prompt | Prompt content is generated or modified at runtime      |
| Flexibility | Not adaptable to input changes         | Adapts based on variables, context, or user input       |
| Use Case    | Simple, repeatable tasks               | Complex or personalized tasks                           |
| Example     | `"Translate this text to French:"`     | `f"Translate this to French: {user_input}"`             |
| Tools Used  | Plain strings                          | String formatting, f-strings, LangChain PromptTemplates |

Use static prompts for fixed instructions. Use dynamic prompts when prompts need to change based on context or input.

## Prompt Template
A Prompt Template in LangChain is a structured way to create prompts dynamically by inserting variables into a predefined template. Instead of `hardcoding prompts`, PromptTemplate allows you to defined palceholders that can be filled in runtime with different inputs.

This makes it resusable, flexible, and easy to manage, especially when working with dynamic user inputs or automated workflows(chains/pipelines).

### Why use Prompt Template over f strings?
1. Defualt validation 
    - `validation_template=True`
2. Reusable
    - `template.json`
3. LangChain Ecosystem
    - `chains`


## Messages

- **Problem**: If you pass the entire chat history as a simple list to a model, it's hard for the LLM to differentiate between ai responses and user inputs, particularly as the conversation lengthens. It's preferable to format the history using dictionaries with clear user and ai keys for every message. Luckily, LangChain comes with a pre-existing module named messages that automatically formats this for you.

```py
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break 
    result = model.invoke(user_input) # send the static prompt
    print("AI: ", result.content)

# Example:
# You: which one is bigger 3 or 7
# AI:  7 is bigger than 3.
# You: multiply bigger number with 10 and give me the result
# AI:  Please provide the bigger number.  I need a number to multiply by 10.
```

### Types of Messages in LangChain
1. SystemMessage
Used to set context, rules, or behavior for the model.
    - **Example**: "You are a helpful assistant."

2. HumanMessage
    - Represents input from the user.
    - This is what the user says or asks.

3. AIMessage
    - Represents responses generated by the model (AI).


## Chat Prompt Templates

![](../assets/4.1-invoke-method.png)

## Message Placeholder
A MessagePlaceholder in LangChain is a special placeholder used inside a ChatPromptTemplate to dynamically insert chat history of a list of messages at runtime.


# 05 Structured Outputs

By default the output is unstructured which is nothing but text.
In LangChain, structured output refers to the practice of having language models return responses in a well-defined data format (e.g., JSON) rather than free-form text. This makes the model output easier to parse and work with programmatically.

- `Prompt` - _Can you create a one day travel plan for paris?_
- `LLMs' Unstructured Response`:

```bash
Here's suggested plan: Morning: Visit the Eiffel Tower.
Afternoon: Walk through the Louvre Museum.
Evening: Enjoy dinner at a Seine riverside cafe.
```
- `JSON enforced output`:

```bash
[
    {"time":"Morning", "activity":"Visit the Eiffel Tower"},
    {"time":"Afternoon","activity":"Walk through the Louvre Museum"},
    {"time":"Evening","activity":"Enjoy dinner at a Seine riverside cafe"}
]
```

### Why do we need Structured Output?
- Data Extraction
Let's suppose we are building an app that extract the candidates' data from their resume:
    - Step 1: Extract the data
    - Step 2: Feed the data to LLM and get result in JSON
    - Step 3: Store the results in a database
- API building
Bulding a review analyzer for amazon like companies:
    - Extract the: topic, pros, cons, sentiment from the review (text input).
    - Build an API and give access to the results
- Agents
Agents need tools to perform tasks. Consider a query `Find the multiplication of 2 and 5`, we can't send the query to a tool like calculator in the given form rather we need to convert it into a JSON like format and then send it to the calculator like tool. These tools can't work with textual data.


### Ways to get Structured Output
1. With Structured Output
LLMs with built-in capability of generating structured output.
2. Without Structured Output
LLMs without built-in capability of generating structured output. This can be solved using output parsers of langchain.

In this module we'll be focusing on the first type `with_structured_output` generating, and in the next module we'll dicuss the output parsers. Usually we call the invoke method to generate the response, just before that we have to call the with_structure_output by specifying the data_format to get the structured output.

- **Three most common ways of format**:
    - TypedDict
    - Pydantic
    - json_schema


## TypedDict
TypedDict is a way to define a dictionary in Python where you specify what keys and vlaues should exist. It helps ensure that your dictionary follows a specific structure.

- **Why use TypedDict**
    - It tells Python what keys are required and what types of values they should have. It won't generate an error if you pass an string to a integer type variable.
    - It does not validate at runtime (it just helps with type hints for better coding).

- **Types of TypedDict**
    - Single TypedDict
    - Automated TypedDict
    - Literal
    - More Complex: With pros and cons

The issue with TypedDict is we can't have the validation rather it's just a representation technique. It means, you explicitly specifying the types of keys such as str for summary, sentiment, etc but there is no gaurantee that the LLM will always returns the reponse in str. So there comes the `Pydantic`.

## Pydantic
Pydantic is a data validation and data parsing library for Python. It ensures that the data you work with is correct, structured, and type-safe.

- **Agends**
    - Basic Example
    - Default values
    - Optional Fields
    - Type Coerce (implicitly)
    - Built-in validation
    - Field Functions: default values, constraints, description, regex
    - Expressions
    - Returns pydantic object: convert to json/dict


## JSON Schema
Sometimes developers design the frontend using JS and backed using Python in such cases the typedDict and Pydantic doesn't support. JSON Schema is a way to deal with such scenarios.


# 06 Output Parsers
Output Parsers in LangChain help convert RAW LLM responses into structured formats like JSON, CSV, Pydantic models and many more. They ensure consistency, validation, and ease of use in applications. 

By default the OpenAI, Anthropic, gemini or some other paid models support the structured output in most of the cases. But most of the open-source models doesn't support the structured output by default. 

- **Four most common output parsers**
    - StrOutputParser
    - JSONOutputParser
    - StructuredOutputParser
    - PydanticOutputParser


### StrOutputParser
The StrOutputParser is the simplest output parser in LangChain. It is used to parse the output of a Language Model (LLM) and returns it as a plain string.

### JSON OutputParser
Its dictionary like object.

```bash
    key:[
        value1,
        value2,
        ...
        value_n
    ]
```


It doesn't enforce a schema. for example:

```bash
    {
    fact1: value1,
    fact2: value2,
    ...
    fact_n:value_n
    }
```

The solution is StructuredOutputParser

### StructuredOutputParser
StructuredOutputParser is an output parser in LangChain that helps extract structured JSON data from LLM responses based on predefined field schemas.


It workds by defining a list of field (ResponseSchema) that the model should return, ensuring the output follows a structured format. 

It doesn't support data validation.

The solution is Pydantic Output Parser.

### Pydantic Output Parser
`PydanticOutputParser` is a structured output parser in LangChain that uses `Pydantic models` to enforce schema validation when processing LLM responses. 


# 07 Chains 🥹
TODO: Add Notes

# 08 Runnables

The compoenents didn't have a connecting module. To connect each module there were written sepearate chains/code that increased the volume of langchain's codebase. They realized their mistake and proposed a solution. 

Workflow

runnable: unit of work

Unit of work: Input process outptu
Common interface: invoke(), batch(), stream()

consider lago blocks

The ideas was to standardize all the components

Types of Runnables
1. Task Specific
2. Runnable Primitives


### Task Specific
These are core langchain compoenents that have been converted into runnables so they can be used  in pipelines. 
Purpose: Perform task-specific operations like LLM calls, prompting, retrieval, etc.
Ex:
    - ChatOpenAI: Runs an LLM Model
    - PromptTemplate: Formats prompts dynamically
    - Retrievar: retrieves relevant documents.


### Runnable Primitives
These are fundamental building blocks for structuring execution logic in AI workflows. They help orchestrate execution by defining different runnables interact (sequentially, in parallel, conditionally, etc)

Ex:
`Runnable Sequence`: Runs steps in order (| operator)
Runnable Parallel: Runs multiple steps simulatenenously
RunnableMap: Maps the same input across multiple functions.
RunnableBranch: implements conditional execution (if-else logic)
RunnableLambda: wraps custom Python functions into runnables
RunnablePassthrough: Just forwards input as output (acts as a placeholder )



## 1. RunnableSequence
RunnableSequence is a sequential chain of runnables in LangChain that executes each step one after another, passing the output of one step as the input to another.

It is useful when you need to compose multiple runnables together in a structured workflow.


## 2. RunnableParallel
RunnableParallel is a runnable primitive that allows multiple runnables to execute in parallel. 

Each runnable receives the same input and processes it independently, producing a dictionary of outputs. 

## 3. RunnablePassthrough
RunnablePassthrough is a special runnable primitive that simply returns the input as output without modifying it. 

Consider a scenario in `SequentialRunnable` having prompt1 `write down a joke` and prompt2 `write down the explanation of joke`. You will get the the explanation of joke but the joke wasn't printed. It means you only get the result of last prompt only. But what if I need the output of all prompts i.e., the joke with its explanation. This situation can be handled by RunnablePassthrough. 


## 4. RunnableLambda
RunnableLambda is a runnable primitive that allows you to apply custom Python functions within an AI pipeline. 

It acts as a middleware between deifferent AI components, enabling preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow. 

## 5. RunnableBranch (Conditional Chains)
RunnableBranch is a flow control component in LangChain that allows you to conditionally route input data to different chains or runnables based on custom logic.

It functions like an if/elif/else block for chains, where you define the set of condition functions, each associated with a runnable (e.g., LLM call, prompt chain, or tool). The first matching condition is executed. If no condition matches a default runnable is used (if provided).


## LCEL (LangChain Expression Language)
LangChain Expression Language (LCEL) is a syntax in LangChain that simplifies composing complex LLM chains and workflows. It allows developers to represent complex workflows with minimal code, offering features like parallel execution, streaming, and asynchronous support. 

LCEL uses a declarative approach, meaning you define what you want to happen rather than how. LangChain then optimizes the execution of these chains at runtime. 

LCEL utilizes a pipe operator (|) to connect different LangChain components in a sequence.


### 1. Component Integration
Use LCEL to link LLMs, prompts, and output parsers.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
llm = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | llm | parser
chain.invoke({"text": "Hello, how are you?"})
```

### 2. Stream and Async Support

Supports async and streaming by default.

```python
# Async
response = await chain.ainvoke({"text": "Good morning"})

# Streaming
async for chunk in chain.astream({"text": "Stream this"}):
    print(chunk, end="")
```

### 3. Parallelization

Run multiple chains at the same time.

```python
from langchain_core.runnables import RunnableParallel

chain1 = prompt | ChatOpenAI(model="gpt-3.5-turbo") | parser
chain2 = prompt | ChatOpenAI(model="gpt-4") | parser

parallel_chain = RunnableParallel(model1=chain1, model2=chain2)
parallel_chain.invoke({"text": "What is the capital of France?"})
```

### 4. Flexibility and Composability

Modify or reuse parts of the chain easily.

```python
# Reuse the prompt with different parsers or models
new_chain = prompt | ChatOpenAI() | StrOutputParser()
```

### 5. Unified Interface

All chains use the same interface methods.

```python
chain.invoke({"text": "Thanks"})         # Synchronous
await chain.ainvoke({"text": "Async"})   # Asynchronous
```

# 09 Document Loaders
Document loaders are components in LangChain used to load data from various sources into a standardized format (usually as Document object), which can then be used for chunking. 

```bash
page_content = "The actual text content",
meta_data = {"source":"filename.pdf",...}
```

![](../assets/9.1-RAG.png)

We will be dicussing the 4 most common document loaders also mentioned in the Figure above. 


## 1. TextLoader
TextLoader is a simple and commonly used document loader in LangChain that reads plain text(.txt) files and converts them into LangChain Document Objects.

Use case
- Ideal for loading chat logs, scraped text, transcripts, code snippets, or any plain text data into a LangChain pipeline.

- Limitation
Works only with .txt files


## 2. PyPDFLoader
PyPDFLoader is a document loader in LangChain used to load context from PDF files and convert each page into a Document object. 

```bash
[
    Document(page_content="Text from page 1", meta_data={"page":0,"srouce":"file.pdf"})
    Document(page_content="Text from page 2", meta_data={"page":1,"srouce":"file.pdf"})
]
```

- Limitations 
It uses the PyPDF library under the hood that's not suitable for scanned PDFs or complex layouts. There are some other PDF loaders available in langchain to tackle with this.

|       Use Case            | Recommended Loader            |
|---------------------------|-------------------------------|
|Simple Clean PDFs          | PyPDF Loader                  |
|PDFs with tables/columns   | PDFPlumberLoader              |
|Scanned/Images PDF         | UnstructuredPDFLoader, AmazomTextExtractorPDFLoader|
|Need layout and image data | PyMuPDFLoader                 |
|Want best structure extraction|UnstructuredPDFLoader       |

[Click Here to read more on LangChain Document Loader ](https://python.langchain.com/docs/concepts/document_loaders/)

## 3. DirectoryLoader
DirectoryLoader is a document loader that lets you load multiple documents from a directory of files.

| Global Pattern    | What is loads     |
|-------------------|-------------------|
| `**/*.txt`        |All .txt files in all subfolders|
|   `*.pdf`         |All .pdf files in the root directory|
|   `data/*.csv`    |All .csv files in the `data/` folder|
|   `**/*`          |All files (any type, all folders)   |


You will notice that loading a directory with 2-3 pdf files with overall around 1000 pages takes between 10 to 20 seconds and what if there are a 100 pdf? To process them you need to load all the PDFs in the RAM that will be time consuming and computational costly. Thankfully LangChain provide a solution to deal with it.

## Load vs Lazy Load
Usually we were using the `.load()` method to load the files but there is also another predefined method `.lazy_load()`. 

### load()
- Eager loading (loads everything at once)
- Returns: A list of Document objects
- Loads all documents immediately into memory.


## 4. WebBaseLoader
WebBaseLoader is a document loader in LangChain used to load and exract text content from web page (URLs).

It uses beautifulsoup under the hood to parse HTML and extract visible text.

- **When to use**
For blogs, news articles, or public websites where the content is primarily text based and static.

- **Limitations**
    - Doesn't handle javascript-heavy pages well (use SeleniumURLLoader for that).
    - Loads only static content (what's in the HTML, not what loads after the page renders).

## 5. CSVLoader
CSVLoader is a document loader used to load CSV files into LangChain Document objects, one per row by default.  


# 10 TextSplitters
Text splitting is the process of breaking large chunks of text (like articles, PDFs, HTML Pages, or books) into smaller, manageable pieces (chunks) that an LLM can handle effectively. 

- **Overcoming model limitation**: Many embedding models and language models that have maximum input size constraints. Splitting allows us to process documents, otherwise exceed these limits. 

- **Downstream tasks**: Text splitting improves nearly every LLM powered task

| Task | Why Splitting Helps |
|------|---------------------|
|Embedding | Sort chunks yield more accurate vectors|
|Semantic Serach | Search results point to focused info not noise|
|Summarization| Prevents hallucination and topic drift |

- **Optimizing computational resources**: Working with smaller chunks of text can be more memory-efficient and allow for better parallelization of processing tasks. 

![](../assets/9.2-text-splitters.png)

![Tool: https://chunkviz.up.railway.app/](https://chunkviz.up.railway.app/)



### 1. Length-Based Splitting
- **What**: Divides text after fixed character/token counts
- **Best for**: Uniform documents where structure matters less
- **Example**: 
  ```python
  # Splitting every 500 characters
  text = "Your long document..."
  chunks = [text[i:i+500] for i in range(0, len(text), 500)]
  ```
- **Pros**: Simple to implement, consistent chunk sizes
- **Cons**: Often breaks sentences/ideas mid-flow

### 2. Text Structure-Based Splitting
- **What**: Uses natural delimiters in the text
- **Best for**: Well-formatted content (Markdown, code, etc.)
- **Delimiters**:
  - `\n\n` - Paragraph breaks
  - `\n` - Line breaks
  - Headers (`#`, `##`) - Section boundaries
- **Example** (Markdown):
  ```python
  chunks = text.split("\n\n")  # Split by paragraphs
  ```
- **Pros**: Preserves logical groupings
- **Cons**: Requires consistent formatting

### 3. Document Structure-Based Splitting
- **What**: Leverages file format hierarchy
- **Best for**: PDFs, HTML, Word docs, presentations
- **Elements**:
  - PDF: Pages, sections, headings
  - HTML: `<div>`, `<p>`, `<h1>` tags
  - Word: Headers, footers, tables
- **Tools**:
  - `PyPDF2` (PDFs)
  - `BeautifulSoup` (HTML)
  - `python-docx` (Word)
- **Pros**: Maintains document semantics
- **Cons**: Parser-dependent, format-specific

### 4. Semantic Meaning-Based Splitting
- **What**: Splits at natural idea boundaries
- **Best for**: Research papers, complex reports
- **Methods**:
  - Sentence transformers
  - Topic modeling
  - LLM-assisted splitting
- **Example** (using NLTK):
  ```python
  from nltk.tokenize import sent_tokenize
  chunks = sent_tokenize(text)  # Split by sentences
  ```
- **Pros**: Preserves complete thoughts
- **Cons**: Computationally intensive

### Comparison Table

| Technique          | Speed  | Accuracy | Best Use Case               |
|--------------------|--------|----------|-----------------------------|
| Length-Based       | Fast   | Low      | Quick preprocessing         |
| Structure-Based    | Medium | Medium   | Formatted docs (Markdown)   |
| Document-Based     | Slow   | High     | PDFs/HTML/Office docs       |
| Semantic-Based     | Slowest| Highest  | Critical analysis tasks     |

**Pro Tip**: For RAG applications, combine structure-based initial splitting with semantic-based refinement for optimal results.


# 11 Vector
A vector store is a system desgined to store and retrieve data represented as numerical vectors. 

- **Characterisitcs**:
    - `Storage`: Ensures vectors and their associated metadata are retained, whether `in-memory` for quick lookups or `on-disk` for durability and large-scale use.
    - `Similarity Search`: Helps retrieve the vectors most similar to a query vector.
    - `Indexing`: Provide a data structure or method enables fast similarity searches on high-dimensional vectors (e.g., approximate nearest neighbor lookups).
    - `CRUD Operations`: Manage the lifecycle of data, adding new vectors, reading them, updating existing entries, removing outdated vectors.

- **Use Cases**:
    - Semantic Search
    - RAG
    - Recommender Systems
    - Image/Multimedia search



## Vector Store vs. Vector Database

### Vector Store

- Typically refers to a lightweight library or service that focuses on storing vectors (embeddings) and performing similarity search.

- May not include many traditional database features like transactions, rich query languages, or role-based access control.

- Ideal for prototyping, smaller-scale applications. 

- Examples: FAISS (where you store vectors and can query them by similarity, but you handle presistence and scaling seperately)

### Vector Database
A full-fledged database system designed to store and query vectors.

- Offers additional `database-like` features.
    - Distributed architecture for horizontal scaling
    - Durability and presistence (replication, backup/restore)
    - Metadata handling (schemas, filters)
    - Potential for ACID or near-ACID guarantees
    - Authentication/Authoriczation and more advanced security

Geared for production environments with significant scaling, large datasets.
- Ex: Milvus, Qdrant, Weaviate, Pinecone

A vector database is effectively a vector store with extra database features (e.g., clustering, scaling, security, metadata filtering, and durability)

## Vector Stores in LangChain
- Supported Stores: LangChain integrates with multiple vector stores (FAISS, Pinecone, Chroma, Qdrant, etc), giving you flexibility in scale, features, and deployment.

- Common interface: A uniform vector store API lets you swap out one backend (e.g., FAISS) for another (e.g., Pinecone) with minimal code changes.

```py
from_documents(...) or from_texts(...)
add_documents(...) or add_texts(...)
similarity_search(query, k=...)
```


- Metadata handling: Most vector stores in LangChain allow you to attach metadata (e.g., timestamps, authors) to each document, enabling filter-based retrieval.



## Chroma
Chroma is a lightweight, open-source vector database that is especially friendly for local development and small scalre to medium scale production. 

![](../assets/11.1-chroma-hierarchy.png)

### Exercises

1. Create LangChain documents
2. add documents
3. view documents
4. search documents
5. search with similarity score
6. meta-data filtering
7. update documents & view documents
8. delete document & view documents

