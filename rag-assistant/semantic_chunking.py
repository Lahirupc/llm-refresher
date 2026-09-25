from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from open_router import OpenRouterEmbeddings

load_dotenv()

embeddings = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")

document = """


                       USER AUTHENTICATION GUIDE (v1.0)

This guide outlines the standard practices, security requirements, and steps
for authenticating users within the application ecosystem.


1. PASSWORDS & SECURITY REQUIREMENTS
-------------------------------------------------------------------------------
To maintain account security, all user passwords must meet the following 
minimum criteria:
* Length: Minimum of 12 characters.
* Complexity: Must include at least:
  - One uppercase letter (A-Z)
  - One lowercase letter (a-z)
  - One numerical digit (0-9)
  - One special character (e.g., !, @, #, $, %, ^, &, *)
* Exclusions: Cannot contain common dictionary words, sequential numbers 
  (e.g., 12345), or personal information (e.g., your username).

2. SIGN-UP / REGISTRATION FLOW
-------------------------------------------------------------------------------
1. Navigate to the application registration page.
2. Enter a valid email address and a unique username.
3. Create a password that complies with the rules in Section 1.
4. Click "Register" or "Sign Up".
5. Check your email for a verification link. 
6. Click the link within 24 hours to activate your account.


3. SIGN-IN / LOGIN FLOW
-------------------------------------------------------------------------------
1. Navigate to the login screen.
2. Enter your registered email address (or username) and password.
3. Click "Log In".
4. If Multi-Factor Authentication (MFA) is enabled, you will be prompted 
   to enter a 6-digit verification code (see Section 4).
5. Upon successful validation, a secure session token will be issued, 
   and you will be redirected to the dashboard.

4. MULTI-FACTOR AUTHENTICATION (MFA)
-------------------------------------------------------------------------------
MFA adds an extra layer of protection to your account. 

Setup Instructions:
1. Go to Account Settings > Security.
2. Click "Enable Multi-Factor Authentication".
3. Use an authenticator app (e.g., Google Authenticator, Authy) to scan 
   the displayed QR code.
4. Enter the temporary 6-digit code shown in your app to confirm.
5. Save the provided backup recovery codes in a secure, offline location.

5. PASSWORD RESET & RECOVERY
-------------------------------------------------------------------------------
If you forget your password, follow these steps to securely regain access:
1. Click the "Forgot Password?" link on the login page.
2. Enter your registered email address.
3. You will receive an email containing a secure, one-time reset link.
4. Click the link (valid for 15 minutes) and enter a new password.
5. Log in using your new credentials.

6. TROUBLESHOOTING & ACCOUNT LOCKOUTS
-------------------------------------------------------------------------------
* Account Lockout: Entering the wrong password 5 consecutive times will 
  temporarily lock your account for 15 minutes to prevent unauthorized access.
* Expired Links: If your email verification or password reset link expires, 
  simply restart the process to request a new one.
* Lost MFA Device: If you lose access to your authenticator app, use one 
  of your saved backup recovery codes during the login prompt.

7. SUPPORT CONTACT
-------------------------------------------------------------------------------
For further technical assistance or account recovery issues, please contact 
our IT Helpdesk:
* Email: support@yourdomain.com
* Internal Extension: 4400


"""


def print_chunk_summary(label: str, chunks):
    print(f"\n{'=' * 80}")
    print(f"{label}")
    print(f"{'=' * 80}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} chars")

    for i, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.strip().split())
        preview = preview[:140] + "..." if len(preview) > 140 else preview
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(preview)


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(document)

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)
semantic_chunks = semantic_chunker.split_text(document)

# semantic_chunker = SentenceTransformersTokenTextSplitter(
#   chunk_overlap=50,
#   tokens_per_chunk=400
# )
# semantic_chunks = semantic_chunker.split_text(document)

recursive_vectorstore = Chroma.from_texts(
  texts=recursive_chunks,
  embedding=embeddings,
  collection_name="recursive_chunks",
)
semantic_vectorstore = Chroma.from_texts(
  texts=semantic_chunks,
  embedding=embeddings,
  collection_name="semantic_chunks",
)

# print("\n=== Recursive vs Semantic Chunking Comparison ===")
# print(f"Recursive chunk count: {len(recursive_chunks)}")
# print(f"Semantic chunk count: {len(semantic_chunks)}")
# print(f"Difference: {len(semantic_chunks) - len(recursive_chunks)} chunks")

# print_chunk_summary("RECURSIVE CHUNKING", recursive_chunks)
# print_chunk_summary("SEMANTIC CHUNKING", semantic_chunks)

print("\n=== Summary ===")
for method_name, chunks in [("Recursive", recursive_chunks), ("Semantic", semantic_chunks)]:
    print(f"{method_name}: {len(chunks)} chunks | avg {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} chars")




# Test queries
test_queries = [
    'How do I authenticate with OAuth2',
    'What happens when I hit the rate limit',
    'What should I do if I don\'t remember the password?'
]

def test_retrieval(query, vectorstore, name):
  print(f"\n=== {name} ===")
  print(f"Query: {query}")

  results = vectorstore.similarity_search(query, k=1)

  print(f"Retrieved: {results[0].page_content[:150]}")
  print(f"Metadata: {results[0].metadata}")
  

print(f"\n{'='*60}")
print("    Retrieval Tests    ")
print(f"\n{'='*60}")

for query in test_queries:
  print("="*60)
  recursive_result = test_retrieval(query, recursive_vectorstore, 'RECURSIVE')
  semantic_result = test_retrieval(query, semantic_vectorstore, 'SEMANTIC')
    