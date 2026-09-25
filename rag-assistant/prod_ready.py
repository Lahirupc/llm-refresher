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

def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500
) -> list[str]:
    
    """
    Production chunking with semantic as primary, recursive as fallback.
    """
    embeddings = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")
    
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )

            chunks = chunker.split_text(text)
            
            # Validate chunks aren't too large
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                # Fallback to recursive oversized chunks
                return _recursive_fallback(text, fallback_chunk_size)
                
            return chunks
        except Exception as e:
            print(f"Semantic chunking failed {e} using fallback")
            return _recursive_fallback(text, fallback_chunk_size)
        
    return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=50
  )
  return splitter.split_text(text)


# Usage
chunks = smart_chunker(document, use_semantic=True)
print(f'Created {len(chunks)} semantic chunks')
