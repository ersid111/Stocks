"""
Fyers Access Token Generator
============================

Run this FIRST, before running gtf_strategy_automation.py.

It opens the Fyers login page in your browser, takes the redirect URL you get
back, exchanges the auth_code for an access token, and saves that token to
`fyers_access_token.txt` in the current folder.

Usage:
    python generate_token.py
"""

import webbrowser

from fyers_apiv3 import fyersModel

# =========================================================================
# === 1. FILL IN YOUR DETAILS HERE ===
# =========================================================================

# Your App ID (e.g. "ABCDEFG-101")
CLIENT_ID = "AAAA1AAAAA-999"

# Your Secret Key (e.g. "H1234567")
SECRET_KEY = "1234567891"

# This must match your Fyers App settings exactly
REDIRECT_URI = "http://127.0.0.1"

TOKEN_FILE = "fyers_access_token.txt"

# =========================================================================
# === 2. SCRIPT TO GENERATE THE AUTH CODE & TOKEN ===
# =========================================================================


def main():
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",  # required for the V3 API
    )

    # 1. Generate the Auth Code URL and open it in the default browser
    auth_url = session.generate_authcode()
    print("Opening login page in your browser...")
    webbrowser.open(auth_url)

    # 2. Ask the user to paste back the redirect URL
    print("\n--- ACTION REQUIRED ---")
    print("1. Log in to Fyers in the browser window that just opened.")
    print("2. You will be redirected to a 'page not found' (http://127.0.0.1). This is NORMAL.")
    print("3. Copy the full URL from your browser's address bar.")
    print("4. Paste the full URL here and press Enter:")
    auth_code_url = input("Paste the full redirect URL: ")

    # 3. Extract the auth_code from the URL
    try:
        auth_code = auth_code_url.split("auth_code=")[1].split("&")[0]
        print(f"\nExtracted auth_code: {auth_code[:10]}...")
    except (IndexError, AttributeError):
        print("Error: Could not parse auth_code from the URL. Make sure you pasted the full URL.")
        print("Please run the script again.")
        return

    # 4. Exchange the auth_code for an access_token
    session.set_token(auth_code)
    response = session.generate_token()

    if "access_token" in response:
        access_token = response["access_token"]
        print("\n--- SUCCESS! ---")
        print("\nYour ACCESS_TOKEN is:")
        print(access_token)

        with open(TOKEN_FILE, "w") as f:
            f.write(access_token)
        print(f"\nToken has been saved to '{TOKEN_FILE}'")
    else:
        print("\n--- FAILED TO GENERATE TOKEN ---")
        print(f"Error: {response.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
