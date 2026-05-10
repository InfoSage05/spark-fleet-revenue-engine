"""
generate_zoho_token.py

A helper script to generate your Zoho CRM Refresh Token from scratch.
"""

import os
import requests

def main():
    print("=== Zoho Refresh Token Generator ===\n")
    
    # 1. Gather credentials
    client_id = input("Enter your ZOHO_CLIENT_ID: ").strip()
    client_secret = input("Enter your ZOHO_CLIENT_SECRET: ").strip()
    
    print("\nWhich Zoho domain are you using?")
    print("1. zoho.com (US / Global)")
    print("2. zoho.in (India)")
    print("3. zoho.eu (Europe)")
    print("4. zoho.com.au (Australia)")
    domain_choice = input("Select an option (1-4) [Default: 1]: ").strip()
    
    domain_map = {
        "1": "zoho.com",
        "2": "zoho.in",
        "3": "zoho.eu",
        "4": "zoho.com.au"
    }
    zoho_domain = domain_map.get(domain_choice, "zoho.com")
    accounts_url = f"https://accounts.{zoho_domain}"
    
    # 2. Generate the Authorization URL
    # We are requesting offline access to the Leads module so we get a refresh token.
    scope = "ZohoCRM.modules.leads.ALL"
    redirect_uri = "http://localhost"
    
    auth_url = (
        f"{accounts_url}/oauth/v2/auth?"
        f"scope={scope}&"
        f"client_id={client_id}&"
        f"response_type=code&"
        f"access_type=offline&"
        f"redirect_uri={redirect_uri}"
    )
    
    print("\n" + "="*50)
    print("STEP 1: Open the following URL in your web browser:")
    print("="*50)
    print(auth_url)
    print("\n* You will be asked to log in to Zoho and click 'Accept'.")
    print("* After you click Accept, your browser will redirect you to an error page (or a blank page) starting with http://localhost/?code=...")
    print("* Look at the URL bar in your browser. Copy the long code right after 'code='")
    
    # 3. Wait for the user to paste the code
    print("\n" + "="*50)
    auth_code = input("STEP 2: Paste the 'code' from the URL here: ").strip()
    
    if not auth_code:
        print("No code entered. Exiting.")
        return
        
    # 4. Exchange the code for a refresh token
    print("\nExchanging code for tokens...")
    token_url = f"{accounts_url}/oauth/v2/token"
    
    payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code": auth_code
    }
    
    response = requests.post(token_url, data=payload)
    
    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            print("\n❌ Error from Zoho:")
            print(data)
        else:
            print("\n✅ SUCCESS! Here are your tokens:")
            print("-" * 40)
            print(f"Access Token  : {data.get('access_token')}")
            print(f"Refresh Token : {data.get('refresh_token')}")
            print("-" * 40)
            print("\nNext Step:")
            print("Copy the Refresh Token above and paste it into your .env file as ZOHO_REFRESH_TOKEN.")
    else:
        print(f"\n❌ Failed to connect to Zoho. Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
