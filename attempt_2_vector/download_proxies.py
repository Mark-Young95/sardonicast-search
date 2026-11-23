# import requests

# url = "https://proxy.webshare.io/api/v2/proxy/list/download/xlkhoaxepcudaxabqnfxdyfnxljmlpcxlsqkspgv/-/any/username/direct/-/?plan_id=14038384"

# response = requests.get(url)

# if response.status_code == 200:
#     with open("proxies.txt", "w", encoding="utf-8") as f:
#         f.write(response.text)
#     print("✅ Successfully downloaded proxies to proxies.txt")
# else:
#     print(f"❌ Failed to download proxies. Status code: {response.status_code}")





# import requests

# # Test using Webshare
# proxy_url = "http://jgrprhtq:pwy6904jczh0@p.webshare.io:80"  # or specific proxy from proxies.txt

# proxies = {
#     "http": proxy_url,
#     "https": proxy_url
# }

# try:
#     res = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
#     print("✅ Proxy connected! IP seen by target:", res.json()["ip"])
# except Exception as e:
#     print("❌ Proxy connection failed:", e)





import requests

# Fill in your exact Webshare PROXY credentials (not your account login)
PROXY_USER = "jgrprhtq"
PROXY_PASS = "pwy6904jczh0"
IP_ADDR = "31.59.20.176"
IP_PORT = "6754"

# Format for Webshare backconnect proxy
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{IP_ADDR}:{IP_PORT}/"

proxies = {
    "http": proxy_url,
    "https": proxy_url
}

try:
    res = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
    print("✅ Success! Proxy IP seen by destination:", res.json()["ip"])
except Exception as e:
    print("❌ Failed:", e)





# curl --proxy "http://jgrprhtq:pwy6904jczh0@31.59.20.176:6754/" https://ipv4.webshare.io/



# import requests
# requests.get(
#     "https://ipv4.webshare.io/",
#     proxies={
#         "http": "http://jgrprhtq:pwy6904jczh0@31.59.20.176:6754/",
#         "https": "http://jgrprhtq:pwy6904jczh0@31.59.20.176:6754/"
#     }
# ).text