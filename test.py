import requests

json_data = "Blank"
response = requests.post('http://127.0.0.1:5000/chat', json={'input': 'Give me the list of consulants involved in the development of cross instance and food folk group functionalities'})

if response.status_code == 200:
    print("success")
if response.status_code == 200 and response.text.strip():
    json_data = response.json()

print(json_data)
